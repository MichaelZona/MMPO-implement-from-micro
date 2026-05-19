from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import os
import time
from tqdm import tqdm

import evaluate
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, PreTrainedModel
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import TrainOutput
from transformers.utils import PaddingStrategy
from trl import RewardConfig, RewardTrainer
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
accelerator = Accelerator()

DEFAULT_HELPSTEER_ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
ULTRAFEEDBACK_ATTRIBUTES = [
    "ultrafeedback-helpfulness",
    "ultrafeedback-honesty",
    "ultrafeedback-instruction-following",
    "ultrafeedback-truthfulness",
]
ALL_KNOWN_ATTRIBUTES = DEFAULT_HELPSTEER_ATTRIBUTES + ULTRAFEEDBACK_ATTRIBUTES
ATTRIBUTE_NAME_TO_ID = {name: idx for idx, name in enumerate(ALL_KNOWN_ATTRIBUTES)}
ATTRIBUTE_ID_TO_NAME = {idx: name for name, idx in ATTRIBUTE_NAME_TO_ID.items()}
CYCLIC_ULTRAFEEDBACK_ATTRIBUTE_ALIASES = {
    "helpfulness": "ultrafeedback-helpfulness",
    "honesty": "ultrafeedback-honesty",
    "instruction_following": "ultrafeedback-instruction-following",
    "truthfulness": "ultrafeedback-truthfulness",
}


@dataclass
class ScriptArguments:
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-3
    num_train_epochs: int = 1
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    max_length: int = 1024
    max_prompt_length: int = 512
    base_model: str = "Qwen/Qwen3-0.6B"
    reference_model: Optional[str] = None
    wandb_name: str = "mmpo_em_only"
    log_dir: str = "./output_models"
    freeze_pretrained: bool = True
    data_path: str = "cyclic_ultrafeedback_all_pairs"
    num_heads: int = 4
    downsample_rate: float = 1.0
    eval_only: bool = False
    manual_seed: int = 42
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 10
    run_name: str = "mmpo-em-only"

    # EM-only controls (MixtureEMDPOTrainer style).
    em_temperature: float = 1.0
    m_step_updates: int = 1
    use_contextual_router: bool = False
    use_closed_form_router_prior_update: bool = True
    use_wandb: bool = True
    save_checkpoint: bool = True
    beta: float = 0.1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
torch.manual_seed(script_args.manual_seed)

# Normalize save flags early so downstream Trainer/EM hooks cannot save checkpoints.
if not script_args.save_checkpoint:
    script_args.save_strategy = "no"
    script_args.save_steps = 0

if script_args.downsample_rate <= 0 or script_args.downsample_rate > 1:
    raise ValueError("`downsample_rate` must be in (0, 1].")
if script_args.em_temperature <= 0:
    raise ValueError("`em_temperature` must be positive.")
if script_args.beta <= 0:
    raise ValueError("`beta` must be positive.")
if script_args.m_step_updates < 1:
    raise ValueError("`m_step_updates` must be >= 1.")

if accelerator.is_main_process:
    print("Arguments:")
    for arg_name in vars(script_args):
        print(format(arg_name, "<34"), format(str(getattr(script_args, arg_name)), "<"))


def resolve_local_dataset_dir(dataset_name: str) -> str:
    candidates = [
        Path("semi-reward-models/dataset") / dataset_name,
        Path("data_process/dataset") / dataset_name,
        Path("dataset") / dataset_name,
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"Could not find local dataset directory for '{dataset_name}'. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def attribute_to_id(attribute_name: Optional[str]) -> int:
    if attribute_name is None:
        return -1
    name = str(attribute_name).strip()
    return ATTRIBUTE_NAME_TO_ID.get(name, -1)


def _truncate_prompt(input_ids: List[int], max_prompt_length: int) -> List[int]:
    if len(input_ids) <= max_prompt_length:
        return input_ids
    return input_ids[-max_prompt_length:]


def _response_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict) and "content" in response:
        return str(response["content"])
    return str(response)


def _prompt_to_messages(prompt: Any) -> List[Dict[str, str]]:
    if isinstance(prompt, list):
        messages: List[Dict[str, str]] = []
        for item in prompt:
            if isinstance(item, dict) and "role" in item and "content" in item:
                messages.append({"role": str(item["role"]), "content": str(item["content"])})
        if messages:
            return messages
    return [{"role": "user", "content": str(prompt)}]


def _parse_lora_target_modules(raw: str) -> List[str]:
    modules = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not modules:
        raise ValueError("`lora_target_modules` must contain at least one module name.")
    return modules


def _tokenize_listwise_example(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    prompt_messages = _prompt_to_messages(example["prompt"])
    prompt_template = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_template, add_special_tokens=False)["input_ids"]
    prompt_ids = _truncate_prompt(prompt_ids, script_args.max_prompt_length)

    responses = example["responses"]
    scores = example.get("scores")
    if scores is not None and len(scores) == len(responses):
        order = sorted(range(len(responses)), key=lambda idx: float(scores[idx]), reverse=True)
        responses = [responses[idx] for idx in order]

    candidate_input_ids: List[List[int]] = []
    candidate_attention_masks: List[List[int]] = []
    candidate_response_masks: List[List[int]] = []
    for response in responses:
        response_text = _response_to_text(response)
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
            response_ids = response_ids + [eos_id]
        response_start = len(prompt_ids)
        input_ids = prompt_ids + response_ids
        if len(input_ids) > script_args.max_length:
            overflow = len(input_ids) - script_args.max_length
            if overflow < len(prompt_ids):
                new_prompt = prompt_ids[overflow:]
                input_ids = new_prompt + response_ids
                response_start = len(new_prompt)
            else:
                input_ids = response_ids[overflow - len(prompt_ids) :]
                response_start = 0
        candidate_input_ids.append(input_ids)
        candidate_attention_masks.append([1] * len(input_ids))
        candidate_response_masks.append([0] * response_start + [1] * (len(input_ids) - response_start))

    return {
        "candidate_input_ids": candidate_input_ids,
        "candidate_attention_mask": candidate_attention_masks,
        "candidate_response_mask": candidate_response_masks,
        "preference_dimension": example.get("preference_dimension"),
        "attribute_id": attribute_to_id(CYCLIC_ULTRAFEEDBACK_ATTRIBUTE_ALIASES.get(example.get("preference_dimension"))),
    }


def load_cyclic_ultrafeedback_listwise_split(split: str, tokenizer: AutoTokenizer) -> Dataset:
    dataset_dir = Path(resolve_local_dataset_dir("cyclic_ultrafeedback_all_pairs")) / split
    rows = []
    for shard_path in sorted(dataset_dir.glob("data-*.arrow")):
        with pa.memory_map(str(shard_path), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                rows.extend(batch.to_pylist())
    if not rows:
        raise ValueError(f"No listwise rows built from {dataset_dir}")
    dataset = Dataset.from_list(rows)
    dataset = dataset.map(lambda ex: _tokenize_listwise_example(ex, tokenizer), batched=False, num_proc=10)
    dataset = dataset.filter(
        lambda x: len(x["candidate_input_ids"]) >= 2
        and all(len(ids) <= script_args.max_length for ids in x["candidate_input_ids"])
        and all(sum(mask) > 0 for mask in x["candidate_response_mask"]),
        num_proc=10,
    )
    if script_args.downsample_rate < 1.0 and split == "train":
        keep_count = max(1, int(len(dataset) * script_args.downsample_rate))
        dataset = dataset.shuffle(seed=script_args.manual_seed).select(range(keep_count))
    return dataset


@dataclass
class ListwiseRewardCollator:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_candidates = max(len(feature["candidate_input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

        candidate_mask_rows: List[List[int]] = []
        batch_ids: List[List[List[int]]] = []
        batch_attn: List[List[List[int]]] = []
        batch_resp: List[List[List[int]]] = []

        max_seq_len = 1
        for feature in features:
            ids_list = feature["candidate_input_ids"]
            max_seq_len = max(max_seq_len, max(len(seq) for seq in ids_list))

        for feature in features:
            ids_list = list(feature["candidate_input_ids"])
            attn_list = list(feature["candidate_attention_mask"])
            resp_list = list(feature["candidate_response_mask"])
            valid_count = len(ids_list)
            candidate_mask_rows.append([1] * valid_count + [0] * (max_candidates - valid_count))
            for _ in range(max_candidates - valid_count):
                ids_list.append([])
                attn_list.append([])
                resp_list.append([])
            padded_item_ids: List[List[int]] = []
            padded_item_attn: List[List[int]] = []
            padded_item_resp: List[List[int]] = []
            for ids, attn, resp in zip(ids_list, attn_list, resp_list):
                pad_len = max_seq_len - len(ids)
                padded_item_ids.append(ids + [pad_id] * pad_len)
                padded_item_attn.append(attn + [0] * pad_len)
                padded_item_resp.append(resp + [0] * pad_len)
            batch_ids.append(padded_item_ids)
            batch_attn.append(padded_item_attn)
            batch_resp.append(padded_item_resp)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
            "response_mask": torch.tensor(batch_resp, dtype=torch.long),
            "candidate_mask": torch.tensor(candidate_mask_rows, dtype=torch.bool),
            "attribute_id": torch.tensor([f.get("attribute_id", -1) for f in features], dtype=torch.long),
            "preference_dimension": [f.get("preference_dimension", "unknown") for f in features],
            "return_loss": True,
        }


def pl_log_prob(
    rewards: torch.Tensor,
    rankings: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ranked = rewards.gather(1, rankings)
    if candidate_mask is None:
        ranked_mask = torch.ones_like(ranked, dtype=torch.bool)
    else:
        ranked_mask = candidate_mask.gather(1, rankings).bool()
    out = torch.zeros(rewards.shape[0], device=rewards.device)
    m_items = rewards.shape[1]
    for t in range(m_items - 1):
        suffix_mask = ranked_mask[:, t:]
        logits = ranked[:, t:].masked_fill(~suffix_mask, float("-inf"))
        term = ranked[:, t] - torch.logsumexp(logits, dim=1)
        out = out + torch.where(ranked_mask[:, t], term, torch.zeros_like(term))
    return out


def mixture_pl_nll(
    router_logits: torch.Tensor,
    rewards: torch.Tensor,
    rankings: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    log_alpha = F.log_softmax(router_logits, dim=1)
    _, k_clusters, _ = rewards.shape
    comp_logp = []
    for c in range(k_clusters):
        comp_logp.append(pl_log_prob(rewards[:, c, :], rankings, candidate_mask=candidate_mask))
    comp_logp = torch.stack(comp_logp, dim=1)
    mix_logp = torch.logsumexp(log_alpha + comp_logp, dim=1)
    if candidate_mask is not None:
        valid_rows = candidate_mask.sum(dim=1) >= 2
        if not torch.any(valid_rows):
            return rewards.new_zeros(()), comp_logp
        mix_logp = mix_logp[valid_rows]
    nll = -mix_logp.mean()
    return nll, comp_logp


def em_responsibilities(router_logits: torch.Tensor, comp_logp: torch.Tensor, temperature: float) -> torch.Tensor:
    log_alpha = F.log_softmax(router_logits, dim=1)
    joint = log_alpha + comp_logp
    return F.softmax(joint / temperature, dim=1)


def em_expected_complete_nll(
    router_logits: torch.Tensor,
    comp_logp: torch.Tensor,
    gamma: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    log_alpha = F.log_softmax(router_logits, dim=1)
    expected_joint = (gamma * (log_alpha + comp_logp)).sum(dim=1)
    if valid_mask is not None:
        if not torch.any(valid_mask):
            return router_logits.new_zeros(())
        expected_joint = expected_joint[valid_mask]
    return -expected_joint.mean()


def listwise_metrics(utilities: torch.Tensor, candidate_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    max_idx = utilities.argmax(dim=1)
    top1_acc = torch.where(
        candidate_mask[:, 0],
        (max_idx == 0).float(),
        torch.zeros_like(max_idx, dtype=torch.float),
    ).mean()
    upper_tri = torch.triu(
        torch.ones((utilities.size(1), utilities.size(1)), device=utilities.device, dtype=torch.bool),
        diagonal=1,
    )
    pairwise_margin = utilities.unsqueeze(2) - utilities.unsqueeze(1)
    valid_pairs = candidate_mask.unsqueeze(2) & candidate_mask.unsqueeze(1) & upper_tri.unsqueeze(0)
    correct_pairs = (pairwise_margin >= 0) & valid_pairs
    pairwise_acc = correct_pairs.sum().float() / valid_pairs.sum().clamp_min(1).float()
    valid_counts = candidate_mask.sum(dim=1)
    valid_rows = valid_counts > 0
    if torch.any(valid_rows):
        last_indices = valid_counts[valid_rows] - 1
        row_indices = torch.arange(last_indices.shape[0], device=utilities.device)
        utility_last = utilities[valid_rows][row_indices, last_indices].mean()
    else:
        utility_last = utilities.new_zeros(())
    utility_first = torch.where(candidate_mask[:, 0], utilities[:, 0], torch.zeros_like(utilities[:, 0])).mean()
    return {
        "listwise/top1_acc": top1_acc,
        "listwise/pairwise_acc": pairwise_acc,
        "listwise/utility_first": utility_first,
        "listwise/utility_last": utility_last,
        "listwise/utility_mean": utilities.mean(),
    }


def compute_cluster_assignment_metrics(
    assignment_scores: torch.Tensor,
    attribute_ids: torch.Tensor,
    num_clusters: int,
) -> Dict[str, float]:
    valid_mask = attribute_ids >= 0
    if not torch.any(valid_mask):
        return {}
    pred = assignment_scores.argmax(dim=1)[valid_mask]
    true = attribute_ids[valid_mask]
    if pred.numel() == 0:
        return {}
    raw_acc = (pred == true).float().mean().item()
    k = int(max(num_clusters, int(pred.max().item()) + 1, int(true.max().item()) + 1))
    confusion = torch.zeros((k, k), dtype=torch.long, device=pred.device)
    for t, p in zip(true.view(-1), pred.view(-1)):
        confusion[t.long(), p.long()] += 1
    cost = confusion.max().item() - confusion.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    pred_to_true = {int(col): int(row) for row, col in zip(row_ind, col_ind)}
    aligned_pred = torch.tensor([pred_to_true.get(int(p.item()), int(p.item())) for p in pred], device=pred.device)
    aligned_acc = (aligned_pred == true).float().mean().item()
    return {"mixture/cluster_acc": aligned_acc, "mixture/cluster_acc_raw": raw_acc}


@dataclass
class ClusterAlignment:
    true_to_pred: Dict[int, int]
    pred_to_true: Dict[int, int]


class RankingMetricAccumulator:
    def __init__(self) -> None:
        self.total_rows = 0
        self.top1_correct = 0
        self.pairwise_correct = 0
        self.pairwise_total = 0
        self.by_dimension = defaultdict(lambda: {"rows": 0, "top1": 0, "pairwise_correct": 0, "pairwise_total": 0})

    def add_batch(self, utilities: torch.Tensor, candidate_mask: torch.Tensor, dimensions: List[str]) -> None:
        utilities = utilities.detach().float().cpu()
        candidate_mask = candidate_mask.detach().bool().cpu()
        for row_idx, dimension in enumerate(dimensions):
            valid_indices = torch.where(candidate_mask[row_idx])[0].tolist()
            if len(valid_indices) < 2:
                continue
            row_utilities = utilities[row_idx]
            pred_top = max(valid_indices, key=lambda idx: float(row_utilities[idx]))
            top1_correct = int(pred_top == 0)
            pairwise_correct = 0
            pairwise_total = 0
            for left_pos, left_idx in enumerate(valid_indices):
                for right_idx in valid_indices[left_pos + 1 :]:
                    pairwise_total += 1
                    if row_utilities[left_idx] > row_utilities[right_idx]:
                        pairwise_correct += 1
            self.total_rows += 1
            self.top1_correct += top1_correct
            self.pairwise_correct += pairwise_correct
            self.pairwise_total += pairwise_total
            dim_stats = self.by_dimension[str(dimension)]
            dim_stats["rows"] += 1
            dim_stats["top1"] += top1_correct
            dim_stats["pairwise_correct"] += pairwise_correct
            dim_stats["pairwise_total"] += pairwise_total

    def metrics(self, prefix: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            f"{prefix}/num_examples": float(self.total_rows),
            f"{prefix}/top1_acc": self.top1_correct / max(self.total_rows, 1),
            f"{prefix}/pairwise_acc": self.pairwise_correct / max(self.pairwise_total, 1),
        }
        dim_top1_values = []
        dim_pairwise_values = []
        for dimension, stats in sorted(self.by_dimension.items()):
            dim_top1 = stats["top1"] / max(stats["rows"], 1)
            dim_pairwise = stats["pairwise_correct"] / max(stats["pairwise_total"], 1)
            dim_top1_values.append(dim_top1)
            dim_pairwise_values.append(dim_pairwise)
            dim_key = _metric_dimension_key(dimension)
            metrics[f"{prefix}/by_dimension/{dim_key}/num_examples"] = float(stats["rows"])
            metrics[f"{prefix}/by_dimension/{dim_key}/top1_acc"] = dim_top1
            metrics[f"{prefix}/by_dimension/{dim_key}/pairwise_acc"] = dim_pairwise
        metrics[f"{prefix}/macro_top1_acc"] = float(np.mean(dim_top1_values)) if dim_top1_values else 0.0
        metrics[f"{prefix}/macro_pairwise_acc"] = float(np.mean(dim_pairwise_values)) if dim_pairwise_values else 0.0
        return metrics


def _metric_dimension_key(dimension: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(dimension)).strip("_") or "unknown"


def _dimension_mapping(ranking_dataset: Dict[str, Dataset]) -> Dict[str, int]:
    dimensions = set()
    for split in ranking_dataset.values():
        if "preference_dimension" in split.column_names:
            dimensions.update(str(dimension) for dimension in split["preference_dimension"])
    return {dimension: idx for idx, dimension in enumerate(sorted(dimensions))}


def _cluster_alignment(true_labels: List[int], pred_labels: List[int], num_clusters: int) -> ClusterAlignment:
    if not true_labels or not pred_labels:
        return ClusterAlignment(true_to_pred={}, pred_to_true={})
    confusion = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label < num_clusters and pred_label < num_clusters:
            confusion[true_label, pred_label] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    true_to_pred = {int(true_idx): int(pred_idx) for true_idx, pred_idx in zip(row_ind, col_ind)}
    pred_to_true = {int(pred_idx): int(true_idx) for true_idx, pred_idx in zip(row_ind, col_ind)}
    return ClusterAlignment(true_to_pred=true_to_pred, pred_to_true=pred_to_true)


def _comb2(value: int) -> float:
    return value * (value - 1) / 2.0


def _ari(true_labels: List[int], pred_labels: List[int]) -> float:
    if len(true_labels) < 2:
        return 0.0
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)
    pair_counts = Counter(zip(true_labels, pred_labels))
    sum_comb = sum(_comb2(count) for count in pair_counts.values())
    true_comb = sum(_comb2(count) for count in true_counts.values())
    pred_comb = sum(_comb2(count) for count in pred_counts.values())
    total_comb = _comb2(len(true_labels))
    expected = true_comb * pred_comb / max(total_comb, 1.0)
    max_index = 0.5 * (true_comb + pred_comb)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 0.0
    return (sum_comb - expected) / denom


def _nmi(true_labels: List[int], pred_labels: List[int]) -> float:
    if not true_labels:
        return 0.0
    total = len(true_labels)
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)
    pair_counts = Counter(zip(true_labels, pred_labels))
    mi = 0.0
    for (true_label, pred_label), count in pair_counts.items():
        joint = count / total
        true_prob = true_counts[true_label] / total
        pred_prob = pred_counts[pred_label] / total
        mi += joint * math.log(joint / max(true_prob * pred_prob, 1e-12))
    h_true = -sum((count / total) * math.log(count / total) for count in true_counts.values())
    h_pred = -sum((count / total) * math.log(count / total) for count in pred_counts.values())
    denom = math.sqrt(h_true * h_pred)
    return mi / denom if denom > 1e-12 else 0.0


def evaluate_ranking_split(
    trainer,
    split_dataset: Dataset,
    dimension_to_id: Dict[str, int],
    cluster_alignment: Optional[ClusterAlignment] = None,
    collect_only_cluster_state: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    processing_class = getattr(trainer, "processing_class", None)
    if processing_class is None:
        processing_class = getattr(trainer, "tokenizer", None)
    if processing_class is None:
        raise ValueError("Trainer is missing processing_class/tokenizer for ranking evaluation.")
    dataloader = DataLoader(
        split_dataset,
        batch_size=script_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=ListwiseRewardCollator(tokenizer=processing_class, max_length=script_args.max_length),
    )

    policy_acc = RankingMetricAccumulator()
    posterior_acc = RankingMetricAccumulator()
    aligned_acc = RankingMetricAccumulator()
    true_labels: List[int] = []
    posterior_pred_labels: List[int] = []
    gamma_entropy_sum = 0.0
    gamma_count = 0
    mixture_nll_sum = 0.0
    mixture_nll_count = 0

    trainer.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = trainer._prepare_inputs(batch)
            candidate_mask = inputs["candidate_mask"]
            dimensions = [str(dim) for dim in inputs.get("preference_dimension", ["unknown"] * candidate_mask.shape[0])]

            mixture_output = trainer._get_batch_mixture_output(trainer.model, inputs)
            rankings = torch.arange(candidate_mask.shape[1], device=candidate_mask.device).unsqueeze(0).expand_as(candidate_mask).long()
            nll, comp_logp = mixture_pl_nll(
                mixture_output["router_logits"],
                mixture_output["rewards"],
                rankings,
                candidate_mask=candidate_mask,
            )
            gamma = em_responsibilities(
                mixture_output["router_logits"],
                comp_logp,
                temperature=script_args.em_temperature,
            )
            posterior_pred = gamma.argmax(dim=1)
            true_cluster = torch.tensor(
                [dimension_to_id.get(dimension, -1) for dimension in dimensions],
                dtype=torch.long,
                device=posterior_pred.device,
            )
            valid_dim_mask = true_cluster >= 0
            true_labels.extend(int(label) for label in true_cluster[valid_dim_mask].detach().cpu().tolist())
            posterior_pred_labels.extend(int(label) for label in posterior_pred[valid_dim_mask].detach().cpu().tolist())
            gamma_entropy = -(gamma * torch.log(gamma.clamp_min(1e-12))).sum(dim=1)
            gamma_entropy_sum += float(gamma_entropy.sum().detach().cpu().item())
            gamma_count += int(gamma_entropy.numel())
            mixture_nll_sum += float(nll.detach().cpu().item()) * int(candidate_mask.shape[0])
            mixture_nll_count += int(candidate_mask.shape[0])

            policy_rewards = mixture_output["rewards"].mean(dim=1)
            if not collect_only_cluster_state:
                policy_acc.add_batch(policy_rewards, candidate_mask, dimensions)

            if collect_only_cluster_state:
                continue

            row_indices = torch.arange(gamma.shape[0], device=gamma.device)
            posterior_rewards = mixture_output["rewards"][row_indices, posterior_pred, :]
            posterior_acc.add_batch(posterior_rewards, candidate_mask, dimensions)

            if cluster_alignment is not None:
                aligned_components = []
                for true_id, pred_id in zip(true_cluster.detach().cpu().tolist(), posterior_pred.detach().cpu().tolist()):
                    if true_id < 0:
                        aligned_components.append(pred_id)
                    else:
                        aligned_components.append(cluster_alignment.true_to_pred.get(int(true_id), int(pred_id)))
                aligned_components_tensor = torch.tensor(aligned_components, dtype=torch.long, device=gamma.device)
                aligned_rewards = mixture_output["rewards"][row_indices, aligned_components_tensor, :]
                aligned_acc.add_batch(aligned_rewards, candidate_mask, dimensions)

    cluster_state = {"true_labels": true_labels, "posterior_pred_labels": posterior_pred_labels}
    if collect_only_cluster_state:
        return {}, cluster_state

    metrics = policy_acc.metrics("ranking")
    if posterior_pred_labels:
        num_clusters = len(dimension_to_id) if dimension_to_id else script_args.num_heads
        alignment = cluster_alignment or _cluster_alignment(true_labels, posterior_pred_labels, num_clusters)
        aligned_pred = [alignment.pred_to_true.get(pred_label, pred_label) for pred_label in posterior_pred_labels]
        cluster_correct = sum(int(pred == true) for pred, true in zip(aligned_pred, true_labels))
        raw_correct = sum(int(pred == true) for pred, true in zip(posterior_pred_labels, true_labels))
        pred_counts = Counter(posterior_pred_labels)
        metrics.update(posterior_acc.metrics("mixture_posterior"))
        if cluster_alignment is not None:
            metrics.update(aligned_acc.metrics("mixture_aligned"))
        metrics.update(
            {
                "mixture/cluster_acc": cluster_correct / max(len(true_labels), 1),
                "mixture/cluster_acc_raw": raw_correct / max(len(true_labels), 1),
                "mixture/nmi": _nmi(true_labels, posterior_pred_labels),
                "mixture/ari": _ari(true_labels, posterior_pred_labels),
                "mixture/gamma_entropy": gamma_entropy_sum / max(gamma_count, 1),
                "mixture/nll": mixture_nll_sum / max(mixture_nll_count, 1),
            }
        )
        for cluster_id in range(num_clusters):
            metrics[f"mixture/cluster_balance/{cluster_id}"] = pred_counts.get(cluster_id, 0) / max(
                len(posterior_pred_labels),
                1,
            )
    return metrics, cluster_state


def evaluate_ranking_splits(trainer, ranking_dataset: Dict[str, Dataset]) -> Dict[str, Dict[str, float]]:
    dimension_to_id = _dimension_mapping(ranking_dataset)
    cluster_alignment = None
    if "validation" in ranking_dataset:
        _, cluster_state = evaluate_ranking_split(
            trainer=trainer,
            split_dataset=ranking_dataset["validation"],
            dimension_to_id=dimension_to_id,
            collect_only_cluster_state=True,
        )
        cluster_alignment = _cluster_alignment(
            cluster_state["true_labels"],
            cluster_state["posterior_pred_labels"],
            num_clusters=max(len(dimension_to_id), 1),
        )
    results: Dict[str, Dict[str, float]] = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in ranking_dataset:
            continue
        split_metrics, _ = evaluate_ranking_split(
            trainer=trainer,
            split_dataset=ranking_dataset[split_name],
            dimension_to_id=dimension_to_id,
            cluster_alignment=cluster_alignment,
        )
        results[split_name] = split_metrics
    return results


class MMPOTrainer(RewardTrainer):
    """EM-only MMPO trainer with explicit E-step / M-step loop."""

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        return dataset

    def _set_active_adapter_no_toggle(self, model, adapter_name: str) -> None:
        """Switch active adapter without touching requires_grad flags."""
        unwrapped = self.accelerator.unwrap_model(model)
        if hasattr(unwrapped, "active_adapter"):
            unwrapped.active_adapter = adapter_name
        for module in unwrapped.modules():
            if isinstance(module, BaseTunerLayer):
                module._active_adapter = [adapter_name]

    def _adapter_grad_sums(self, model) -> Dict[str, float]:
        unwrapped = self.accelerator.unwrap_model(model)
        adapter_names = getattr(unwrapped, "mixture_adapter_names", [])
        grad_sums: Dict[str, float] = {name: 0.0 for name in adapter_names}
        for param_name, param in unwrapped.named_parameters():
            if param.grad is None:
                continue
            for adapter_name in adapter_names:
                if f".{adapter_name}." in param_name:
                    grad_sums[adapter_name] += float(param.grad.detach().abs().sum().item())
        return grad_sums

    @staticmethod
    def _sequence_logp_from_logits(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Causal LM token log-probability over response tokens only (no length normalization).
        shift_logits = logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        shift_mask = response_mask[:, 1:].to(dtype=shift_logits.dtype)
        token_logp = torch.log_softmax(shift_logits, dim=-1).gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        seq_logp = (token_logp * shift_mask).sum(dim=1)
        return seq_logp

    def _get_batch_mixture_output(self, model, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        response_mask = batch["response_mask"]
        candidate_mask = batch["candidate_mask"]
        batch_size, num_candidates, seq_len = input_ids.shape

        flat_input_ids = input_ids.view(batch_size * num_candidates, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_candidates, seq_len)
        flat_response_mask = response_mask.view(batch_size * num_candidates, seq_len)
        unwrapped = self.accelerator.unwrap_model(model)
        adapter_names = getattr(unwrapped, "mixture_adapter_names")
        reference_model = getattr(self, "reference_model", None)
        if reference_model is None:
            raise RuntimeError("reference_model is required for log-prob ratio reward.")
        with torch.no_grad():
            ref_outputs = reference_model(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
                output_hidden_states=False,
            )
            ref_logp = self._sequence_logp_from_logits(ref_outputs.logits, flat_input_ids, flat_response_mask)

        reward_list: List[torch.Tensor] = []
        pooled_for_router: Optional[torch.Tensor] = None
        need_context = script_args.use_contextual_router
        for adapter_idx, adapter_name in enumerate(adapter_names):
            self._set_active_adapter_no_toggle(model, adapter_name)
            outputs = model(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
                output_hidden_states=(need_context and adapter_idx == 0),
            )
            seq_logp = self._sequence_logp_from_logits(outputs.logits, flat_input_ids, flat_response_mask)
            reward = script_args.beta * (seq_logp - ref_logp)
            reward_list.append(reward.view(batch_size, num_candidates))
            if need_context and adapter_idx == 0:
                hidden = outputs.hidden_states[-1]
                last_indices = flat_attention_mask.long().sum(dim=1).clamp_min(1) - 1
                pooled_for_router = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]
        self._set_active_adapter_no_toggle(model, adapter_names[0])

        rewards = torch.stack(reward_list, dim=1)

        if script_args.use_contextual_router:
            if pooled_for_router is None:
                raise RuntimeError("Contextual router requires hidden states, but none were collected.")
            pooled = pooled_for_router.view(batch_size, num_candidates, -1)
            weights = candidate_mask.to(dtype=pooled.dtype).unsqueeze(-1)
            context = (pooled * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            context = context.detach()
            mixture_router = getattr(unwrapped, "mixture_router")
            router_logits = mixture_router(context.to(dtype=next(mixture_router.parameters()).dtype))
        else:
            global_logits = getattr(unwrapped, "mixture_router_logits")
            router_logits = global_logits.unsqueeze(0).expand(batch_size, -1)

        return {"router_logits": router_logits, "rewards": rewards}

    def _compute_em_loss_metrics(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        gamma: Optional[torch.Tensor] = None,
        include_cluster_metrics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor], torch.Tensor]:
        mixture_output = self._get_batch_mixture_output(model, batch)
        candidate_mask = batch["candidate_mask"]
        rankings = torch.arange(candidate_mask.shape[1], device=candidate_mask.device).unsqueeze(0).expand_as(candidate_mask).long()
        valid_rows = candidate_mask.sum(dim=1) >= 2
        mixture_nll, comp_logp = mixture_pl_nll(
            mixture_output["router_logits"],
            mixture_output["rewards"],
            rankings,
            candidate_mask=candidate_mask,
        )
        if gamma is None:
            gamma = em_responsibilities(
                mixture_output["router_logits"],
                comp_logp,
                temperature=script_args.em_temperature,
            ).detach()
        em_nll = em_expected_complete_nll(
            mixture_output["router_logits"],
            comp_logp,
            gamma,
            valid_mask=valid_rows,
        )

        posterior_pred = gamma.argmax(dim=1)
        row_indices = torch.arange(gamma.shape[0], device=gamma.device)
        posterior_rewards = mixture_output["rewards"][row_indices, posterior_pred, :]
        lm = listwise_metrics(posterior_rewards, candidate_mask)

        metrics: Dict[str, float] = {
            "mixture/nll": float(mixture_nll.detach().item()),
            "mixture/em_nll": float(em_nll.detach().item()),
            "listwise/top1_acc": float(lm["listwise/top1_acc"].detach().item()),
            "listwise/pairwise_acc": float(lm["listwise/pairwise_acc"].detach().item()),
            "listwise/utility_first": float(lm["listwise/utility_first"].detach().item()),
            "listwise/utility_last": float(lm["listwise/utility_last"].detach().item()),
            "listwise/utility_mean": float(lm["listwise/utility_mean"].detach().item()),
        }
        if include_cluster_metrics:
            metrics.update(
                compute_cluster_assignment_metrics(
                    assignment_scores=gamma,
                    attribute_ids=batch.get("attribute_id", torch.full((gamma.shape[0],), -1, device=gamma.device)),
                    num_clusters=script_args.num_heads,
                )
            )
        return em_nll, metrics, mixture_output, gamma

    def _m_step_backward_single_cluster(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        gamma: torch.Tensor,
        cluster_idx: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Run memory-efficient M-step for one cluster with fixed gamma."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        response_mask = batch["response_mask"]
        candidate_mask = batch["candidate_mask"]
        batch_size, num_candidates, seq_len = input_ids.shape
        rankings = torch.arange(num_candidates, device=candidate_mask.device).unsqueeze(0).expand_as(candidate_mask).long()
        valid_rows = candidate_mask.sum(dim=1) >= 2

        flat_input_ids = input_ids.view(batch_size * num_candidates, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_candidates, seq_len)
        flat_response_mask = response_mask.view(batch_size * num_candidates, seq_len)

        unwrapped = self.accelerator.unwrap_model(model)
        adapter_names = getattr(unwrapped, "mixture_adapter_names")
        reference_model = getattr(self, "reference_model", None)
        if reference_model is None:
            raise RuntimeError("reference_model is required for log-prob ratio reward.")
        if cluster_idx < 0 or cluster_idx >= len(adapter_names):
            raise ValueError(f"Invalid cluster_idx={cluster_idx}, expected [0, {len(adapter_names) - 1}]")

        with torch.no_grad():
            ref_outputs = reference_model(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
                output_hidden_states=False,
            )
            ref_logp = self._sequence_logp_from_logits(ref_outputs.logits, flat_input_ids, flat_response_mask)

        adapter_name = adapter_names[cluster_idx]
        self._set_active_adapter_no_toggle(model, adapter_name)
        outputs = model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            output_hidden_states=False,
        )
        seq_logp = self._sequence_logp_from_logits(outputs.logits, flat_input_ids, flat_response_mask)
        reward = script_args.beta * (seq_logp - ref_logp)
        reward_matrix = reward.view(batch_size, num_candidates)
        comp_logp_c = pl_log_prob(reward_matrix, rankings, candidate_mask=candidate_mask)
        comp_term = -(gamma[:, cluster_idx] * comp_logp_c)
        if torch.any(valid_rows):
            comp_term = comp_term[valid_rows].mean()
        else:
            comp_term = reward_matrix.new_zeros(())
        if comp_term.requires_grad:
            self.accelerator.backward(comp_term)

        lm = listwise_metrics(reward_matrix.detach(), candidate_mask)
        metrics: Dict[str, float] = {
            "cluster/index": float(cluster_idx),
            "cluster/nll": float(comp_term.detach().item()),
            "listwise/top1_acc": float(lm["listwise/top1_acc"].detach().item()),
            "listwise/pairwise_acc": float(lm["listwise/pairwise_acc"].detach().item()),
            "listwise/utility_first": float(lm["listwise/utility_first"].detach().item()),
            "listwise/utility_last": float(lm["listwise/utility_last"].detach().item()),
            "listwise/utility_mean": float(lm["listwise/utility_mean"].detach().item()),
        }
        self._set_active_adapter_no_toggle(model, adapter_names[0])
        return float(comp_term.detach().item()), metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        loss, metrics, output, gamma = self._compute_em_loss_metrics(model, inputs, gamma=None, include_cluster_metrics=True)
        if return_outputs:
            return loss, {"metrics": metrics, "mixture_output": output, "gamma": gamma}
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        del ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None

        gamma = outputs["gamma"]
        rewards = outputs["mixture_output"]["rewards"]
        candidate_mask = inputs["candidate_mask"]
        posterior_pred = gamma.argmax(dim=1)
        row_indices = torch.arange(gamma.shape[0], device=gamma.device)
        posterior_rewards = rewards[row_indices, posterior_pred, :]
        logits = posterior_rewards.masked_fill(~candidate_mask, -1e4)
        logits = nested_detach(logits)

        valid_counts = candidate_mask.sum(dim=1).to(dtype=logits.dtype)
        attribute_ids = inputs.get("attribute_id", torch.full((logits.shape[0],), -1, device=logits.device))
        labels = torch.stack((valid_counts, attribute_ids.to(device=logits.device, dtype=logits.dtype)), dim=1)
        labels = self._prepare_inputs(labels)
        return loss.detach(), logits, labels

    def _maybe_evaluate_and_save_em(self) -> None:
        was_training = self.model.training
        eval_steps = int(getattr(self.args, "eval_steps", 0) or 0)
        if eval_steps > 0 and self.state.global_step % eval_steps == 0:
            metrics = safe_trainer_evaluate(self)
            if metrics and accelerator.is_main_process and wandb.run is not None:
                wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=self.state.global_step)
        save_steps = int(getattr(self.args, "save_steps", 0) or 0)
        if script_args.save_checkpoint and save_steps > 0 and self.state.global_step % save_steps == 0:
            checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
            _safe_save_checkpoint(self, checkpoint_dir)
        if was_training:
            self.model.train()

    def _full_dataset_e_step(self, train_dataloader) -> Tuple[List[Dict], List[torch.Tensor]]:
        """Iterate entire training set once to compute per-batch gamma (E-step).
        """
        self.model.eval()
        epoch_batches: List[Dict] = []
        epoch_gammas: List[torch.Tensor] = []

        e_pbar = tqdm(
            train_dataloader,
            desc="E-step",
            disable=not accelerator.is_main_process,
            dynamic_ncols=True,
            leave=False,
        )
        with torch.no_grad():
            for batch in e_pbar:
                inputs = self._prepare_inputs(batch)
                _, _, _, gamma = self._compute_em_loss_metrics(
                    self.model, inputs, gamma=None, include_cluster_metrics=False,
                )
                # Save to CPU so GPU memory is free for M-step
                epoch_batches.append(
                    {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                )
                epoch_gammas.append(gamma.cpu())

        return epoch_batches, epoch_gammas

    def _epoch_router_prior_update(self, epoch_gammas: List[torch.Tensor]) -> None:
        """Closed-form router prior update using full-epoch gamma aggregate."""
        if script_args.use_contextual_router or not script_args.use_closed_form_router_prior_update:
            return
        full_gamma = torch.cat(epoch_gammas, dim=0)   # [N_train, K]
        prior = full_gamma.mean(dim=0)
        prior = prior / prior.sum().clamp_min(1e-12)
        router = getattr(self.accelerator.unwrap_model(self.model), "mixture_router_logits")
        with torch.no_grad():
            router.copy_(
                torch.log(prior.clamp_min(1e-12)).to(dtype=router.dtype, device=router.device)
            )

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):  # noqa: D401
        del resume_from_checkpoint, trial, ignore_keys_for_eval, kwargs
        if self.args.gradient_accumulation_steps != 1:
            raise ValueError("This EM-only trainer currently requires gradient_accumulation_steps=1.")

        args = self.args
        train_dataloader = self.get_train_dataloader()
        m_step_updates = script_args.m_step_updates
        num_train_epochs = math.ceil(args.num_train_epochs)
        # Strict cluster-wise full-dataset M-step:
        # for each m-step and each cluster, traverse whole dataset then step once.
        steps_per_epoch = m_step_updates * script_args.num_heads
        total_steps = num_train_epochs * steps_per_epoch

        self.create_optimizer_and_scheduler(num_training_steps=total_steps)
        self.model, self.optimizer, train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            train_dataloader,
            self.lr_scheduler,
        )
        self.model_wrapped = self.model
        self.optimizer.zero_grad()
        self.state.global_step = 0
        self.state.max_steps = total_steps
        self.state.num_train_epochs = num_train_epochs

        total_loss = 0.0
        start_time = time.time()
        accel_device = self.accelerator.device

        pbar = tqdm(
            total=total_steps,
            desc="MMPO training",
            disable=not accelerator.is_main_process,
            dynamic_ncols=True,
        )

        for epoch in range(num_train_epochs):
            # ----------------------------------------------------------------
            # Full-dataset E-step: compute gamma for every training sample.
            # ----------------------------------------------------------------
            if accelerator.is_main_process:
                print(f"\nEpoch {epoch + 1}/{num_train_epochs}: running full-dataset E-step...")
            epoch_batches, epoch_gammas = self._full_dataset_e_step(train_dataloader)

            # Update alpha from full-dataset gamma before cluster-wise theta updates.
            self._epoch_router_prior_update(epoch_gammas)

            # ----------------------------------------------------------------
            # M-step: strict cluster-wise full-dataset updates.
            # ----------------------------------------------------------------
            self.model.train()
            for m_step in range(m_step_updates):
                for cluster_idx in range(script_args.num_heads):
                    should_log = args.logging_steps > 0 and (self.state.global_step + 1) % args.logging_steps == 0
                    self.optimizer.zero_grad()

                    dataset_loss_sum = 0.0
                    dataset_pairwise_sum = 0.0
                    dataset_top1_sum = 0.0
                    dataset_metric_count = 0
                    latest_metrics: Dict[str, float] = {}

                    for cpu_inputs, gamma_cpu in zip(epoch_batches, epoch_gammas):
                        inputs = {
                            k: v.to(accel_device) if isinstance(v, torch.Tensor) else v
                            for k, v in cpu_inputs.items()
                        }
                        gamma = gamma_cpu.to(accel_device)
                        batch_loss_val, metrics = self._m_step_backward_single_cluster(
                            self.model,
                            inputs,
                            gamma=gamma,
                            cluster_idx=cluster_idx,
                        )
                        dataset_loss_sum += float(batch_loss_val)
                        dataset_pairwise_sum += float(metrics.get("listwise/pairwise_acc", 0.0))
                        dataset_top1_sum += float(metrics.get("listwise/top1_acc", 0.0))
                        dataset_metric_count += 1
                        latest_metrics = metrics

                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    grad_sums = self._adapter_grad_sums(self.model) if should_log else {}
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    self.state.global_step += 1
                    loss_val = dataset_loss_sum / max(dataset_metric_count, 1)
                    pairwise_acc = dataset_pairwise_sum / max(dataset_metric_count, 1)
                    top1_acc = dataset_top1_sum / max(dataset_metric_count, 1)
                    total_loss += float(loss_val)
                    pbar.set_postfix({
                        "epoch": f"{epoch + 1}/{num_train_epochs}",
                        "cluster": cluster_idx,
                        "loss": f"{loss_val:.4f}",
                        "acc": f"{pairwise_acc:.3f}",
                        "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}" if self.lr_scheduler else "?",
                    })
                    pbar.update(1)

                    if should_log and accelerator.is_main_process and wandb.run is not None:
                        log_payload = {
                            "train/loss": float(loss_val),
                            "train/accuracy": float(pairwise_acc),
                            "train/top1_acc": float(top1_acc),
                            "train/cluster_idx": float(cluster_idx),
                            "train/epoch": epoch + (m_step + 1) / max(m_step_updates, 1),
                        }
                        log_payload.update({f"train/{k}": float(v) for k, v in latest_metrics.items()})
                        for adapter_name, grad_sum in grad_sums.items():
                            log_payload[f"train/grad_sum_{adapter_name}"] = grad_sum
                        wandb.log(log_payload, step=self.state.global_step)

                    self._maybe_evaluate_and_save_em()

        pbar.close()

        runtime = time.time() - start_time
        train_loss = total_loss / max(self.state.global_step, 1)
        metrics = {
            "train_runtime": runtime,
            "train_samples_per_second": self.state.global_step * args.train_batch_size / max(runtime, 1e-12),
            "train_steps_per_second": self.state.global_step / max(runtime, 1e-12),
            "total_flos": float(getattr(self.state, "total_flos", 0.0)),
            "train_loss": train_loss,
        }
        return TrainOutput(self.state.global_step, train_loss, metrics)


def compute_metrics(eval_pred):
    prediction_scores = eval_pred.predictions
    if not isinstance(prediction_scores, np.ndarray) or prediction_scores.ndim != 2:
        return {"accuracy": 0.0}

    label_ids = eval_pred.label_ids
    if isinstance(label_ids, np.ndarray) and label_ids.ndim == 2 and label_ids.shape[1] > 0:
        valid_counts = label_ids[:, 0].astype(np.int64)
    else:
        valid_counts = np.full((prediction_scores.shape[0],), prediction_scores.shape[1], dtype=np.int64)

    total_pairs = 0
    correct_pairs = 0
    attr_correct: Dict[int, int] = defaultdict(int)
    attr_total: Dict[int, int] = defaultdict(int)
    attribute_ids = None
    if isinstance(label_ids, np.ndarray) and label_ids.ndim == 2 and label_ids.shape[1] > 1:
        attribute_ids = label_ids[:, 1].astype(np.int64)

    for row_idx, scores in enumerate(prediction_scores):
        m = int(valid_counts[row_idx])
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                total_pairs += 1
                is_correct = scores[i] >= scores[j]
                if is_correct:
                    correct_pairs += 1
                if attribute_ids is not None:
                    aid = int(attribute_ids[row_idx])
                    if aid >= 0:
                        attr_total[aid] += 1
                        if is_correct:
                            attr_correct[aid] += 1

    accuracy = float(correct_pairs / total_pairs) if total_pairs > 0 else 0.0
    metrics: Dict[str, float] = {"accuracy": accuracy, "pairwise_accuracy": accuracy}
    if attr_total:
        attr_accs: List[float] = []
        for aid, total in attr_total.items():
            acc = float(attr_correct[aid] / total) if total > 0 else 0.0
            attr_name = ATTRIBUTE_ID_TO_NAME.get(int(aid), f"attr_{int(aid)}").replace("-", "_")
            metrics[f"accuracy_{attr_name}"] = acc
            attr_accs.append(acc)
        if attr_accs:
            metrics["accuracy_attribute_mean"] = float(np.mean(attr_accs))
            metrics["accuracy_attribute_best"] = float(np.max(attr_accs))
            metrics["accuracy_attribute_worst"] = float(np.min(attr_accs))
    return metrics


def safe_trainer_evaluate(trainer: RewardTrainer) -> Dict[str, float]:
    try:
        return trainer.evaluate()
    except ValueError as exc:
        message = str(exc)
        if "ZeRO inference only makes sense with ZeRO Stage 3" in message:
            if accelerator.is_main_process:
                print("DeepSpeed ZeRO-2 detected: running manual evaluation loop instead of trainer.evaluate().")
            return _manual_evaluate(trainer)
        raise


def _manual_evaluate(trainer: RewardTrainer) -> Dict[str, float]:
    """Run evaluation by directly calling prediction_step, bypassing DeepSpeed inference context."""
    model = trainer.model
    was_training = model.training
    model.eval()
    eval_dataloader = trainer.get_eval_dataloader()

    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            loss, logits, labels = trainer.prediction_step(model, batch, prediction_loss_only=False)
            if logits is not None:
                all_logits.append(logits.cpu().float().numpy() if isinstance(logits, torch.Tensor) else logits)
            if labels is not None:
                all_labels.append(labels.cpu().float().numpy() if isinstance(labels, torch.Tensor) else labels)
            if loss is not None:
                total_loss += float(loss.item())
                num_batches += 1

    if was_training:
        model.train()

    if not all_logits:
        return {}

    predictions_arr = np.concatenate(all_logits, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    from transformers.trainer_utils import EvalPrediction
    eval_pred = EvalPrediction(predictions=predictions_arr, label_ids=labels_arr)
    metrics = trainer.compute_metrics(eval_pred) if trainer.compute_metrics is not None else {}
    if num_batches > 0:
        metrics["loss"] = total_loss / num_batches
    return metrics


def _safe_save_checkpoint(trainer: RewardTrainer, checkpoint_dir: str) -> None:
    """Save model checkpoint without going through DeepSpeed's get_state_dict path."""
    if not script_args.save_checkpoint:
        return
    if not accelerator.is_main_process:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    tmp_path = os.path.join(checkpoint_dir, "pytorch_model.bin.tmp")
    final_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    try:
        unwrapped = accelerator.unwrap_model(trainer.model)
        state_dict = {k: v.cpu() for k, v in unwrapped.state_dict().items()}
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, final_path)
        if hasattr(unwrapped, "config"):
            unwrapped.config.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: checkpoint save failed at step {trainer.state.global_step}: {exc}")
        for leftover in (tmp_path, final_path):
            try:
                if os.path.exists(leftover):
                    os.remove(leftover)
            except OSError:
                pass


if accelerator.is_main_process and script_args.use_wandb:
    wandb.init(project="MultiRewardLearning", name=script_args.wandb_name, config=vars(script_args))

tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast=False)
tokenizer.model_max_length = script_args.max_length
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if "cyclic_ultrafeedback_all_pairs" not in script_args.data_path:
    raise NotImplementedError(
        "This MMPO reimplementation currently expects listwise cyclic UltraFeedback data. "
        "Please use --data_path cyclic_ultrafeedback_all_pairs."
    )

train_dataset = load_cyclic_ultrafeedback_listwise_split("train", tokenizer)
ranking_dataset: Dict[str, Dataset] = {"train": train_dataset}
try:
    ranking_dataset["validation"] = load_cyclic_ultrafeedback_listwise_split("validation", tokenizer)
except Exception:
    pass
try:
    ranking_dataset["test"] = load_cyclic_ultrafeedback_listwise_split("test", tokenizer)
except Exception:
    pass

eval_split = "validation" if "validation" in ranking_dataset else ("test" if "test" in ranking_dataset else "train")
eval_dataset = ranking_dataset[eval_split]

if accelerator.is_main_process and eval_split == "test" and not script_args.eval_only:
    print(
        "Warning: validation split not found, using test split for in-training evaluation. "
        "This may leak test information into model selection."
    )

if accelerator.is_main_process:
    print(
        f"Loaded listwise cyclic UltraFeedback: train={len(train_dataset)} "
        f"eval_split={eval_split} eval={len(eval_dataset)}"
    )

model_name_split = script_args.base_model.split("/")[-1]
output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}"
training_args = RewardConfig(
    output_dir=os.path.join(output_name, "logs"),
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    eval_strategy=script_args.eval_strategy,
    eval_steps=script_args.eval_steps,
    save_strategy=script_args.save_strategy,
    save_steps=script_args.save_steps,
    save_total_limit=1,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    label_names=[],
    bf16=True,
    logging_strategy="steps",
    logging_steps=script_args.logging_steps,
    warmup_ratio=0.05,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name=script_args.run_name,
    report_to="wandb" if script_args.use_wandb else "none",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
)
if script_args.eval_only and getattr(training_args, "deepspeed", None) is not None:
    training_args.deepspeed = None

local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
print(device)

reference_model_name = script_args.reference_model or script_args.base_model
reference_model = AutoModelForCausalLM.from_pretrained(
    reference_model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
reference_model.config.pad_token_id = tokenizer.pad_token_id
reference_model.resize_token_embeddings(len(tokenizer))
reference_model.config.use_cache = False
reference_model.eval()
for param in reference_model.parameters():
    param.requires_grad = False
reference_model.to(device)

model = AutoModelForCausalLM.from_pretrained(
    script_args.base_model,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))
# RewardTrainer checks model.config.num_labels == 1; patch it so causal LM passes that guard
model.config.num_labels = 1
model.config.use_cache = False

lora_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=_parse_lora_target_modules(script_args.lora_target_modules),
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config, adapter_name="expert_0")
for head_idx in range(1, script_args.num_heads):
    model.add_adapter(f"expert_{head_idx}", lora_config)
model.mixture_adapter_names = [f"expert_{idx}" for idx in range(script_args.num_heads)]
model.set_adapter(model.mixture_adapter_names[0])

# Keep all expert adapters trainable; we'll switch active adapter without toggling grad flags.
for name, param in model.named_parameters():
    if any(f".{adapter_name}." in name for adapter_name in model.mixture_adapter_names):
        param.requires_grad = True
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
if accelerator.is_main_process:
    model.print_trainable_parameters()

if script_args.use_contextual_router:
    model.mixture_router = nn.Linear(model.config.hidden_size, script_args.num_heads, bias=False).to(device, dtype=torch.bfloat16)
else:
    model.mixture_router_logits = nn.Parameter(torch.zeros(script_args.num_heads, dtype=torch.bfloat16, device=device))
    if script_args.use_closed_form_router_prior_update:
        model.mixture_router_logits.requires_grad_(False)

trainer = MMPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=ListwiseRewardCollator(tokenizer=tokenizer, max_length=script_args.max_length),
)
trainer.reference_model = reference_model
trainer.ranking_dataset = ranking_dataset
trainer.ranking_eval_split_name = eval_split
trainer.dimension_to_id = _dimension_mapping(ranking_dataset)

if script_args.eval_only:
    print("eval_only mode: evaluating checkpoint")
    eval_metrics = safe_trainer_evaluate(trainer)
    if eval_metrics:
        trainer.log_metrics("eval_only", eval_metrics)
        trainer.save_metrics("eval_only", eval_metrics)
        if accelerator.is_main_process and wandb.run is not None:
            wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=trainer.state.global_step)
else:
    print("training")
    trainer.train()
    print("final evaluating")
    final_eval_metrics = safe_trainer_evaluate(trainer)
    if final_eval_metrics:
        trainer.log_metrics("eval_final", final_eval_metrics)
        trainer.save_metrics("eval_final", final_eval_metrics)
        if accelerator.is_main_process and wandb.run is not None:
            wandb.log({f"eval/{k}": v for k, v in final_eval_metrics.items()}, step=trainer.state.global_step)
    if script_args.save_checkpoint:
        _safe_save_checkpoint(trainer, output_name)

if accelerator.is_main_process and script_args.use_wandb and wandb.run is not None:
    wandb.finish()
