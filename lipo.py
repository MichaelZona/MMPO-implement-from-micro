
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os

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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, PreTrainedModel
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import PaddingStrategy
from trl import RewardConfig, RewardTrainer

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


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
@dataclass
class ScriptArguments:
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-3
    num_train_epochs: int = 3
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    max_length: int = 1024
    max_prompt_length: int = 512
    base_model: str = "Qwen/Qwen3-0.6B"
    wandb_name: str = "lipo"
    log_dir: str = "./output_models"
    freeze_pretrained: bool = True
    data_path: str = "cyclic_ultrafeedback_all_pairs"
    downsample_rate: float = 1.0
    eval_only: bool = False
    manual_seed: int = 42
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 10
    run_name: str = "lipo"
    use_wandb: bool = True


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
torch.manual_seed(script_args.manual_seed)

if script_args.downsample_rate <= 0 or script_args.downsample_rate > 1:
    raise ValueError("`downsample_rate` must be in (0, 1].")

if accelerator.is_main_process:
    print("Arguments:")
    for arg_name in vars(script_args):
        print(format(arg_name, "<34"), format(str(getattr(script_args, arg_name)), "<"))


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------
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
    return ATTRIBUTE_NAME_TO_ID.get(str(attribute_name).strip(), -1)


def _truncate_prompt(input_ids: List[int], max_prompt_length: int) -> List[int]:
    return input_ids[-max_prompt_length:] if len(input_ids) > max_prompt_length else input_ids


def _response_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict) and "content" in response:
        return str(response["content"])
    return str(response)


def _prompt_to_messages(prompt: Any) -> List[Dict[str, str]]:
    if isinstance(prompt, list):
        messages = [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in prompt
            if isinstance(item, dict) and "role" in item and "content" in item
        ]
        if messages:
            return messages
    return [{"role": "user", "content": str(prompt)}]


def _tokenize_listwise_example(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    prompt_messages = _prompt_to_messages(example["prompt"])
    prompt_template = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_template, add_special_tokens=False)["input_ids"]
    prompt_ids = _truncate_prompt(prompt_ids, script_args.max_prompt_length)

    responses = example["responses"]
    scores = example.get("scores")

    # Fix 3: validate scores exist and match response count.
    if scores is None or len(scores) != len(responses):
        raise ValueError(
            f"LiPO-λ requires a valid 'scores' list for every response. "
            f"Got {len(scores) if scores is not None else None} scores for {len(responses)} responses."
        )

    # Fix 4: sort by score descending and keep sorted scores for Δ_ij.
    order = sorted(range(len(responses)), key=lambda i: float(scores[i]), reverse=True)
    responses = [responses[i] for i in order]
    sorted_scores = [float(scores[i]) for i in order]  # saved for LiPO-λ weighting

    candidate_input_ids: List[List[int]] = []
    candidate_attention_masks: List[List[int]] = []
    for response in responses:
        response_text = _response_to_text(response)
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
            response_ids = response_ids + [eos_id]
        input_ids = prompt_ids + response_ids
        if len(input_ids) > script_args.max_length:
            overflow = len(input_ids) - script_args.max_length
            if overflow < len(prompt_ids):
                input_ids = prompt_ids[overflow:] + response_ids
            else:
                input_ids = response_ids[overflow - len(prompt_ids):]
        candidate_input_ids.append(input_ids)
        candidate_attention_masks.append([1] * len(input_ids))

    return {
        "candidate_input_ids": candidate_input_ids,
        "candidate_attention_mask": candidate_attention_masks,
        "candidate_scores": sorted_scores,
        "preference_dimension": example.get("preference_dimension"),
        "attribute_id": attribute_to_id(
            CYCLIC_ULTRAFEEDBACK_ATTRIBUTE_ALIASES.get(example.get("preference_dimension"))
        ),
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
        raise ValueError(f"No listwise rows found at {dataset_dir}")
    dataset = Dataset.from_list(rows)
    dataset = dataset.map(lambda ex: _tokenize_listwise_example(ex, tokenizer), batched=False, num_proc=10)
    dataset = dataset.filter(
        lambda x: len(x["candidate_input_ids"]) >= 2
        and all(len(ids) <= script_args.max_length for ids in x["candidate_input_ids"]),
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
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_candidates = max(len(f["candidate_input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

        max_seq_len = max(
            len(seq)
            for f in features
            for seq in f["candidate_input_ids"]
        )

        batch_ids, batch_attn, candidate_mask_rows, batch_scores = [], [], [], []
        for feature in features:
            ids_list = list(feature["candidate_input_ids"])
            attn_list = list(feature["candidate_attention_mask"])
            scores_list = list(feature["candidate_scores"])
            valid_count = len(ids_list)
            candidate_mask_rows.append([1] * valid_count + [0] * (max_candidates - valid_count))
            scores_list += [0.0] * (max_candidates - valid_count)
            batch_scores.append(scores_list)
            for _ in range(max_candidates - valid_count):
                ids_list.append([])
                attn_list.append([])
            padded_ids, padded_attn = [], []
            for ids, attn in zip(ids_list, attn_list):
                pad_len = max_seq_len - len(ids)
                padded_ids.append(ids + [pad_id] * pad_len)
                padded_attn.append(attn + [0] * pad_len)
            batch_ids.append(padded_ids)
            batch_attn.append(padded_attn)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
            "candidate_mask": torch.tensor(candidate_mask_rows, dtype=torch.bool),
            "candidate_scores": torch.tensor(batch_scores, dtype=torch.float),
            "attribute_id": torch.tensor([f.get("attribute_id", -1) for f in features], dtype=torch.long),
            "preference_dimension": [f.get("preference_dimension", "unknown") for f in features],
            "return_loss": True,
        }


# ---------------------------------------------------------------------------
# LiPO-λ loss (DCG lambda-weighted pairwise logistic)
# ---------------------------------------------------------------------------
def _rowwise_minmax_normalize(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize valid label scores in each row to [0, 1] for a stable gain."""
    high = torch.finfo(values.dtype).max
    low = torch.finfo(values.dtype).min
    masked_min_values = values.masked_fill(~mask, high)
    masked_max_values = values.masked_fill(~mask, low)
    row_min = masked_min_values.min(dim=1, keepdim=True).values
    row_max = masked_max_values.max(dim=1, keepdim=True).values
    denom = (row_max - row_min).clamp_min(eps)
    normalized = (values - row_min) / denom
    return normalized.masked_fill(~mask, 0.0)


def lipo_lambda_loss(
    rewards: torch.Tensor,         # [B, C] model reward scores
    label_scores: torch.Tensor,    # [B, C] ground-truth label values, sorted best→worst
    candidate_mask: torch.Tensor,  # [B, C] bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LiPO-λ reward-model variant.

    This implements the LambdaLoss-style objective:
        sum_{label_i > label_j} Δ_ij * softplus(-(r_i - r_j))

    where Δ_ij is the DCG lambda weight induced by current model ranks.
    The model is still a reward model, so `rewards` are scalar reward-head
    outputs rather than DPO-style policy/reference log-ratio scores.
    """
    B, C = rewards.shape
    device = rewards.device

    label_scores = label_scores.to(device=device, dtype=rewards.dtype)
    candidate_mask = candidate_mask.to(device=device, dtype=torch.bool)

    # Current model-induced ranks τ(i). Invalid candidates are pushed to the end.
    masked_rewards = rewards.masked_fill(~candidate_mask, torch.finfo(rewards.dtype).min)
    sorted_idx = torch.argsort(masked_rewards, dim=1, descending=True)
    rank_pos = torch.arange(C, device=device, dtype=torch.long).unsqueeze(0).expand(B, C)
    pred_rank = torch.empty_like(sorted_idx)
    pred_rank.scatter_(1, sorted_idx, rank_pos)
    pred_rank = pred_rank.to(dtype=rewards.dtype) + 1.0

    # Stable DCG gain. If your labels are already in [0, 1], this mostly keeps
    # the same ordering but avoids exploding gains for raw rating scales.
    normalized_labels = _rowwise_minmax_normalize(label_scores, candidate_mask)
    gain = torch.pow(torch.tensor(2.0, device=device, dtype=rewards.dtype), normalized_labels) - 1.0
    discount = torch.log1p(pred_rank).clamp_min(1e-12)

    r_i = rewards.unsqueeze(2)
    r_j = rewards.unsqueeze(1)
    y_i = label_scores.unsqueeze(2)
    y_j = label_scores.unsqueeze(1)
    g_i = gain.unsqueeze(2)
    g_j = gain.unsqueeze(1)
    d_i = discount.unsqueeze(2)
    d_j = discount.unsqueeze(1)

    # Only optimize real preference pairs. This avoids treating tied labels as ordered.
    valid_pair = (
        candidate_mask.unsqueeze(2)
        & candidate_mask.unsqueeze(1)
        & (y_i > y_j)
    )

    margin = r_i - r_j
    lambda_weight = (g_i - g_j).abs() * ((1.0 / d_i) - (1.0 / d_j)).abs()
    pair_loss = lambda_weight * F.softplus(-margin)

    valid_loss = pair_loss[valid_pair]
    if valid_loss.numel() == 0:
        zero = rewards.sum() * 0.0
        return zero, zero.detach()

    loss = valid_loss.mean()
    pairwise_acc = (margin[valid_pair] > 0).to(torch.float32).mean()
    return loss, pairwise_acc


# ---------------------------------------------------------------------------
# listwise_metrics: keep the same keys used for train logging, but mask pads.
# ---------------------------------------------------------------------------
def listwise_metrics(rewards: torch.Tensor, candidate_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    masked_rewards = rewards.masked_fill(~candidate_mask, torch.finfo(rewards.dtype).min)
    max_idx = masked_rewards.argmax(dim=1)
    top1_acc = (max_idx == 0).float().mean()

    upper_tri = torch.triu(
        torch.ones(rewards.size(1), rewards.size(1), device=rewards.device, dtype=torch.bool),
        diagonal=1,
    )
    margin = rewards.unsqueeze(2) - rewards.unsqueeze(1)
    valid_pairs = candidate_mask.unsqueeze(2) & candidate_mask.unsqueeze(1) & upper_tri.unsqueeze(0)
    pairwise_acc = ((margin >= 0) & valid_pairs).sum().float() / valid_pairs.sum().clamp_min(1).float()

    valid_counts = candidate_mask.sum(dim=1).clamp_min(1)
    last_indices = valid_counts - 1
    utility_first = rewards[:, 0]
    utility_last = rewards.gather(1, last_indices.unsqueeze(1)).squeeze(1)
    utility_mean = (rewards * candidate_mask.float()).sum(dim=1) / valid_counts.float()
    return {
        "listwise/top1_acc": top1_acc,
        "listwise/pairwise_acc": pairwise_acc,
        "listwise/utility_first": utility_first.mean(),
        "listwise/utility_last": utility_last.mean(),
        "listwise/utility_mean": utility_mean.mean(),
    }


# ---------------------------------------------------------------------------
# compute_metrics: keep the original final metric interface unchanged.
# It still evaluates whether the first candidate beats the last candidate and
# keeps the same attribute-wise metric names.
# ---------------------------------------------------------------------------
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    prediction_scores = eval_pred.predictions
    main_scores = prediction_scores[:, :2] if prediction_scores.ndim == 2 else prediction_scores
    predictions = np.argmax(main_scores, axis=1)
    labels = np.zeros(predictions.shape, dtype=np.int64)
    metrics = accuracy_metric.compute(predictions=predictions, references=labels)

    label_ids = eval_pred.label_ids
    if isinstance(label_ids, np.ndarray) and label_ids.ndim == 2 and label_ids.shape[1] > 1:
        attribute_ids = label_ids[:, 1].astype(np.int64)
        valid_mask = attribute_ids >= 0
        attribute_accuracies = []
        for attr_id in np.unique(attribute_ids[valid_mask]):
            attr_mask = attribute_ids == attr_id
            if not np.any(attr_mask):
                continue
            attr_name = ATTRIBUTE_ID_TO_NAME.get(int(attr_id), f"attr_{int(attr_id)}")
            attr_acc = (predictions[attr_mask] == labels[attr_mask]).mean().item()
            attribute_accuracies.append(float(attr_acc))
            metric_key = attr_name.replace("-", "_")
            metrics[f"accuracy_{metric_key}"] = float(attr_acc)
            metrics[f"count_{metric_key}"] = int(attr_mask.sum())
        if attribute_accuracies:
            metrics["accuracy_attribute_mean"] = float(np.mean(attribute_accuracies))
            metrics["accuracy_attribute_best"] = float(np.max(attribute_accuracies))
            metrics["accuracy_attribute_worst"] = float(np.min(attribute_accuracies))
    return metrics


# ---------------------------------------------------------------------------
# Safe evaluate / save helpers
# ---------------------------------------------------------------------------
def safe_trainer_evaluate(trainer: RewardTrainer) -> Dict[str, float]:
    try:
        return trainer.evaluate()
    except ValueError as exc:
        if "ZeRO inference only makes sense with ZeRO Stage 3" in str(exc):
            if accelerator.is_main_process:
                print("DeepSpeed ZeRO-2 detected: running manual evaluation loop.")
            return _manual_evaluate(trainer)
        raise


def _manual_evaluate(trainer: RewardTrainer) -> Dict[str, float]:
    model = trainer.model
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in trainer.get_eval_dataloader():
            loss, logits, labels = trainer.prediction_step(model, batch, prediction_loss_only=False)
            if logits is not None:
                all_logits.append(logits.cpu().float().numpy() if isinstance(logits, torch.Tensor) else logits)
            if labels is not None:
                all_labels.append(labels.cpu().float().numpy() if isinstance(labels, torch.Tensor) else labels)
            if loss is not None:
                total_loss += float(loss.item())
                num_batches += 1
    model.train()
    if not all_logits:
        return {}
    from transformers.trainer_utils import EvalPrediction
    eval_pred = EvalPrediction(
        predictions=np.concatenate(all_logits, axis=0),
        label_ids=np.concatenate(all_labels, axis=0),
    )
    metrics = trainer.compute_metrics(eval_pred) if trainer.compute_metrics is not None else {}
    if num_batches > 0:
        metrics["loss"] = total_loss / num_batches
    return metrics


def _safe_save_checkpoint(trainer: RewardTrainer, checkpoint_dir: str) -> None:
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
    except Exception as exc:
        print(f"Warning: checkpoint save failed: {exc}")
        for leftover in (tmp_path, final_path):
            try:
                if os.path.exists(leftover):
                    os.remove(leftover)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# LiPO Trainer
# ---------------------------------------------------------------------------
class LiPOTrainer(RewardTrainer):
    """LiPO-λ reward model trainer (reward model variant)."""

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        return dataset

    def _get_batch_scores(self, model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass → reward scores [B, C]."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        B, C, L = input_ids.shape
        outputs = model(
            input_ids=input_ids.reshape(B * C, L),
            attention_mask=attention_mask.reshape(B * C, L),
        )
        return outputs.logits.squeeze(-1).reshape(B, C)   # [B, C]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        candidate_mask = inputs["candidate_mask"]         # [B, C]
        label_scores = inputs["candidate_scores"]         # [B, C]
        rewards = self._get_batch_scores(model, inputs)   # [B, C]

        loss, pairwise_acc = lipo_lambda_loss(rewards, label_scores, candidate_mask)

        lm = listwise_metrics(rewards, candidate_mask)
        metrics = {k: float(v.detach().item()) for k, v in lm.items()}
        metrics["loss"] = float(loss.detach().item())

        if model.training and accelerator.is_main_process and wandb.run is not None:
            wandb.log(
                {
                    "train/loss": metrics["loss"],
                    "train/accuracy": float(pairwise_acc.detach().item()),
                    "train/top1_acc": metrics["listwise/top1_acc"],
                },
                step=self.state.global_step,
            )

        if return_outputs:
            return loss, {"rewards": rewards, "metrics": metrics}
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

        rewards = outputs["rewards"]            # [B, C]
        candidate_mask = inputs["candidate_mask"]
        valid_counts = candidate_mask.sum(dim=1).clamp_min(1)
        last_indices = valid_counts - 1

        # Keep the final eval metric unchanged: compare first vs. last candidate.
        score_first = rewards[:, 0]
        score_last = rewards.gather(1, last_indices.unsqueeze(1)).squeeze(1)
        logits = nested_detach(torch.stack([score_first, score_last], dim=1))  # [B, 2]

        labels = torch.zeros(logits.shape[0], device=logits.device)
        if "attribute_id" in inputs:
            labels = torch.stack(
                (labels, inputs["attribute_id"].to(labels.device, dtype=labels.dtype)), dim=1
            )
        labels = self._prepare_inputs(labels)
        return loss.detach(), logits, labels


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
if accelerator.is_main_process and script_args.use_wandb:
    wandb.init(project="MultiRewardLearning", name=script_args.wandb_name, config=vars(script_args))

tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast=False)
tokenizer.model_max_length = script_args.max_length
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if "cyclic_ultrafeedback_all_pairs" not in script_args.data_path:
    raise NotImplementedError("lipo.py currently only supports --data_path cyclic_ultrafeedback_all_pairs.")

train_dataset = load_cyclic_ultrafeedback_listwise_split("train", tokenizer)
eval_split = "validation"
try:
    eval_dataset = load_cyclic_ultrafeedback_listwise_split(eval_split, tokenizer)
except Exception:
    eval_split = "test"
    eval_dataset = load_cyclic_ultrafeedback_listwise_split(eval_split, tokenizer)

if accelerator.is_main_process:
    print(f"Loaded listwise cyclic UltraFeedback: train={len(train_dataset)} eval_split={eval_split} eval={len(eval_dataset)}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
print(device)

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.base_model,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

if script_args.freeze_pretrained:
    for name, param in model.named_parameters():
        if "score" not in name:
            param.requires_grad = False

model.config.pad_token_id = tokenizer.pad_token_id

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable} || all params: {total} || trainable%: {100 * trainable / total:.2f}")

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
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
    max_length=script_args.max_length,
)

trainer = LiPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=ListwiseRewardCollator(tokenizer=tokenizer, max_length=script_args.max_length),
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
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
    _safe_save_checkpoint(trainer, output_name)

if accelerator.is_main_process and script_args.use_wandb and wandb.run is not None:
    wandb.finish()
