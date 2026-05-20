"""Standard DPO training with a hand-rolled Trainer subclass.

Kept manual on purpose: easier to extend the loss (SimPO, IPO, length-norm,
per-attribute reweighting, ...). For a drop-in TRL version use trl.DPOTrainer.

Data sources (combine with '-' in --data_path):
  cyclic_ultrafeedback_all_pairs           arrow shards, expanded to all pairs
  ultrafeedback_per_attribute_pairwise     already-pairwise, load_from_disk
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import os

import evaluate
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import PaddingStrategy

torch.backends.cuda.matmul.allow_tf32 = True
accelerator = Accelerator()


# ---------------------------------------------------------------------------
# Attribute bookkeeping (for per-dimension eval metrics)
# ---------------------------------------------------------------------------
ALL_KNOWN_ATTRIBUTES = [
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
    "ultrafeedback-helpfulness",
    "ultrafeedback-honesty",
    "ultrafeedback-instruction-following",
    "ultrafeedback-truthfulness",
]
ATTRIBUTE_NAME_TO_ID = {name: idx for idx, name in enumerate(ALL_KNOWN_ATTRIBUTES)}
ATTRIBUTE_ID_TO_NAME = {idx: name for name, idx in ATTRIBUTE_NAME_TO_ID.items()}

# Map raw attribute strings on disk -> canonical name in ALL_KNOWN_ATTRIBUTES
ATTRIBUTE_ALIASES = {
    "instruction_following": "ultrafeedback-instruction-following",
    "truthfulness": "ultrafeedback-truthfulness",
    "honesty": "ultrafeedback-honesty",
    "helpfulness": "helpfulness",
}
CYCLIC_ULTRAFEEDBACK_ATTRIBUTE_ALIASES = {
    "helpfulness": "ultrafeedback-helpfulness",
    "honesty": "ultrafeedback-honesty",
    "instruction_following": "ultrafeedback-instruction-following",
    "truthfulness": "ultrafeedback-truthfulness",
}


def attribute_to_id(name: Optional[str]) -> int:
    if name is None:
        return -1
    name = str(name).strip()
    canonical = ATTRIBUTE_ALIASES.get(name, name)
    return ATTRIBUTE_NAME_TO_ID.get(canonical, -1)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
@dataclass
class ScriptArguments:
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=5e-7)
    num_train_epochs: int = field(default=1)
    optim: str = field(default="adamw_torch")
    lr_scheduler_type: str = field(default="cosine")
    max_length: int = field(default=4096)
    base_model: str = field(default="Qwen/Qwen3-0.6B")
    wandb_name: str = field(default="standard_dpo")
    log_dir: str = field(default="./output_models")
    data_path: str = field(default="cyclic_ultrafeedback_all_pairs")
    downsample_rate: float = field(default=1.0)
    eval_only: bool = field(default=False)
    manual_seed: int = field(default=0)
    eval_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    eval_steps: int = field(default=200)
    save_steps: int = field(default=200)
    logging_steps: int = field(default=10)
    beta: float = field(default=0.1)
    use_wandb: bool = field(default=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_cyclic_ultrafeedback_pairwise_split(split: str) -> Dataset:
    """Expand a (responses, scores, preference_dimension) group into all strict pairs."""
    dataset_dir = Path("data_process/dataset/cyclic_ultrafeedback_all_pairs") / split
    rows: List[Dict[str, Any]] = []
    for shard_path in sorted(dataset_dir.glob("data-*.arrow")):
        with pa.memory_map(str(shard_path), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                for example in batch.to_pylist():
                    responses = example["responses"]
                    scores = example["scores"]
                    attribute = CYCLIC_ULTRAFEEDBACK_ATTRIBUTE_ALIASES.get(
                        example["preference_dimension"], example["preference_dimension"]
                    )
                    n = len(responses)
                    for i in range(n - 1):
                        for j in range(i + 1, n):
                            if scores[i] == scores[j]:
                                continue
                            if scores[i] > scores[j]:
                                ci, ri = i, j
                            else:
                                ci, ri = j, i
                            rows.append(
                                {
                                    "prompt": example["prompt"],
                                    "chosen": responses[ci],
                                    "rejected": responses[ri],
                                    "attribute": attribute,
                                }
                            )
    if not rows:
        raise ValueError(f"No pairwise rows built from {dataset_dir}")
    return Dataset.from_list(rows)


def build_dataset_pairwise(ds: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    """Tokenize prompt+chosen / prompt+rejected, mask the prompt prefix in labels."""

    def fmt(example: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(example["chosen"], list):
            chosen_msgs = example["chosen"]
            rejected_msgs = example["rejected"]
        else:
            prompt = example["prompt"]
            chosen_msgs = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": example["chosen"]},
            ]
            rejected_msgs = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": example["rejected"]},
            ]

        # The prompt prefix = apply_chat_template(messages[:-1], add_generation_prompt=True).
        # For all common chat templates this is a strict token-level prefix of the
        # full-conversation tokenization, so just slice by its length.
        prompt_text = tokenizer.apply_chat_template(
            chosen_msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        chosen_text = tokenizer.apply_chat_template(chosen_msgs, tokenize=False)
        rejected_text = tokenizer.apply_chat_template(rejected_msgs, tokenize=False)

        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
        tok_chosen = tokenizer(chosen_text, return_tensors="pt")
        tok_rejected = tokenizer(rejected_text, return_tensors="pt")
        prompt_len = len(prompt_ids)

        label_chosen = tok_chosen["input_ids"][0].clone()
        label_rejected = tok_rejected["input_ids"][0].clone()
        label_chosen[:prompt_len] = -100
        label_rejected[:prompt_len] = -100

        return {
            "input_ids_chosen": tok_chosen["input_ids"][0],
            "attention_mask_chosen": tok_chosen["attention_mask"][0],
            "label_chosen": label_chosen,
            "input_ids_rejected": tok_rejected["input_ids"][0],
            "attention_mask_rejected": tok_rejected["attention_mask"][0],
            "label_rejected": label_rejected,
            "attribute_id": attribute_to_id(example.get("attribute")),
        }

    ds = ds.map(fmt, batched=False, num_proc=10)
    ds = ds.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length,
        num_proc=10,
    )
    ds.set_format(type="torch")
    return ds


def load_all_datasets(data_path_arg: str, tokenizer: AutoTokenizer, max_length: int, seed: int):
    train_parts, eval_parts = [], []
    for data_path in data_path_arg.split("-"):
        if "ultrafeedback_per_attribute_pairwise" in data_path:
            ds = load_from_disk("data_process/dataset/ultrafeedback_per_attribute_pairwise")["train"]
            ds = build_dataset_pairwise(ds, tokenizer, max_length)
            sp = ds.train_test_split(test_size=0.01, seed=seed)
            train_ds, eval_ds = sp["train"], sp["test"]
        elif "cyclic_ultrafeedback_all_pairs" in data_path:
            train_ds = load_cyclic_ultrafeedback_pairwise_split("train")
            try:
                eval_ds = load_cyclic_ultrafeedback_pairwise_split("validation")
            except (FileNotFoundError, ValueError):
                eval_ds = load_cyclic_ultrafeedback_pairwise_split("test")
            train_ds = build_dataset_pairwise(train_ds, tokenizer, max_length)
            eval_ds = build_dataset_pairwise(eval_ds, tokenizer, max_length)
        else:
            raise ValueError(f"Data path {data_path} not supported")
        train_parts.append(train_ds)
        eval_parts.append(eval_ds)
    return concatenate_datasets(train_parts), concatenate_datasets(eval_parts)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
@dataclass
class PairwiseDPOCollator:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        def pad(side: str):
            batch = [
                {
                    "input_ids": f[f"input_ids_{side}"],
                    "attention_mask": f[f"attention_mask_{side}"],
                    "labels": f[f"label_{side}"],
                }
                for f in features
            ]
            return self.tokenizer.pad(
                batch,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

        c = pad("chosen")
        r = pad("rejected")
        return {
            "input_ids_chosen": c["input_ids"],
            "attention_mask_chosen": c["attention_mask"],
            "labels_chosen": c["labels"],
            "input_ids_rejected": r["input_ids"],
            "attention_mask_rejected": r["attention_mask"],
            "labels_rejected": r["labels"],
            "attribute_id": torch.tensor([f.get("attribute_id", -1) for f in features], dtype=torch.long),
            "return_loss": True,
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Overall + per-attribute pairwise accuracy."""
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = np.zeros(preds.shape, dtype=np.int64)  # index 0 = chosen
    metrics = accuracy_metric.compute(predictions=preds, references=labels)

    label_ids = eval_pred.label_ids
    if isinstance(label_ids, np.ndarray) and label_ids.ndim == 2 and label_ids.shape[1] > 1:
        attribute_ids = label_ids[:, 1].astype(np.int64)
        for attr_id in np.unique(attribute_ids[attribute_ids >= 0]):
            mask = attribute_ids == attr_id
            if not mask.any():
                continue
            name = ATTRIBUTE_ID_TO_NAME.get(int(attr_id), f"attr_{int(attr_id)}").replace("-", "_")
            metrics[f"accuracy_{name}"] = float((preds[mask] == labels[mask]).mean())
            metrics[f"count_{name}"] = int(mask.sum())
    return metrics


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class StandardDPOTrainer(Trainer):

    def __init__(self, *args, beta: float = 0.1, ref_model: Optional[PreTrainedModel] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        # accelerator.prepare_model handles DDP / DeepSpeed placement correctly.
        if ref_model is not None:
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
            self.ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        else:
            self.ref_model = None

    @staticmethod
    def _sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Sum of log p(token) over non-masked positions, fp32 for stability."""
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        valid = (shift_labels != -100).to(dtype=shift_logits.dtype)
        gather_labels = shift_labels.masked_fill(shift_labels == -100, 0)
        token_logp = torch.log_softmax(shift_logits, dim=-1).gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)
        return (token_logp * valid).sum(dim=1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        c_ids, c_attn, c_lbl = inputs["input_ids_chosen"], inputs["attention_mask_chosen"], inputs["labels_chosen"]
        r_ids, r_attn, r_lbl = inputs["input_ids_rejected"], inputs["attention_mask_rejected"], inputs["labels_rejected"]

        policy_logp_c = self._sequence_logps(
            model(input_ids=c_ids, attention_mask=c_attn, use_cache=False).logits, c_lbl
        )
        policy_logp_r = self._sequence_logps(
            model(input_ids=r_ids, attention_mask=r_attn, use_cache=False).logits, r_lbl
        )

        with torch.no_grad():
            ref_logp_c = self._sequence_logps(
                self.ref_model(input_ids=c_ids, attention_mask=c_attn, use_cache=False).logits, c_lbl
            )
            ref_logp_r = self._sequence_logps(
                self.ref_model(input_ids=r_ids, attention_mask=r_attn, use_cache=False).logits, r_lbl
            )

        chosen_reward = self.beta * (policy_logp_c - ref_logp_c)
        rejected_reward = self.beta * (policy_logp_r - ref_logp_r)
        margin = chosen_reward - rejected_reward
        loss = -F.logsigmoid(margin).mean()

        if model.training and self.accelerator.is_main_process and wandb.run is not None:
            wandb.log(
                {
                    "train/loss": float(loss.detach().cpu().item()),
                    "train/accuracy": float((margin.detach() > 0).float().mean().item()),
                    "train/margin": float(margin.detach().mean().item()),
                    "train/chosen_reward": float(chosen_reward.detach().mean().item()),
                    "train/rejected_reward": float(rejected_reward.detach().mean().item()),
                },
                step=self.state.global_step,
            )

        if return_outputs:
            return loss, {"chosen_reward": chosen_reward, "rejected_reward": rejected_reward, "margin": margin}
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        del ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, out = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None

        # logits = [chosen_reward, rejected_reward] -> argmax==0 means correct
        logits = torch.stack([out["chosen_reward"], out["rejected_reward"]], dim=1)
        logits = nested_detach(logits).softmax(dim=1)
        labels = torch.zeros(logits.shape[0], device=logits.device)
        if "attribute_id" in inputs:
            labels = torch.stack(
                (labels, inputs["attribute_id"].to(labels.device, dtype=labels.dtype)), dim=1
            )
        return loss.detach(), logits, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    torch.manual_seed(args.manual_seed)

    if not (0.0 < args.downsample_rate <= 1.0):
        raise ValueError("`downsample_rate` must be in (0, 1].")
    if args.beta <= 0:
        raise ValueError("`beta` must be > 0.")

    if accelerator.is_main_process:
        print("=== Arguments ===")
        for k, v in vars(args).items():
            print(f"  {k:<32} {v}")
        if args.use_wandb:
            wandb.init(project="MultiRewardLearning", name=args.wandb_name, config=vars(args))

    output_name = f"{args.log_dir}/{args.base_model.split('/')[-1]}_{args.wandb_name}"

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.model_max_length = args.max_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Data ----
    train_dataset, eval_dataset = load_all_datasets(args.data_path, tokenizer, args.max_length, args.manual_seed)
    if args.downsample_rate < 1.0:
        keep = max(1, int(len(train_dataset) * args.downsample_rate))
        train_dataset = train_dataset.shuffle(seed=args.manual_seed).select(range(keep))
    if accelerator.is_main_process:
        print(f"train rows: {len(train_dataset)} | eval rows: {len(eval_dataset)}")

    # ---- Models ----
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    policy_model.resize_token_embeddings(len(tokenizer))
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    policy_model.config.use_cache = False

    reference_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    reference_model.resize_token_embeddings(len(tokenizer))
    reference_model.config.pad_token_id = tokenizer.pad_token_id
    reference_model.config.use_cache = False

    # ---- TrainingArguments ----
    training_args = TrainingArguments(
        output_dir=os.path.join(output_name, "logs"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=3,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        label_names=[],
        bf16=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        warmup_ratio=0.05,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        run_name=args.wandb_name,
        report_to="wandb" if args.use_wandb else "none",
        ddp_find_unused_parameters=False,
        seed=args.manual_seed,
    )
    if args.eval_only and getattr(training_args, "deepspeed", None) is not None:
        training_args.deepspeed = None

    # ---- Trainer ----
    trainer = StandardDPOTrainer(
        model=policy_model,
        ref_model=reference_model,
        beta=args.beta,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=PairwiseDPOCollator(tokenizer=tokenizer, max_length=args.max_length),
    )

    if args.eval_only:
        m = trainer.evaluate()
        trainer.log_metrics("eval_only", m)
        trainer.save_metrics("eval_only", m)
    else:
        trainer.train()
        m = trainer.evaluate()
        trainer.log_metrics("eval_final", m)
        trainer.save_metrics("eval_final", m)
        trainer.save_model(output_name)

    if accelerator.is_main_process and args.use_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()