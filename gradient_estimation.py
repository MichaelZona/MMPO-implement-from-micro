"""
Standalone validation script for Algorithm 1 (Gradient-based Estimation for
Ranking Model Probability) from the MMPO paper.

Setting: a = 1 anchor per ranking instance. The anchor is the TOP-RANKED
response (index 0 after sorting by ground-truth score, descending). This
matches Algorithm 1's "top a ranked responses as anchors" with a = 1.

Reward definition (length-normalized):
    r_j = (1 / L_resp_j) * sum_{t in response} log pi_base(y_t | x, y_<t)
i.e. the negative cross-entropy of the response tokens.
Equivalently:
    CE_j = -r_j
where CE_j is the mean per-token cross-entropy over response tokens.

For each prompt with m candidate responses:
  1. Anchor = candidate 0. Run forward + backward:
       r_anchor:  scalar
       h_anchor:  [L_a, D]   full inputs_embeds
       g_anchor:  [L_a, D]   d r_anchor / d h_anchor (full tensor)
  2. For each non-anchor j: forward only, get exact r_j and h_j.
  3. Taylor estimate (sequence-level inner product):
       hat_r_j = r_anchor + <g_anchor, h_j - h_anchor>
     (h_j is zero-padded to match L_a; positions beyond L_a in either
      direction contribute via the natural extension explained inline.)
  4. Report two metrics over non-anchor candidates:
     - Relative MSE on rewards:        rel_MSE(r)  = mean (hat_r - r)^2 / mean r^2
     - Relative MSE on CE losses:      rel_MSE(CE) = mean (hat_CE - CE)^2 / mean CE^2
       where hat_CE_j = -hat_r_j and CE_j = -r_j.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import argparse
import json

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data_path", type=str, default="cyclic_ultrafeedback_all_pairs")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_examples", type=int, default=200,
                        help="Cap the number of ranking instances to evaluate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="taylor_validation.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Dataset loading (matches the main training script's logic)
# ----------------------------------------------------------------------------

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


def _prompt_to_messages(prompt: Any) -> List[Dict[str, str]]:
    if isinstance(prompt, list):
        messages: List[Dict[str, str]] = []
        for item in prompt:
            if isinstance(item, dict) and "role" in item and "content" in item:
                messages.append({"role": str(item["role"]), "content": str(item["content"])})
        if messages:
            return messages
    return [{"role": "user", "content": str(prompt)}]


def _response_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict) and "content" in response:
        return str(response["content"])
    return str(response)


def _truncate_prompt(input_ids: List[int], max_prompt_length: int) -> List[int]:
    if len(input_ids) <= max_prompt_length:
        return input_ids
    return input_ids[-max_prompt_length:]


def _tokenize_listwise_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
    max_prompt_length: int,
) -> Dict[str, Any]:
    prompt_messages = _prompt_to_messages(example["prompt"])
    prompt_template = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_template, add_special_tokens=False)["input_ids"]
    prompt_ids = _truncate_prompt(prompt_ids, max_prompt_length)

    responses = example["responses"]
    scores = example.get("scores")
    if scores is not None and len(scores) == len(responses):
        order = sorted(range(len(responses)), key=lambda idx: float(scores[idx]), reverse=True)
        responses = [responses[idx] for idx in order]

    candidate_input_ids: List[List[int]] = []
    candidate_response_masks: List[List[int]] = []
    for response in responses:
        response_text = _response_to_text(response)
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
            response_ids = response_ids + [eos_id]
        response_start = len(prompt_ids)
        input_ids = prompt_ids + response_ids
        if len(input_ids) > max_length:
            overflow = len(input_ids) - max_length
            if overflow < len(prompt_ids):
                new_prompt = prompt_ids[overflow:]
                input_ids = new_prompt + response_ids
                response_start = len(new_prompt)
            else:
                input_ids = response_ids[overflow - len(prompt_ids):]
                response_start = 0
        candidate_input_ids.append(input_ids)
        candidate_response_masks.append(
            [0] * response_start + [1] * (len(input_ids) - response_start)
        )

    return {
        "candidate_input_ids": candidate_input_ids,
        "candidate_response_mask": candidate_response_masks,
        "preference_dimension": example.get("preference_dimension"),
    }


def load_listwise_split(
    data_path: str,
    split: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    max_prompt_length: int,
) -> Dataset:
    dataset_dir = Path(resolve_local_dataset_dir(data_path)) / split
    rows = []
    for shard_path in sorted(dataset_dir.glob("data-*.arrow")):
        with pa.memory_map(str(shard_path), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                rows.extend(batch.to_pylist())
    if not rows:
        raise ValueError(f"No listwise rows built from {dataset_dir}")
    dataset = Dataset.from_list(rows)
    dataset = dataset.map(
        lambda ex: _tokenize_listwise_example(ex, tokenizer, max_length, max_prompt_length),
        batched=False,
        num_proc=4,
    )
    dataset = dataset.filter(
        lambda x: len(x["candidate_input_ids"]) >= 2
        and all(sum(mask) > 0 for mask in x["candidate_response_mask"]),
        num_proc=4,
    )
    return dataset


# ----------------------------------------------------------------------------
# Forward passes
# ----------------------------------------------------------------------------

def _length_normalized_reward(
    logits: torch.Tensor,            # [1, L, V]
    input_ids: torch.Tensor,         # [1, L]
    response_mask: torch.Tensor,     # [1, L]
) -> torch.Tensor:
    """Length-normalized log-prob of response tokens.

    r = (1 / L_resp) * sum_{t in response} log p(y_t | y_<t)
      = - cross_entropy_per_token over the response

    So CE_per_token = -r.
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = response_mask[:, 1:].to(dtype=shift_logits.dtype)
    token_logp = torch.log_softmax(shift_logits, dim=-1).gather(
        -1, shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    total_logp = (token_logp * shift_mask).sum()
    response_len = shift_mask.sum().clamp_min(1.0)
    # return total_logp / response_len
    return total_logp


def compute_reward_h_and_grad(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,        # [L]
    response_mask: torch.Tensor,    # [L]
    embed_layer: torch.nn.Embedding,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For the ANCHOR: compute r, full inputs_embeds h, and full g = d r / d h.

    Returns:
        reward:   scalar (length-normalized)
        h:        [L, D]  full input-embedding tensor for this sequence
        g:        [L, D]  d reward / d h  (same shape as h)
    """
    input_ids = input_ids.unsqueeze(0)         # [1, L]
    response_mask = response_mask.unsqueeze(0) # [1, L]

    inputs_embeds = embed_layer(input_ids).detach().clone()
    inputs_embeds.requires_grad_(True)

    outputs = model(
        inputs_embeds=inputs_embeds,
        use_cache=False,
    )
    reward = _length_normalized_reward(outputs.logits, input_ids, response_mask)

    grad_inputs = torch.autograd.grad(
        outputs=reward,
        inputs=inputs_embeds,
        retain_graph=False,
        create_graph=False,
    )[0]                                                # [1, L, D]

    h = inputs_embeds.detach()[0]                       # [L, D]
    g = grad_inputs.detach()[0]                         # [L, D]
    return reward.detach(), h, g


@torch.no_grad()
def compute_reward_and_h(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,        # [L]
    response_mask: torch.Tensor,    # [L]
    embed_layer: torch.nn.Embedding,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """For NON-ANCHOR candidates: compute exact r and full inputs_embeds h."""
    input_ids = input_ids.unsqueeze(0)
    response_mask = response_mask.unsqueeze(0)

    inputs_embeds = embed_layer(input_ids)              # [1, L, D]
    outputs = model(
        inputs_embeds=inputs_embeds,
        use_cache=False,
    )
    reward = _length_normalized_reward(outputs.logits, input_ids, response_mask)
    h = inputs_embeds.detach()[0]                       # [L, D]
    return reward.detach(), h


# ----------------------------------------------------------------------------
# Validation: a = 1 anchor (top-ranked), m-1 candidates approximated
# ----------------------------------------------------------------------------

def taylor_estimate(
    r_anchor: torch.Tensor,    # scalar
    h_anchor: torch.Tensor,    # [L_a, D]
    g_anchor: torch.Tensor,    # [L_a, D]
    h_j: torch.Tensor,         # [L_j, D]
) -> torch.Tensor:

    L_a, D = h_anchor.shape
    L_j = h_j.shape[0]
    L_max = max(L_a, L_j)

    if L_a < L_max:
        pad_h = torch.zeros(L_max - L_a, D, dtype=h_anchor.dtype, device=h_anchor.device)
        h_anchor = torch.cat([h_anchor, pad_h], dim=0)
        pad_g = torch.zeros(L_max - L_a, D, dtype=g_anchor.dtype, device=g_anchor.device)
        g_anchor = torch.cat([g_anchor, pad_g], dim=0)
    if L_j < L_max:
        pad_h = torch.zeros(L_max - L_j, D, dtype=h_j.dtype, device=h_j.device)
        h_j = torch.cat([h_j, pad_h], dim=0)

    delta = h_j - h_anchor                              # [L_max, D]
    correction = (g_anchor * delta).sum()               # scalar
    return r_anchor + correction


def validate_example(
    candidate_input_ids: List[List[int]],
    candidate_response_mask: List[List[int]],
    model: AutoModelForCausalLM,
    embed_layer: torch.nn.Embedding,
    device: torch.device,
) -> Dict[str, Any]:
    """One ranking instance. Top-ranked response (index 0) is the anchor (a=1).
    Approximate rewards for the remaining m-1 responses via Taylor.
    """
    m = len(candidate_input_ids)
    if m < 2:
        return {}

    # 1) Anchor: candidate index 0 (best by ground-truth score, since
    #    load_listwise_split already sorted responses by score descending).
    anchor_ids = torch.tensor(candidate_input_ids[0], dtype=torch.long, device=device)
    anchor_mask = torch.tensor(candidate_response_mask[0], dtype=torch.long, device=device)
    r_anchor, h_anchor, g_anchor = compute_reward_h_and_grad(
        model=model,
        input_ids=anchor_ids,
        response_mask=anchor_mask,
        embed_layer=embed_layer,
    )
    r_anchor = r_anchor.float()
    h_anchor = h_anchor.float()
    g_anchor = g_anchor.float()

    # 2) Remaining m-1 candidates: exact reward + full inputs_embeds.
    r_true_others: List[torch.Tensor] = []
    r_hat_others: List[torch.Tensor] = []
    rel_distance: List[torch.Tensor] = []
    nonanchor_response_indices: List[int] = []
    for j in range(1, m):
        ids_t = torch.tensor(candidate_input_ids[j], dtype=torch.long, device=device)
        mask_t = torch.tensor(candidate_response_mask[j], dtype=torch.long, device=device)
        r_j, h_j = compute_reward_and_h(model, ids_t, mask_t, embed_layer)
        r_j = r_j.float()
        h_j = h_j.float()
        r_hat = taylor_estimate(r_anchor, h_anchor, g_anchor, h_j)
        r_true_others.append(r_j)
        r_hat_others.append(r_hat)
        scale = torch.maximum(r_j.abs(), r_hat.abs()) + 1e-12
        rel_distance.append(((r_hat - r_j) / scale).abs())
        nonanchor_response_indices.append(j)
    r_true_others = torch.stack(r_true_others)         # [m-1]
    r_hat_others = torch.stack(r_hat_others)           # [m-1]
    rel_distance = torch.stack(rel_distance)           # [m-1]
    
    # 3 Relative MSE on (length-normalized) reward.
    scale = torch.maximum(r_true_others.abs(), r_hat_others.abs()) + 1e-12
    rel_mse_reward = ((((r_hat_others - r_true_others) / scale) ** 2).mean()).item()
    r_true_all = torch.cat([r_anchor.unsqueeze(0), r_true_others])
    r_hat_all = torch.cat([r_anchor.unsqueeze(0), r_hat_others])
    # Backward-compat: keep per-example rel_distance list aligned with full [m]
    # candidates, where index 0 (anchor) is always 0.0.
    rel_distance_all = torch.cat(
        [torch.tensor([0.0], dtype=rel_distance.dtype, device=rel_distance.device), rel_distance]
    )
    nonanchor_atoms: List[Dict[str, Any]] = []
    for atom_idx, response_idx in enumerate(nonanchor_response_indices):
        scale = torch.maximum(r_true_others[atom_idx].abs(), r_hat_others[atom_idx].abs()) + 1e-12
        rel_mse_atom = (((r_hat_others[atom_idx] - r_true_others[atom_idx]) / scale) ** 2).item()
        nonanchor_atoms.append(
            {
                "response_index": int(response_idx),
                "r_true": float(r_true_others[atom_idx].item()),
                "r_hat": float(r_hat_others[atom_idx].item()),
                "rel_distance": float(rel_distance[atom_idx].item()),
                "rel_mse_reward": float(rel_mse_atom),
                "abs_err": float((r_hat_others[atom_idx] - r_true_others[atom_idx]).abs().item()),
            }
        )

    return {
        "m": m,
        "r_true": r_true_all.detach().cpu().tolist(),
        "r_hat": r_hat_all.detach().cpu().tolist(),
        "mean_rel_reward_err": rel_mse_reward,
        "rel_distance": rel_distance_all.detach().cpu().tolist(),
        "nonanchor_atoms": nonanchor_atoms,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading tokenizer / model from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()
    # Keep model weights in the graph so autograd can build the path from
    # inputs_embeds -> reward.
    for p in model.parameters():
        p.requires_grad_(True)

    embed_layer = model.get_input_embeddings()

    print(f"Loading dataset split: {args.data_path}/{args.split}")
    dataset = load_listwise_split(
        data_path=args.data_path,
        split=args.split,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    if args.max_examples > 0 and len(dataset) > args.max_examples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_examples))
    print(f"Evaluating on {len(dataset)} ranking instances. "
          f"(a = 1 anchor = top-ranked, m-1 approximated; length-normalized reward)")

    rel_distance: List[float] = []
    nonanchor_atomic_records: List[Dict[str, Any]] = []
    per_example_records: List[Dict[str, Any]] = []

    for ex_idx, example in enumerate(tqdm(dataset, desc="Validating Algorithm 1 (a=1)")):
        try:
            result = validate_example(
                candidate_input_ids=example["candidate_input_ids"],
                candidate_response_mask=example["candidate_response_mask"],
                model=model,
                embed_layer=embed_layer,
                device=device,
            )
        except RuntimeError as e:
            print(f"[skip] example {ex_idx}: {e}")
            torch.cuda.empty_cache()
            continue

        if not result:
            continue

        # Backward-compatible per-example vector includes anchor at index 0.
        for atom in result.get("nonanchor_atoms", []):
            nonanchor_atomic_records.append(
                {
                    "example_idx": int(ex_idx),
                    "preference_dimension": example.get("preference_dimension"),
                    **atom,
                }
            )
        per_example_records.append({
            "idx": ex_idx,
            "m": result["m"],
            "preference_dimension": example.get("preference_dimension"),
            "r_true": result["r_true"],
            "r_hat": result["r_hat"],
            "mean_rel_reward_err": result["mean_rel_reward_err"],
            "rel_distance": result["rel_distance"]
        })

    def _stats(arr: np.ndarray) -> Dict[str, float]:
        if arr.size == 0:
            return {"mean": 0.0, "median": 0.0, "std": 0.0}
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
        }

    rel_distance_arr = np.array(rel_distance, dtype=np.float64)
    atomic_rel_dist = np.array([row["rel_distance"] for row in nonanchor_atomic_records], dtype=np.float64)
    atomic_rel_mse = np.array([row["rel_mse_reward"] for row in nonanchor_atomic_records], dtype=np.float64)

    distance_bin_edges = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")]
    distance_buckets: Dict[str, Dict[str, float]] = {}
    for lo, hi in zip(distance_bin_edges[:-1], distance_bin_edges[1:]):
        if np.isinf(hi):
            key = f"[{lo:.2f}, +inf)"
            mask = atomic_rel_dist >= lo
        else:
            key = f"[{lo:.2f}, {hi:.2f})"
            mask = (atomic_rel_dist >= lo) & (atomic_rel_dist < hi)
        bucket_dist = atomic_rel_dist[mask]
        bucket_rel_mse = atomic_rel_mse[mask]
        distance_buckets[key] = {
            "count": int(bucket_dist.size),
            "mean_rel_distance": float(bucket_dist.mean()) if bucket_dist.size > 0 else 0.0,
            "mean_rel_mse_reward": float(bucket_rel_mse.mean()) if bucket_rel_mse.size > 0 else 0.0,
            "var_rel_mse_reward": float(bucket_rel_mse.var()) if bucket_rel_mse.size > 0 else 0.0,
        }

    summary = {
        "n_atomic_nonanchor": int(atomic_rel_dist.size),
        "distance_buckets": distance_buckets,
    }

    print("\n=== Taylor approximation with a=1 anchor (top-ranked), length-normalized reward ===")
    print("Distance bucket stats:")
    for bucket, stats in summary["distance_buckets"].items():
        print(
            f"  {bucket:>16}  count={stats['count']:>6}  "
            f"mean_rel_distance={stats['mean_rel_distance']:.6f}  "
            f"mean_rel_mse_reward={stats['mean_rel_mse_reward']:.6f}  "
            f"var_rel_mse_reward={stats['var_rel_mse_reward']:.6f}"
        )

    payload = {
        "base_model": args.base_model,
        "data_path": args.data_path,
        "split": args.split,
        "num_anchors": 1,
        "anchor_strategy": "top_ranked",
        "reward_definition": "length_normalized_log_prob",
        "summary": summary,
        "atomic_nonanchor_records": nonanchor_atomic_records,
        "examples": per_example_records,
    }
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved per-example results to: {args.output_json}")


if __name__ == "__main__":
    main()