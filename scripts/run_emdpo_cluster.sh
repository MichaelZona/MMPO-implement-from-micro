#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$PROJECT_ROOT"

lr=2e-3
gradient_accumulation_steps=1
bs=4
eval_bs=4
num_heads=4
loss_type=em_dpo
num_train_epochs=5
downsample_rate=0.01
em_temperature=1.0
em_prior_smoothing=1e-3

# Use disagreement listwise dataset (converted to pairwise for training;
# cluster_posterior metrics are computed on listwise eval split).
data_path="/home/michael/project/MMPO-implement-from-micro/data_process/dataset/ultrafeedback_disagreement"

mkdir -p log

for seed in 42 44 46; do
  run_name="emdpo_cluster_${loss_type}_disagreement_heads${num_heads}_epoch${num_train_epochs}_seed${seed}"
  log_file="log/${run_name}.log"
  rm -f "$log_file"

  CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file configs/config.yaml \
    --num_processes=1 \
    --main_process_port=29506 \
    em-dpo.py \
      --learning_rate=$lr \
      --loss_type=$loss_type \
      --num_heads=$num_heads \
      --wandb_name=$run_name \
      --data_path="$data_path" \
      --per_device_train_batch_size=$bs \
      --per_device_eval_batch_size=$eval_bs \
      --num_train_epochs=$num_train_epochs \
      --gradient_accumulation_steps=$gradient_accumulation_steps \
      --base_model="Qwen/Qwen3-0.6B" \
      --downsample_rate=$downsample_rate \
      --manual_seed=$seed \
      --em_temperature=$em_temperature \
      --em_prior_smoothing=$em_prior_smoothing \
      | tee -a "$log_file"
done
