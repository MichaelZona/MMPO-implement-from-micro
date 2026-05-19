#!/usr/bin/env bash
set -uo pipefail

lr=2e-3
gradient_accumulation_steps=1
bs=2
num_heads=4
data_path="cyclic_ultrafeedback_all_pairs"
num_train_epochs=3
downsample_rate=1
eval_steps=50
save_steps=50
logging_steps=10
seed=42

mkdir -p log

for em_temperature in 1.0 1.2 1.5 2.0; do
  for m_step_updates in 1 2 3 4 5; do
    tag="mmpo_grid_${data_path}_heads${num_heads}_epochs${num_train_epochs}_temp${em_temperature}_mstep${m_step_updates}_seed${seed}"
    echo "=== Running: em_temperature=${em_temperature}  m_step_updates=${m_step_updates} ==="
    CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/config.yaml \
      --num_processes=1 --main_process_port=29516 --gradient_accumulation_steps=$gradient_accumulation_steps mmpo.py \
      --learning_rate=$lr --num_heads=$num_heads \
      --wandb_name=$tag --run_name=$tag \
      --data_path=$data_path \
      --per_device_train_batch_size=$bs --per_device_eval_batch_size=$bs \
      --num_train_epochs=$num_train_epochs \
      --gradient_accumulation_steps=$gradient_accumulation_steps \
      --base_model="Qwen/Qwen3-0.6B" \
      --downsample_rate=$downsample_rate \
      --manual_seed=$seed \
      --em_temperature=$em_temperature \
      --m_step_updates=$m_step_updates \
      --eval_steps=$eval_steps --save_steps=$save_steps --logging_steps=$logging_steps \
      --use_wandb=False \
      --save_checkpoint=False \
      | tee -a log/${tag}.log || echo "WARNING: run failed (temp=${em_temperature}, mstep=${m_step_updates}), continuing grid..."
  done
done
