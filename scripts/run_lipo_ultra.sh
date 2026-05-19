#!/usr/bin/env bash
set -uo pipefail

lr=2e-3
gradient_accumulation_steps=1
bs=2
data_path="cyclic_ultrafeedback_all_pairs"
num_train_epochs=3
downsample_rate=1
eval_steps=50
save_steps=50
logging_steps=10

mkdir -p log

for seed in 42 44 46; do
  tag="lipo_${data_path}_epochs${num_train_epochs}_seed${seed}"
  echo "=== Running seed=${seed} ==="
  CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/config.yaml \
    --num_processes=1 --main_process_port=29516 --gradient_accumulation_steps=$gradient_accumulation_steps lipo.py \
    --learning_rate=$lr \
    --wandb_name=$tag --run_name=lipo-${seed} \
    --data_path=$data_path \
    --per_device_train_batch_size=$bs --per_device_eval_batch_size=$bs \
    --num_train_epochs=$num_train_epochs \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --base_model="Qwen/Qwen3-0.6B" \
    --downsample_rate=$downsample_rate \
    --manual_seed=$seed \
    --eval_steps=$eval_steps --save_steps=$save_steps --logging_steps=$logging_steps \
    | tee -a log/${tag}.log
done
