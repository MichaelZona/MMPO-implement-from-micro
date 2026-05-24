#!/bin/bash
# ---------------------------------------------------------------------------
# DPO training with QLoRA + Taylor gradient approximation (Qwen3-0.6B)
# ---------------------------------------------------------------------------
lr=5e-6
beta=0.01
bs=2
gradient_accumulation_steps=4   # effective batch = 8
max_length=1024

data_path="cyclic_ultrafeedback_all_pairs"
attribute=""
downsample_rate=1.0
num_train_epochs=1

# LoRA / QLoRA
use_peft=True
load_in_4bit=True
lora_r=64
lora_alpha=64
lora_dropout=0.05

mkdir -p log

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for seed in 42; do
    run_name=dpo_gradient_qlora_${data_path}_ds${downsample_rate}_beta${beta}_lr${lr}_ep${num_train_epochs}_seed${seed}

    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file configs/config.yaml \
        --num_processes=1 \
        --main_process_port=29506 \
        dpo.py \
            --learning_rate=$lr \
            --beta=$beta \
            --wandb_name=$run_name \
            --data_path=$data_path \
            --attribute="$attribute" \
            --per_device_train_batch_size=$bs \
            --per_device_eval_batch_size=$bs \
            --num_train_epochs=$num_train_epochs \
            --gradient_accumulation_steps=$gradient_accumulation_steps \
            --max_length=$max_length \
            --base_model="Qwen/Qwen3-0.6B" \
            --downsample_rate=$downsample_rate \
            --manual_seed=$seed \
            --use_peft=$use_peft \
            --load_in_4bit=$load_in_4bit \
            --lora_r=$lora_r \
            --lora_alpha=$lora_alpha \
            --lora_dropout=$lora_dropout \
            --use_taylor_approx=True \
            --taylor_anchor_strategy=random \
        | tee -a log/${run_name}.log
done
