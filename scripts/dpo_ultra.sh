#!/bin/bash
lr=5e-7 
beta=0.1                                 # DPO temperature, 0.05 ~ 0.5
gradient_accumulation_steps=1
bs=8
data_path="ultrafeedback_per_attribute_pairwise"
num_train_epochs=2
downsample_rate=1

mkdir -p log

for seed in 42 44 46 ; do
    run_name=dpo_${data_path}_beta${beta}_lr${lr}_ep${num_train_epochs}_seed${seed}
    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file configs/config.yaml \
        --num_processes=1 \
        --main_process_port=29506 \
        --gradient_accumulation_steps=$gradient_accumulation_steps \
        dpo.py \
            --learning_rate=$lr \
            --beta=$beta \
            --wandb_name=$run_name \
            --data_path=$data_path \
            --per_device_train_batch_size=$bs \
            --num_train_epochs=$num_train_epochs \
            --gradient_accumulation_steps=$gradient_accumulation_steps \
            --base_model="Qwen/Qwen3-0.6B" \
            --downsample_rate=$downsample_rate \
            --manual_seed=$seed \
        | tee -a log/${run_name}.log
done