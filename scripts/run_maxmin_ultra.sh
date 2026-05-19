lr=2e-3
gradient_accumulation_steps=1
bs=8
num_heads=4
data_path="cyclic_ultrafeedback_all_pairs"
loss_type=maxmin_em
num_train_epochs=5
# seed=42
downsample_rate=1

loss_components=$loss_type

mkdir -p log

for seed in 42 44 46 ; do
    CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/config.yaml \
        --num_processes=1 --main_process_port=29506 --gradient_accumulation_steps=$gradient_accumulation_steps maxmin-rlhf.py \
        --learning_rate=$lr --loss_type=$loss_type --num_heads=$num_heads \
        --wandb_name=maxmin_${loss_type}_${data_path}_heads${num_heads}_epoch${num_train_epochs}_seed${seed} --data_path=$data_path \
        --per_device_train_batch_size=$bs --num_train_epochs=$num_train_epochs \
        --gradient_accumulation_steps=$gradient_accumulation_steps \
        --base_model="Qwen/Qwen3-0.6B" \
        --downsample_rate=$downsample_rate \
        --manual_seed=$seed \
        | tee -a log/maxmin_${loss_type}_${data_path}_heads${num_heads}_lr${lr}_${loss_components}_seed${seed}.log # 2>&1

done
