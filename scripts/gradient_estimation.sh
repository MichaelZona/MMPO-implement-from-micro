python gradient_estimation.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --data_path cyclic_ultrafeedback_all_pairs \
    --split validation \
    --max_examples 200 \
    --output_json gradient_estimation.json