
# python gradient_estimation.py \
#     --base_model Qwen/Qwen3-0.6B \
#     --data_path helpsteer2_per_attribute_pairwise \
#     --split validation \
#     --max_examples 200 \
#     --output_json gradient_estimation.json \


# python gradient_estimation.py \
#     --base_model google/gemma-2-2b-it \
#     --data_path helpsteer2_per_attribute_pairwise \
#     --split validation \
#     --max_examples 200 \
#     --output_json gradient_estimation.json \

# python gradient_estimation.py \
#     --base_model deepseek-ai/deepseek-llm-7b-chat \
#     --data_path helpsteer2_per_attribute_pairwise \
#     --split validation \
#     --max_examples 200 \
#     --output_json gradient_estimation.json \


python gradient_estimation.py \
    --base_model meta-llama/Llama-2-13b-hf \
    --data_path helpsteer2_per_attribute_pairwise \
    --split validation \
    --max_examples 200 \
    --output_json gradient_estimation.json \
    --quantization 4bit


# python gradient_estimation.py \
#     --base_model codellama/CodeLlama-34b-Instruct-hf \
#     --data_path helpsteer2_per_attribute_pairwise \
#     --split validation \
#     --max_examples 200 \
#     --output_json gradient_estimation.json \
#     --quantization 4bit