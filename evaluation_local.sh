#!/bin/bash

# Local evaluation script for smaller ViCrit models
# Choose one of the following model configurations:

# Option 1: LLaVA 7B (Recommended for most local setups)
model_id="llava-v1.5-7b"
model_name="llava-v1.5-7b"

# Option 2: LLaVA 13B (Better performance, needs more VRAM)
# model_id="llava-v1.5-13b"
# model_name="llava-v1.5-13b"

# Option 3: Qwen 7B (Alternative to LLaVA)
# model_id="Qwen/Qwen2.5-VL-7B-Instruct"
# model_name="Qwen/Qwen2.5-VL-7B-Instruct"

# Option 4: InstructBLIP 7B (Lightweight option)
# model_id="Salesforce/instructblip-vicuna-7b"
# model_name="Salesforce/instructblip-vicuna-7b"

echo "Using model: $model_id"
echo "This model is suitable for local hardware"

# Create output directory
output_dir="./eval_files/local/answers"
mkdir -p $output_dir

# Run evaluation with local model
python model_vicrit_local.py \
    --model_id $model_name \
    --answers-file "${output_dir}/${model_id}.jsonl" \
    --batch-size 1 \
    --max-samples 100  # Start with 100 samples for testing

echo "Evaluation completed! Results saved to: ${output_dir}/${model_id}.jsonl"
echo "To score the results, run: python score.py" 