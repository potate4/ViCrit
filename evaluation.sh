#!/bin/bash
model_id="Qwen/Qwen2.5-VL-72B-Instruct"
model_name="Qwen/Qwen2.5-VL-72B-Instruct"

datasets=("captioncritic")


for dataset in "${datasets[@]}"; do
   output_prefix="./eval_files/${dataset}/answers/${model_id}"

   model_script="./model_${dataset}_qwen.py"

   python $model_script \
           --model_id $model_name \
           --answers-file "${output_prefix}.jsonl" \
           --batch-size 16
done

