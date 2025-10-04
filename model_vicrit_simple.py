import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import math
from datasets import load_dataset
import io

# Simple prompt for smaller models
simple_prompt = '''Look at this image and description. Find the one thing that is wrong or doesn't match the image.

Description: {}

What doesn't match:'''

def dump_to_jsonl(obj: list[dict], path: str):
    with open(path, 'w') as file:
        file.writelines([json.dumps(x) + '\n' for x in obj])

def eval_model_simple(args):
    """Simple evaluation using Qwen model (which works better locally)"""
    
    print(f"Loading model: {args.model_id}")
    
    # Use Qwen model which has better local support
    if "qwen" in args.model_id.lower():
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
            processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading model with device_map='auto': {e}")
            print("Trying with device_map=None and manual device placement...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                device_map=None,
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
            if torch.cuda.is_available():
                model = model.cuda()
            processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    else:
        print("This simplified version only supports Qwen models.")
        print("Please use: Qwen/Qwen2.5-VL-7B-Instruct")
        return
    
    # Load dataset
    print("Loading ViCrit dataset...")
    questions = list(load_dataset("russwang/ViCrit-Bench", split="train"))
    
    if args.max_samples:
        questions = questions[:args.max_samples]
        print(f"Using {len(questions)} samples for evaluation")
    
    answers_file = os.path.expanduser(args.answers_file)
    answers_dir = os.path.dirname(answers_file)
    if answers_dir:  # Only create directory if it's not empty (i.e., not current directory)
        os.makedirs(answers_dir, exist_ok=True)
    
    final_response = []
    
    print("Starting evaluation...")
    
    for i, data in enumerate(tqdm(questions)):
        try:
            # Load image
            img = Image.open(io.BytesIO(data['image']))
            if img.height < 28 or img.width < 28:
                continue
            
            # Prepare prompt
            prompt = simple_prompt.format(data['changed_caption'])
            
            # Process with Qwen
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = processor(
                text=[text], 
                images=[img], 
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=False
                )
            
            response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            response = response.split("What doesn't match:")[-1].strip()
            
            # Store result
            result = {
                'response': response,
                'changed_caption': data['changed_caption'],
                'original_noun_phrases': data.get('original_noun_phrases', ''),
                'changed_noun_phrases': data.get('changed_noun_phrases', ''),
                'image': []
            }
            final_response.append(result)
            
            # Print progress
            if i % 10 == 0:
                print(f"Processed {i} samples...")
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Save results
    dump_to_jsonl(final_response, answers_file)
    print(f"Saved {len(final_response)} results to {answers_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    
    args = parser.parse_args()
    eval_model_simple(args)
