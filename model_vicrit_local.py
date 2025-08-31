import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import math
from datasets import load_dataset
import numpy as np
import io

# Simplified prompt for smaller models
multichoice_prompt = '''Look at this image and description. There is one hallucination (false information) in the description. Find the hallucination phrase and answer with just that phrase.

Description: {} 

Hallucination phrase:'''

def dump_to_jsonl(obj: list[dict], path: str):
    with open(path, 'w') as file:
        file.writelines([json.dumps(x) + '\n' for x in obj])

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def eval_model_local(args):
    """Local evaluation function for smaller models"""
    
    # Load model and processor
    print(f"Loading model: {args.model_id}")
    
    # For LLaVA models
    if "llava" in args.model_id.lower():
        try:
            from transformers import LlavaForConditionalGeneration
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_id,
                dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            processor = AutoProcessor.from_pretrained(args.model_id)
        except Exception as e:
            print(f"Error loading LLaVA model: {e}")
            print("Trying alternative LLaVA model names...")
            # Try alternative LLaVA model names
            alternative_models = [
                "llava-hf/llava-1.5-7b-hf",
                "llava-hf/llava-1.5-13b-hf",
                "llava-hf/llava-1.6-34b-hf"
            ]
            for alt_model in alternative_models:
                try:
                    print(f"Trying: {alt_model}")
                    model = LlavaForConditionalGeneration.from_pretrained(
                        alt_model,
                        dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
                    processor = AutoProcessor.from_pretrained(alt_model)
                    print(f"Successfully loaded: {alt_model}")
                    break
                except Exception as e2:
                    print(f"Failed to load {alt_model}: {e2}")
                    continue
            else:
                raise Exception("Could not load any LLaVA model")
    
    # For Qwen models
    elif "qwen" in args.model_id.lower():
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_id,
                dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            processor = AutoProcessor.from_pretrained(args.model_id)
        except Exception as e:
            print(f"Error loading Qwen model: {e}")
            raise
    
    # For InstructBLIP models
    elif "instructblip" in args.model_id.lower():
        try:
            from transformers import InstructBlipForConditionalGeneration
            model = InstructBlipForConditionalGeneration.from_pretrained(
                args.model_id,
                dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            processor = AutoProcessor.from_pretrained(args.model_id)
        except Exception as e:
            print(f"Error loading InstructBLIP model: {e}")
            raise
    
    else:
        # Generic fallback
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            processor = AutoProcessor.from_pretrained(args.model_id)
        except Exception as e:
            print(f"Error loading generic model: {e}")
            raise
    
    model.eval()
    
    # Load dataset
    print("Loading ViCrit dataset...")
    questions = list(load_dataset("russwang/ViCrit-Bench", split="train"))
    
    # Limit dataset size for local testing
    if args.max_samples:
        questions = questions[:args.max_samples]
        print(f"Using {len(questions)} samples for evaluation")
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    final_response = []
    batch_size = args.batch_size
    
    print(f"Starting evaluation with batch size: {batch_size}")
    
    for chunk in tqdm(split_list(questions, math.ceil(len(questions) / batch_size))):
        batch_responses = []
        
        for data in chunk:
            try:
                # Load and process image
                img = Image.open(io.BytesIO(data['image']))
                if img.height < 28 or img.width < 28:
                    print('Skipping small image sample')
                    continue
                
                # Prepare prompt
                qs = multichoice_prompt.format(data['changed_caption'])
                
                # Format input based on model type
                if "llava" in args.model_id.lower() or "llava" in str(type(model)).lower():
                    # LLaVA format
                    inputs = processor(
                        text=qs,
                        images=img,
                        return_tensors="pt"
                    ).to(model.device)
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=processor.tokenizer.eos_token_id
                        )
                    
                    response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Extract only the generated part
                    response = response.split("Hallucination phrase:")[-1].strip()
                
                elif "qwen" in args.model_id.lower() or "qwen" in str(type(model)).lower():
                    # Qwen format
                    message = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user", 
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": qs}
                            ]
                        }
                    ]
                    
                    text = processor.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    inputs = processor(text=text, images=img, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=processor.tokenizer.eos_token_id
                        )
                    
                    response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response.split(text)[-1].strip()
                
                elif "instructblip" in args.model_id.lower() or "instructblip" in str(type(model)).lower():
                    # InstructBLIP format
                    inputs = processor(
                        text=qs,
                        images=img,
                        return_tensors="pt"
                    ).to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=processor.tokenizer.eos_token_id
                        )
                    
                    response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response.split(qs)[-1].strip()
                
                else:
                    # Generic format
                    inputs = processor(
                        text=qs,
                        images=img,
                        return_tensors="pt"
                    ).to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=processor.tokenizer.eos_token_id
                        )
                    
                    response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response.split(qs)[-1].strip()
                
                # Store response
                data['response'] = response
                data['image'] = []  # Remove image data to save space
                final_response.append(data)
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        # Save intermediate results
        if len(final_response) % 10 == 0:
            print(f"Processed {len(final_response)} samples...")
    
    # Save final results
    print(f"Saving {len(final_response)} responses to {answers_file}")
    dump_to_jsonl(final_response, answers_file)
    print("Evaluation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local ViCrit evaluation for smaller models")
    parser.add_argument("--model_id", type=str, required=True, 
                       help="Model ID (e.g., llava-hf/llava-1.5-7b-hf, Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--answers-file", type=str, required=True,
                       help="Path to save evaluation results")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for processing (default: 1 for local)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for testing)")
    
    args = parser.parse_args()
    
    # Validate model selection
    if "72b" in args.model_id.lower() or "70b" in args.model_id.lower():
        print("Warning: You selected a 70B+ model which may not fit in local memory!")
        print("Consider using a smaller model like llava-hf/llava-1.5-7b-hf")
    
    eval_model_local(args) 