import json
import os
import argparse

def extract_answer(text):
    """Extract answer from model response"""
    if not text:
        return ""
    
    # Clean up the response
    text = text.strip()
    
    # Try to extract from common patterns
    if 'boxed{' in text:
        # Extract from \boxed{} format
        start = text.find('boxed{') + 6
        end = text.find('}', start)
        if end != -1:
            return text[start:end].strip()
    
    # Return the last non-empty line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        return lines[-1]
    
    return text

def extract_changed_segment(before, after):
    """
    Given a 'before' string and an 'after' string,
    returns the list of tokens in 'after' that are not shared
    as a prefix or suffix with 'before'.
    """
    before_tokens = before.split()
    after_tokens = after.split()

    # Find longest common prefix
    prefix = 0
    while prefix < len(before_tokens) and prefix < len(after_tokens) and before_tokens[prefix] == after_tokens[prefix]:
        prefix += 1

    # Find longest common suffix (avoid overlapping the prefix)
    suffix = 0
    while (suffix < (len(before_tokens) - prefix)) and (suffix < (len(after_tokens) - prefix)) \
          and before_tokens[-(suffix + 1)] == after_tokens[-(suffix + 1)]:
        suffix += 1

    # Extract the middle tokens from 'after'
    if suffix == 0:
        changed = after_tokens[prefix:]
    else:
        changed = after_tokens[prefix:-suffix]
    return changed

def relaxed_correctness(prediction, target, org) -> bool:
    """Calculate relaxed correctness score"""
    if not prediction or not target or not org:
        return 0.0
    
    predicted_after = prediction.lower()
    before, ground_truth_after = target.lower(), org.lower()

    gt_diff = extract_changed_segment(before, ground_truth_after)
    if not gt_diff:
        return 0.0
    
    # Check if any part of the predicted answer contains the ground truth difference
    for diff_token in gt_diff:
        if diff_token.lower() in predicted_after:
            return 1.0
    
    return 0.0

def score_results(results_file):
    """Score the evaluation results"""
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return
    
    print(f"📊 Scoring results from: {results_file}")
    
    # Load results
    datas = []
    try:
        with open(results_file, 'r') as f:
            for line in f:
                datas.append(json.loads(line.strip()))
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    print(f"📈 Loaded {len(datas)} samples")
    
    # Score each sample
    total = 0
    correct = 0
    detailed_results = []
    
    for i, data in enumerate(datas):
        try:
            # Extract required fields
            response = data.get('response', '')
            original_noun_phrases = data.get('original_noun_phrases', '')
            changed_noun_phrases = data.get('changed_noun_phrases', '')
            
            # Extract answer from response
            answer = extract_answer(response)
            
            # Calculate reward
            reward = relaxed_correctness(answer, original_noun_phrases, changed_noun_phrases)
            
            # Store results
            data['extracted_answer'] = answer
            data['reward'] = reward
            data['original_noun_phrases'] = original_noun_phrases
            data['changed_noun_phrases'] = changed_noun_phrases
            
            detailed_results.append(data)
            correct += reward
            total += 1
            
            # Print first few examples
            if i < 3:
                print(f"\n--- Sample {i+1} ---")
                print(f"Original: {original_noun_phrases}")
                print(f"Changed:  {changed_noun_phrases}")
                print(f"Response: {response}")
                print(f"Extracted: {answer}")
                print(f"Correct: {bool(reward)}")
                
        except Exception as e:
            print(f"⚠️  Error processing sample {i}: {e}")
            continue
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"\n🎯 Final Results:")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save detailed results
    output_file = results_file.replace('.jsonl', '_scored.jsonl')
    with open(output_file, 'w') as f:
        for result in detailed_results:
            f.write(json.dumps(result) + '\n')
    
    print(f"📁 Detailed results saved to: {output_file}")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Score ViCrit local evaluation results")
    parser.add_argument("--results-file", type=str, 
                       default="./eval_files/local/answers/llava-hf_llava-1.5-7b-hf.jsonl",
                       help="Path to results file to score")
    
    args = parser.parse_args()
    
    # Score the results
    score_results(args.results_file)

if __name__ == "__main__":
    main() 