import json

def extract_boxed_content(s):
    keyword = r'\boxed{'
    start = s.find(keyword)
    if start == -1:
        return None
    start_brace = s.find('{', start)
    if start_brace == -1:
        return None
    count = 1
    i = start_brace + 1
    while i < len(s) and count > 0:
        if s[i] == '{':
            count += 1
        elif s[i] == '}':
            count -= 1
        i += 1
    return s[start_brace+1:i-1]

def extract_answer(text):
    response = text
    if 'boxed{' in response:
        final_answer = extract_boxed_content(response)
        # .split('boxed{')[-1]
        return final_answer
    else:
        return response.split('\n')[-1]

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
    predicted_after = prediction.lower()
    before, ground_truth_after = target.lower(), org.lower()

    gt_diff = extract_changed_segment(before, ground_truth_after)
    reward = 1.0 if gt_diff[0] in prediction.lower() else 0.0
    return reward > 0.0

datas = []
with open("Your_eval_results_file.jsonl", 'r') as files:
    for line in files:
        datas.append(json.loads(line))

total = 0
correct = 0
res = []
for data in datas:
    reward = relaxed_correctness(extract_answer(data['response']), data['original_noun_phrases'], data['changed_noun_phrases'])
    data['reward'] = reward
    correct += reward
    total += 1
print('ACC:', correct/total)