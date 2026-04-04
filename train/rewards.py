import re


def _extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def correctness_reward(completions, solution, **kwargs):
    scores = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        predicted = _extract_answer(content)
        ground_truth = solution.strip() if isinstance(solution, str) else solution[0].strip()
        scores.append(1.0 if predicted == ground_truth else 0.0)
    return scores


def format_reward(completions, **kwargs):
    scores = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        has_think  = "<think>" in content and "</think>" in content
        has_answer = "<answer>" in content and "</answer>" in content
        scores.append(0.2 if has_think and has_answer else 0.0)
    return scores
