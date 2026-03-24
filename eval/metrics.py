import re


def extract_boxed(text):
    match = re.search(r"\\boxed\{(.*?)\}", text)
    return match.group(1) if match else None


def compute_accuracy(preds, gts):
    correct = 0
    total = len(preds)
    for p, g in zip(preds, gts):
        if p is not None and g is not None and p == g:
            correct += 1
    return correct / total if total > 0 else 0
