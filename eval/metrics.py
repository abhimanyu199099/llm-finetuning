import re


def extract_boxed(text):
    """Extract content of the last \\boxed{} in text, handling nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    content = text[start:i - 1].strip()
    return content if content else None


def _normalize(s):
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    return s.lower()


def compute_accuracy(preds, gts):
    correct = 0
    total = len(preds)
    for p, g in zip(preds, gts):
        if p is not None and g is not None and _normalize(p) == _normalize(g):
            correct += 1
    return correct / total if total > 0 else 0

