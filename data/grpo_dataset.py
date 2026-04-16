import re

from datasets import load_dataset


def _extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def format_for_grpo(example):
    conversations = example["conversations"]

    # First human turn is the problem prompt
    human_turn = next(t for t in conversations if t["from"] == "human")
    # Last gpt turn contains the reference answer
    gpt_turns = [t for t in conversations if t["from"] == "gpt"]
    reference = _extract_answer(gpt_turns[-1]["value"]) if gpt_turns else ""

    return {
        "prompt": [
            {"role": "system", "content": example["system"]},
            {"role": "user",   "content": human_turn["value"]},
        ],
        "solution": reference,
    }


def get_grpo_dataset(config, tokenizer=None):
    ds = load_dataset(config.dataset_name, split="train")

    print(f"Dataset size: {len(ds)}")

    subset_size = min(20000, len(ds))
    ds = ds.shuffle(seed=42).select(range(subset_size))

    ds = ds.map(format_for_grpo, remove_columns=ds.column_names)

    # drop examples where we couldn't extract a reference answer
    ds = ds.filter(lambda x: x["solution"] != "")

    split = ds.train_test_split(test_size=0.05)
    return split["train"], split["test"]
