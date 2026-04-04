from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are an expert at solving olympiad mathematics problems. "
    "Think step by step inside <think></think> tags, "
    "then give your final answer inside <answer></answer> tags."
)


def format_for_grpo(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": example["problem"]},
        ],
        "solution": example["solution"],
    }


def get_grpo_dataset(config, tokenizer):
    ds = load_dataset(config.dataset_name, split="train")
    ds = ds.filter(lambda x: x.get("gpt_difficulty_parsed", 0) >= 9)

    subset_size = min(20000, len(ds))
    ds = ds.shuffle(seed=42).select(range(subset_size))

    ds = ds.map(format_for_grpo, remove_columns=ds.column_names)

    split = ds.train_test_split(test_size=0.05)
    return split["train"], split["test"]
