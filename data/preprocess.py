from datasets import load_dataset

def format_chat(example, tokenizer):
    text = f"""### Problem:
{example['problem']}

### Solution:
{example['solution']}"""
    return {"text": text}

def get_dataset(config, tokenizer):
    ds = load_dataset(config.dataset_name, split="train")

    # filter high difficulty
    ds = ds.filter(lambda x: x.get("gpt_difficulty_parsed", 0) >= 9)

    print(f"Dataset size after filtering: {len(ds)}")

    # safe subset
    subset_size = min(20000, len(ds))
    ds = ds.shuffle(seed=42).select(range(subset_size))

    ds = ds.map(lambda x: format_chat(x, tokenizer))

    split = ds.train_test_split(test_size=0.05)
    return split["train"], split["test"]
