from datasets import load_dataset


def _build_think_mask(input_ids, think_id, end_think_id):
    """Return a bool list: True for tokens strictly inside <think>...</think>."""
    mask = [False] * len(input_ids)
    inside = False
    for i, tok in enumerate(input_ids):
        if tok == think_id:
            inside = True
        elif tok == end_think_id:
            inside = False
        elif inside:
            mask[i] = True
    return mask


def _build_messages(example):
    system = example["system"]
    messages = [{"role": "system", "content": system}]
    for turn in example["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    return messages


def get_dataset(config, tokenizer):
    ds = load_dataset(config.dataset_name, split="train")

    print(f"Dataset size: {len(ds)}")

    subset_size = min(20000, len(ds))
    ds = ds.shuffle(seed=42).select(range(subset_size))

    think_id     = tokenizer.think_token_id
    end_think_id = tokenizer.end_think_token_id
    max_length   = config.max_length

    def process(example, idx):
        messages = _build_messages(example)

        # Apply Qwen's ChatML template and tokenize
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = tokenizer(text, truncation=False, add_special_tokens=False)
        input_ids      = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        think_token_mask = _build_think_mask(input_ids, think_id, end_think_id)

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "think_token_mask": think_token_mask,
            "c6_seed":         idx & 0xFFFF,
        }

    ds = ds.map(
        process,
        with_indices=True,
        remove_columns=ds.column_names,
    )

    # Drop examples that exceed the context window
    before = len(ds)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= max_length)
    print(f"Kept {len(ds)}/{before} examples with length <= {max_length}")

    split = ds.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]
