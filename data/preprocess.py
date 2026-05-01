import random

from datasets import load_dataset

IGNORE_INDEX = -100
_IM_START_STR = "<|im_start|>"
_IM_END_STR   = "<|im_end|>"


def _build_think_mask(input_ids, think_id, end_think_id):
    """True for tokens strictly inside <think>...</think>."""
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


def _build_prompt_mask(input_ids, tokenizer):
    """True for every system/user token — these are always excluded from loss."""
    im_start_id  = tokenizer.convert_tokens_to_ids(_IM_START_STR)
    im_end_id    = tokenizer.convert_tokens_to_ids(_IM_END_STR)
    assistant_id = tokenizer.encode("assistant", add_special_tokens=False)[0]

    mask = [True] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i] == im_start_id:
            if i + 1 < len(input_ids) and input_ids[i + 1] == assistant_id:
                j = i
                while j < len(input_ids):
                    mask[j] = False
                    if input_ids[j] == im_end_id and j > i:
                        break
                    j += 1
                i = j + 1
                continue
        i += 1
    return mask


def _build_labels(input_ids, think_mask, prompt_mask, strategy, idx):
    labels = list(input_ids)

    # Always mask prompt tokens
    for j, is_prompt in enumerate(prompt_mask):
        if is_prompt:
            labels[j] = IGNORE_INDEX

    think_positions = [j for j, m in enumerate(think_mask) if m]

    if strategy == "c1":
        pass

    elif strategy == "c2":
        for j in think_positions:
            labels[j] = IGNORE_INDEX

    elif strategy == "c3":
        cutoff = int(len(think_positions) * 0.65)
        for j in think_positions[cutoff:]:
            labels[j] = IGNORE_INDEX

    elif strategy == "c5":
        # Think masking is deferred to MaskingCollator so it can resample each
        # step. Only prompt tokens are masked here.
        pass

    elif strategy == "c6":
        rng = random.Random(idx & 0xFFFF)
        if think_positions:
            to_mask = rng.sample(think_positions, k=len(think_positions) // 2)
            for j in to_mask:
                labels[j] = IGNORE_INDEX

    return labels


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

    min(config.train_subset_size, len(ds))
    ds = ds.shuffle(seed=42).select(range(config.train_subset_size))

    think_id     = tokenizer.think_token_id
    end_think_id = tokenizer.end_think_token_id
    max_length   = config.max_length
    strategy     = config.masking_strategy

    def process(example, idx):
        messages = _build_messages(example)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]

        think_mask  = _build_think_mask(input_ids, think_id, end_think_id)
        prompt_mask = _build_prompt_mask(input_ids, tokenizer)
        labels      = _build_labels(input_ids, think_mask, prompt_mask, strategy, idx)

        return {"input_ids": input_ids, "labels": labels}

    ds = ds.map(process, with_indices=True, remove_columns=ds.column_names)

    before = len(ds)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= max_length)
    print(f"Kept {len(ds)}/{before} examples with length <= {max_length}")

    split = ds.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]

