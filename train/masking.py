import random
import torch

IGNORE_INDEX = -100


def _think_positions(input_ids: list[int], think_id: int, end_think_id: int) -> list[int]:
    """Return indices of tokens strictly inside <think>...</think>."""
    positions = []
    inside = False
    for i, tok in enumerate(input_ids):
        if tok == think_id:
            inside = True
        elif tok == end_think_id:
            inside = False
        elif inside:
            positions.append(i)
    return positions


class MaskingCollator:
    """
    Pads a batch of pre-tokenized examples and, for C5, applies a fresh random
    think-block mask each call (resampled every step/epoch).

    For all other strategies (c1–c3, c6) masking is already baked into the
    `labels` column by data/preprocess.py. This collator only needs to pad
    those and handle C5's live resampling.

    c4 (segment attribution) is deferred — raises NotImplementedError.
    """

    def __init__(self, tokenizer, strategy: str):
        if strategy == "c4":
            raise NotImplementedError(
                "C4 (segment attribution) is not yet implemented. "
                "Run data/segment_attribution.py first, then retry."
            )
        self.pad_id       = tokenizer.pad_token_id
        self.strategy     = strategy
        self.think_id     = tokenizer.think_token_id
        self.end_think_id = tokenizer.end_think_token_id

    def __call__(self, batch: list[dict]) -> dict:
        input_ids_list = [ex["input_ids"] for ex in batch]
        labels_list    = [list(ex["labels"]) for ex in batch]  # mutable copy

        # C5: re-derive think positions from input_ids and resample each call
        if self.strategy == "c5":
            for input_ids, labels in zip(input_ids_list, labels_list):
                positions = _think_positions(input_ids, self.think_id, self.end_think_id)
                if positions:
                    to_mask = random.sample(positions, k=len(positions) // 2)
                    for j in to_mask:
                        labels[j] = IGNORE_INDEX

        max_len = max(len(ids) for ids in input_ids_list)

        padded_input_ids      = []
        padded_attention_mask = []
        padded_labels         = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(     [self.pad_id]  * pad_len + input_ids)
            padded_attention_mask.append([0]            * pad_len + [1] * len(input_ids))
            padded_labels.append(        [IGNORE_INDEX] * pad_len + labels)

        return {
            "input_ids":      torch.tensor(padded_input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels":         torch.tensor(padded_labels,         dtype=torch.long),
        }

