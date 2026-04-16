import random
import torch

IGNORE_INDEX = -100

# im_start / im_end token ids for Qwen2.5's ChatML format
_IM_START_STR = "<|im_start|>"
_IM_END_STR   = "<|im_end|>"


def _prompt_mask(input_ids: list[int], tokenizer) -> list[bool]:
    """
    Return True for every token that belongs to a prompt (system or user) turn.
    Assistant tokens are False — they receive loss.
    Works by scanning for <|im_start|>assistant ... <|im_end|> boundaries.
    """
    im_start_id = tokenizer.convert_tokens_to_ids(_IM_START_STR)
    im_end_id   = tokenizer.convert_tokens_to_ids(_IM_END_STR)

    # Encode role strings as bare token ids (without special tokens)
    assistant_id = tokenizer.encode("assistant", add_special_tokens=False)[0]

    mask = [True] * len(input_ids)   # default: mask everything (prompt)
    i = 0
    while i < len(input_ids):
        if input_ids[i] == im_start_id:
            # Check if the next token is the 'assistant' role token
            if i + 1 < len(input_ids) and input_ids[i + 1] == assistant_id:
                # Unmask tokens in this assistant turn
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


class MaskingCollator:
    """
    Pads a batch of pre-tokenized examples and builds labels according to the
    selected masking strategy.

    Strategies
    ----------
    c1  Full SFT (baseline)   — loss on all assistant tokens
    c2  Answer-only           — entire <think> block masked
    c3  Explore-phase         — last 35% of think tokens masked
    c4  Segment attribution   — deferred (raises NotImplementedError)
    c5  Random-half resampled — random 50% of think tokens masked, new each step
    c6  Random-half frozen    — same 50% every epoch, seeded per example
    """

    def __init__(self, tokenizer, strategy: str):
        if strategy == "c4":
            raise NotImplementedError(
                "C4 (segment attribution) is not yet implemented. "
                "Run data/segment_attribution.py first, then retry."
            )
        self.tokenizer = tokenizer
        self.strategy  = strategy

    def __call__(self, batch: list[dict]) -> dict:
        input_ids_list      = [ex["input_ids"]       for ex in batch]
        attention_mask_list = [ex["attention_mask"]   for ex in batch]
        think_mask_list     = [ex["think_token_mask"] for ex in batch]
        c6_seeds            = [ex.get("c6_seed", 0)  for ex in batch]

        max_len = max(len(ids) for ids in input_ids_list)
        pad_id  = self.tokenizer.pad_token_id

        padded_input_ids      = []
        padded_attention_mask = []
        padded_labels         = []

        for i, (input_ids, attn_mask, think_mask) in enumerate(
            zip(input_ids_list, attention_mask_list, think_mask_list)
        ):
            seq_len  = len(input_ids)
            pad_len  = max_len - seq_len

            # Build labels: start from input_ids, then apply masks
            labels = list(input_ids)

            # Always mask prompt tokens (system + user turns)
            prompt_mask = _prompt_mask(input_ids, self.tokenizer)
            for j, is_prompt in enumerate(prompt_mask):
                if is_prompt:
                    labels[j] = IGNORE_INDEX

            # Apply strategy-specific think-block masking
            think_positions = [j for j, m in enumerate(think_mask) if m]

            if self.strategy == "c1":
                pass  # no additional masking

            elif self.strategy == "c2":
                for j in think_positions:
                    labels[j] = IGNORE_INDEX

            elif self.strategy == "c3":
                # Mask the last 35% of think tokens (convergent tail)
                cutoff = int(len(think_positions) * 0.65)
                for j in think_positions[cutoff:]:
                    labels[j] = IGNORE_INDEX

            elif self.strategy == "c5":
                # Random 50%, resampled each call
                rng = random.Random()
                to_mask = rng.sample(think_positions, k=len(think_positions) // 2)
                for j in to_mask:
                    labels[j] = IGNORE_INDEX

            elif self.strategy == "c6":
                # Random 50%, frozen per example via stored seed
                rng = random.Random(c6_seeds[i])
                to_mask = rng.sample(think_positions, k=len(think_positions) // 2)
                for j in to_mask:
                    labels[j] = IGNORE_INDEX

            # Left-pad inputs; pad labels with IGNORE_INDEX
            padded_input_ids.append(      [pad_id] * pad_len + input_ids)
            padded_attention_mask.append( [0]      * pad_len + attn_mask)
            padded_labels.append(         [IGNORE_INDEX] * pad_len + labels)

        return {
            "input_ids":      torch.tensor(padded_input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels":         torch.tensor(padded_labels,         dtype=torch.long),
        }
