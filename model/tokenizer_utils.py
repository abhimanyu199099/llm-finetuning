# Remap Qwen2.5's FIM token slots to reasoning special tokens.
#
# Qwen2.5 includes FIM (fill-in-the-middle) tokens in its vocabulary that are
# unused during instruction-following. We repurpose two of them so that <think>
# and </think> resolve to existing token IDs — no embedding resize needed,
# which avoids the training instability that comes with newly initialised rows.
#
# Slot assignments (Qwen2.5 vocab):
#   <|fim_prefix|>  (151659)  →  <think>
#   <|fim_middle|>  (151660)  →  </think>

QWEN_SPECIAL_TOKENS = {
    "<think>":  "<|fim_prefix|>",
    "</think>": "<|fim_middle|>",
}


def patch_tokenizer(tokenizer):
    for new_token, slot in QWEN_SPECIAL_TOKENS.items():
        slot_id = tokenizer.convert_tokens_to_ids(slot)
        tokenizer.added_tokens_encoder[new_token] = slot_id
        tokenizer.added_tokens_decoder[slot_id] = new_token

    tokenizer.think_token_id     = tokenizer.convert_tokens_to_ids("<|fim_prefix|>")
    tokenizer.end_think_token_id = tokenizer.convert_tokens_to_ids("<|fim_middle|>")
    return tokenizer
