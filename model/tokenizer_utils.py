SPECIAL_TOKENS = {
    "<think>":   "<unused0>",
    "</think>":  "<unused1>",
    "<answer>":  "<unused2>",
    "</answer>": "<unused3>",
}


def patch_tokenizer(tokenizer):
    """Remap empty Gemma token slots to reasoning special tokens."""
    for new_token, slot in SPECIAL_TOKENS.items():
        slot_id = tokenizer.convert_tokens_to_ids(slot)
        tokenizer.added_tokens_encoder[new_token] = slot_id
        tokenizer.added_tokens_decoder[slot_id] = new_token

    tokenizer.think_token      = "<think>"
    tokenizer.end_think_token  = "</think>"
    tokenizer.answer_token     = "<answer>"
    tokenizer.end_answer_token = "</answer>"
    return tokenizer
