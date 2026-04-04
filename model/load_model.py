from unsloth import FastLanguageModel
from model.tokenizer_utils import patch_tokenizer


def load_model(config):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_length,
        load_in_4bit=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer = patch_tokenizer(tokenizer)
    return model, tokenizer
