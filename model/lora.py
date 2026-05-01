from unsloth import FastLanguageModel


def apply_lora(model, config):
    return FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing=True,
    )
