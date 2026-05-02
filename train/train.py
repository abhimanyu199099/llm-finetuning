import torch

from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from config.config import Config
from data.preprocess import get_dataset
from model.load_model import load_model
from model.lora import apply_lora
from train.masking import MaskingCollator


class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, _args, _state, _control, **_kwargs):
        torch.cuda.empty_cache()


def train(config=None):
    if config is None:
        config = Config()

    model, tokenizer = load_model(config)
    model = apply_lora(model, config)
    train_ds, val_ds = get_dataset(config, tokenizer)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=config.grad_accum,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.eval_steps,
        bf16=True,
        max_grad_norm=1.0,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        data_collator=MaskingCollator(tokenizer, strategy=config.masking_strategy),
        processing_class=tokenizer,
    )

    trainer.add_callback(ClearCacheCallback())
    trainer.train()

