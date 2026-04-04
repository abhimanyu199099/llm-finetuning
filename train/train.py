import torch
import wandb

from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from config.config import Config
from data.preprocess import get_dataset
from model.load_model import load_model
from model.lora import apply_lora

class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

def train():
    config = Config()

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or None,
        config=vars(config),
    )

    model, tokenizer = load_model(config)
    model = apply_lora(model, config)
    train_ds, val_ds = get_dataset(config, tokenizer)

    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
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
        fp16=False,
        max_grad_norm=1.0,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.add_callback(ClearCacheCallback)
    trainer.train()
