import wandb

from trl import GRPOTrainer, GRPOConfig
from config.config import Config
from data.grpo_dataset import get_grpo_dataset
from model.load_model import load_model
from model.lora import apply_lora
from train.rewards import correctness_reward, format_reward


def train_grpo():
    config = Config()

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or None,
        config=vars(config),
    )

    model, tokenizer = load_model(config)
    model = apply_lora(model, config)
    train_ds, val_ds = get_grpo_dataset(config, tokenizer)

    grpo_config = GRPOConfig(
        output_dir="./outputs_grpo",
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.grpo_lr,
        num_train_epochs=config.epochs,
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.eval_steps,
        report_to="wandb",
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=1.0,
        beta=0.0,
        loss_type="grpo",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, format_reward],
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    trainer.train()
