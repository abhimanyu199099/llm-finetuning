from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "open-thoughts/OpenThoughts-114k"
    max_length: int = 4096
    batch_size: int = 2
    grad_accum: int = 16
    lr: float = 2e-5
    epochs: int = 4
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    eval_steps: int = 400
    train_subset_size: int = 80000
    # GRPO-specific
    grpo_lr: float = 1e-6
    num_generations: int = 4
    max_completion_length: int = 512
    # Loss masking strategy: c1 | c2 | c3 | c5 | c6  (c4 is deferred)
    masking_strategy: str = "c1"

