from dataclasses import dataclass
import datetime


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "open-thoughts/OpenThoughts-114k"
    max_length: int = 8192
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 2e-4
    epochs: int = 3
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    eval_steps: int = 400
    train_subset_size: int = 100000
    # GRPO-specific
    grpo_lr: float = 1e-6
    num_generations: int = 4
    max_completion_length: int = 512
    # Loss masking strategy: c1 | c2 | c3 | c5 | c6  (c4 is deferred)
    masking_strategy: str = "c1"
    # Set by run.py before training; do not rely on the default value
    output_dir: str = "outputs/sft_c1_local"

    def make_output_dir(self, mode: str) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{self.masking_strategy}" if mode == "sft" else ""
        self.output_dir = f"outputs/{mode}{suffix}_{ts}"
        return self.output_dir

