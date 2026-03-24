from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "google/gemma-2b"
    dataset_name: str = "felixZzz/numina_162k_olympiads_problems"
    max_length: int = 1024
    batch_size: int = 1
    grad_accum: int = 4
    lr: float = 1e-5
    epochs: int = 2
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    eval_steps: int = 200

