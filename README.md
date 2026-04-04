# LLM Fine-tuning: Olympiad Math Reasoning

Fine-tuning **Gemma 3 12B** on olympiad-level math problems using Unsloth, with support for both supervised fine-tuning (SFT) and reinforcement learning via GRPO.

## Dataset

[felixZzz/numina_162k_olympiads_problems](https://huggingface.co/datasets/felixZzz/numina_162k_olympiads_problems) — filtered to high-difficulty problems (`gpt_difficulty_parsed >= 9`), capped at 20,000 examples, 95/5 train/eval split.

## Setup

```bash
pip install unsloth trl transformers datasets peft
```

> Requires Linux + CUDA. Multi-GPU training uses DDP via `torchrun`.

## Usage

**Supervised fine-tuning (SFT):**
```bash
python run.py --mode sft
# multi-GPU:
torchrun --nproc_per_node=NUM_GPUS run.py --mode sft
```

**GRPO reinforcement learning:**
```bash
python run.py --mode grpo
# multi-GPU:
torchrun --nproc_per_node=NUM_GPUS run.py --mode grpo
```

**Evaluation:**
```bash
python run.py --mode eval
```

## Project Structure

```
├── config/
│   └── config.py           # All hyperparameters in one dataclass
├── data/
│   ├── preprocess.py       # Dataset loading and formatting for SFT
│   └── grpo_dataset.py     # Dataset formatting for GRPO (prompt + solution columns)
├── model/
│   ├── load_model.py       # FastLanguageModel loading (4-bit, Unsloth)
│   ├── lora.py             # LoRA adapter application
│   └── tokenizer_utils.py  # Patches <unused> slots with reasoning special tokens
├── train/
│   ├── train.py            # SFT training loop (SFTTrainer)
│   ├── grpo.py             # GRPO training loop (GRPOTrainer + PatchFastRL)
│   └── rewards.py          # Reward functions: correctness + format
├── eval/
│   ├── evaluate.py         # Greedy decoding evaluation on val set
│   └── metrics.py          # \boxed{} extraction and accuracy computation
└── run.py                  # Entry point: --mode sft | grpo | eval
```

## Key Design Decisions

- **Special tokens** — `<think>`, `</think>`, `<answer>`, `</answer>` are mapped to Gemma 3's pre-existing `<unused0-3>` token slots, so no embedding resize is needed.
- **GRPO rewards** — two functions: `correctness_reward` (1.0/0.0 exact match on `<answer>` content) and `format_reward` (0.2 bonus for using both `<think>` and `<answer>` tags).
- **4-bit quantization** — handled by Unsloth's `FastLanguageModel`, no manual `BitsAndBytesConfig` needed.
- **LoRA** — applied to `q_proj` and `v_proj` with `r=4`, `alpha=16`.

## Configuration

All settings live in `config/config.py`:

| Field | Default | Description |
|---|---|---|
| `model_name` | `google/gemma-3-12b` | Base model |
| `max_length` | `1024` | Max sequence length |
| `batch_size` | `1` | Per-device batch size |
| `grad_accum` | `4` | Gradient accumulation steps |
| `lr` | `1e-5` | SFT learning rate |
| `grpo_lr` | `1e-6` | GRPO learning rate |
| `num_generations` | `4` | GRPO group size |
| `max_completion_length` | `512` | Max tokens generated per GRPO step |
| `lora_r` | `4` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha |
| `epochs` | `2` | Training epochs |
| `eval_steps` | `200` | Eval/checkpoint frequency |
