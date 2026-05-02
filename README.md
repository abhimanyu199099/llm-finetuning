# LLM Fine-tuning: Math Reasoning

Fine-tuning **Qwen2.5-7B-Instruct** on [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) using Unsloth, with support for supervised fine-tuning (SFT), GRPO reinforcement learning, custom held-out evaluation, and standard benchmark evaluation via lm-eval.

## Setup

```bash
pip install -r requirements.txt
```

Requires Linux + CUDA. Tested on A100-80GB.

## Usage

**Supervised fine-tuning (SFT):**
```bash
python run.py --mode sft [--masking c1|c2|c3|c5|c6] [--train-subset-size 80000]
```

**GRPO reinforcement learning:**
```bash
python run.py --mode grpo
```

**Held-out evaluation (custom, uses `\boxed{}` extraction):**
```bash
python run.py --mode eval [--checkpoint outputs/checkpoint-2000] [--n-examples 100]
```

**Benchmark evaluation (lm-eval harness):**
```bash
# presets
python run.py --mode bench --preset easy          # math + mmlu
python run.py --mode bench --preset hard          # math + gpqa

# individual benchmarks
python run.py --mode bench --benchmarks math mmlu gpqa

# with a trained checkpoint
python run.py --mode bench --preset easy --checkpoint outputs/checkpoint-2000

# few-shot
python run.py --mode bench --benchmarks math --num-fewshot 4
```

**SLURM:**
```bash
sbatch scripts/run_bench.sh --preset easy --checkpoint outputs/checkpoint-2000
sbatch scripts/run_sft.sh
```

## Project Structure

```
├── config/
│   └── config.py               # All hyperparameters in one dataclass
├── data/
│   ├── preprocess.py           # Dataset loading and tokenization for SFT
│   └── grpo_dataset.py         # Dataset formatting for GRPO
├── model/
│   ├── load_model.py           # Unsloth FastLanguageModel (4-bit)
│   ├── lora.py                 # LoRA adapter application
│   └── tokenizer_utils.py      # Remaps Qwen FIM tokens to <think>/<answer> specials
├── train/
│   ├── train.py                # SFT training loop (SFTTrainer + MaskingCollator)
│   ├── grpo.py                 # GRPO training loop (GRPOTrainer)
│   ├── masking.py              # Loss masking strategies c1–c6
│   └── rewards.py              # GRPO reward functions: correctness + format
├── eval/
│   ├── evaluate.py             # Batched greedy eval on held-out split
│   ├── bench.py                # lm-eval benchmark dispatcher
│   └── metrics.py              # \boxed{} extraction and accuracy
├── scripts/
│   ├── run_bench.sh            # SLURM script for benchmark eval
│   ├── run_sft.sh              # SLURM script for SFT training
│   ├── run_eval_tiny.sh        # SLURM script for quick held-out eval
│   ├── debug.sh                # SLURM script to inspect dataset
│   └── check_gpu_util.sh       # SLURM script to check GPU utilization
└── run.py                      # Entry point: --mode sft | grpo | eval | bench
```

## Configuration

All settings live in `config/config.py`:

| Field | Default | Description |
|---|---|---|
| `model_name` | `Qwen/Qwen2.5-7B-Instruct` | Base model |
| `dataset_name` | `open-thoughts/OpenThoughts-114k` | Training dataset |
| `train_subset_size` | `80000` | Examples used for training |
| `max_length` | `8192` | Max sequence length |
| `batch_size` | `1` | Per-device batch size |
| `grad_accum` | `16` | Gradient accumulation steps |
| `lr` | `1e-5` | SFT learning rate |
| `lora_r` | `32` | LoRA rank |
| `lora_alpha` | `64` | LoRA alpha |
| `lora_dropout` | `0.0` | LoRA dropout |
| `epochs` | `2` | Training epochs |
| `eval_steps` | `200` | Eval/checkpoint frequency |
| `grpo_lr` | `1e-6` | GRPO learning rate |
| `num_generations` | `4` | GRPO group size |
| `max_completion_length` | `512` | Max tokens per GRPO completion |
| `masking_strategy` | `c1` | Loss masking strategy for SFT |

## Loss Masking Strategies

| Strategy | Description |
|---|---|
| `c1` | Full sequence loss |
| `c2` | Answer-only loss (mask think phase) |
| `c3` | Explore-phase loss (mask answer) |
| `c5` | Random half of think blocks, resampled each step |
| `c6` | Random half of think blocks, frozen seed per example |

## Benchmarks

Available via `--mode bench` (backed by lm-eval 0.4.4):

| Key | Task | Metric |
|---|---|---|
| `math` | hendrycks_math (full, ~12k problems) | exact_match |
| `math500` | minerva_math (500-problem subset) | flexible-extract |
| `mmlu` | mmlu | acc |
| `gpqa` | gpqa_diamond_generative_n_shot | acc |

## Results (checkpoint-2000, c2 masking)

| Benchmark | n-shot | Score |
|---|---|---|
| MMLU | 0 | 0.71 |
| Hendrycks Math (full) | 4 | 0.225 |
| MATH-500 (minerva_math) | 4 | 0.312 |
