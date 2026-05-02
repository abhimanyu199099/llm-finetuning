# Project Notes

## Goal

Fine-tune Qwen2.5-7B-Instruct to reason through math problems, producing structured outputs with chain-of-thought inside `<think>` tags and a final answer inside `<answer>` tags. Training proceeds in two stages:

1. **SFT** — supervised imitation on OpenThoughts-114k problem/solution pairs
2. **GRPO** — reinforcement learning to refine reasoning quality, rewarding correct and well-formatted answers

Target: ≥0.90 accuracy on held-out `\boxed{}` extraction eval, and near-SOTA performance on popular math, science, code and QA benchmarks.

## Architecture Decisions

### Tokenizer patching instead of adding new tokens
Qwen2.5 has unused FIM (fill-in-the-middle) tokens in its vocabulary. The four reasoning tokens (`<think>`, `</think>`, `<answer>`, `</answer>`) are remapped to these slots. This avoids resizing the embedding matrix, which would require re-initializing weights.

See: `model/tokenizer_utils.py`

### Unsloth FastLanguageModel with 4-bit quantization
Model is loaded via `unsloth.FastLanguageModel` in 4-bit NF4. This halves VRAM usage compared to bf16. The 4-bit model cannot be cast with `.to(dtype)` or merged with `merge_and_unload()` — inference must run with the adapter attached.

### LoRA configuration
Applied to all linear layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Current config: r=32, alpha=64, dropout=0.0.

### GRPO reward design
Two reward functions:
- **`correctness_reward`**: Extracts text from `<answer>...</answer>` and exact-matches against ground truth. Binary 1.0 / 0.0.
- **`format_reward`**: Small bonus (0.2) for structurally correct output — both `<think>` and `<answer>` tags present. Kept small so it doesn't dominate over correctness.

## Data Pipeline

### SFT
```
OpenThoughts-114k (raw)
  → shuffle seed=42
  → take first train_subset_size (default 80,000)
  → apply chat template + tokenize
  → build masked labels (strategy c1–c6)
  → held-out eval = examples after train_subset_size index
```

The held-out split is deterministic: same shuffle seed, different index range. This prevents leakage from training data into eval.

### GRPO
```
OpenThoughts-114k (raw)
  → extract problem prompt + reference answer
  → format as message list for GRPOTrainer
```

## Loss Masking Strategies

The `MaskingCollator` in `train/masking.py` applies one of five strategies:

- **c1**: Full sequence loss — train on everything including think tokens
- **c2**: Answer-only — mask the think phase, only backprop through final answer
- **c3**: Explore-phase — mask the answer, only backprop through think phase
- **c5**: Random half of think blocks resampled at each training step
- **c6**: Random half of think blocks with a frozen per-example seed (consistent across epochs)

c4 is deferred. c2 is the current primary experiment (answer-only loss, cleaner signal).

## Evaluation

### Held-out eval (`--mode eval`)
Runs batched greedy decoding (batch=8, left-padded) on held-out examples. Extracts `\boxed{}` content using a brace-counting parser with `rfind` to find the last `\boxed{}`. Normalizes whitespace and LaTeX spacing tokens before comparison.

### Benchmark eval (`--mode bench`)
Uses lm-eval 0.4.4 (`python -m lm_eval`). Available tasks confirmed installed on the cluster:

| Key | lm-eval task name |
|---|---|
| `math` | `hendrycks_math` |
| `mmlu` | `mmlu` |
| `gpqa` | `gpqa_diamond_generative_n_shot` |

Note: `humaneval`, `aime`, and `livecodebench` are not bundled in lm-eval 0.4.4 and are not available on this install. `minerva_math` is available as an alternative to `hendrycks_math` with more lenient scoring.

The `--num-fewshot` flag (default 0) controls few-shot examples. For math, `--num-fewshot 4` significantly improves exact_match scores by showing the model expected output format.

## Cluster Setup

- Hardware: A100-80GB (single GPU per job)
- Conda env: `post-training`
- Project path: `/nfs_home/users/dhruvk/team-post-training/llm-finetuning`
- HF token: loaded from `.env` in project root (`HF_TOKEN=hf_...`)
- SLURM scripts in `scripts/`

Known issues:
- Flash Attention 2 is broken (conflicting flash_attn wheel). Unsloth falls back to xformers automatically. No action needed.

## Outputs

| Mode | Output dir |
|---|---|
| SFT checkpoints | `./outputs/checkpoint-N/` |
| GRPO checkpoints | `./outputs_grpo/` |
| Benchmark results | `./outputs/bench_TIMESTAMP/` |

Checkpoints save every `eval_steps` (default 200) steps.

## Results (checkpoint-2000, c2 masking)

| Benchmark | n-shot | Score |
|---|---|---|
| MMLU | 0 | 0.71 |
| Hendrycks Math | 0 | 0.023 |
| Hendrycks Math | 4 | 0.225 |

Subtask breakdown (4-shot):

| Subtask | Score |
|---|---|
| prealgebra | 0.357 |
| algebra | 0.234 |
| counting_and_prob | 0.222 |
| num_theory | 0.202 |
| geometry | 0.196 |
| precalc | 0.196 |
| intermediate_algebra | 0.135 |

0-shot exact_match is dominated by formatting mismatch; 4-shot is the standard comparison point.
