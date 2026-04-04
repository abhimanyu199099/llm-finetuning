# Project Notes

## Goal

Train Gemma 3 12B to reason through olympiad-level math problems, producing structured outputs with explicit chain-of-thought inside `<think>` tags and a final answer inside `<answer>` tags.

Training proceeds in two stages:
1. **SFT** — teach the model the output format and domain knowledge by supervised imitation on problem/solution pairs
2. **GRPO** — refine reasoning quality via reinforcement learning, rewarding correct and well-formatted answers

## Architecture Decisions

### Tokenizer patching instead of adding new tokens
Gemma 3 has 256 reserved `<unusedN>` slots in its vocabulary. The four reasoning tokens (`<think>`, `</think>`, `<answer>`, `</answer>`) are remapped to `<unused0-3>`. This avoids resizing the embedding matrix, which would require re-initializing weights and training from a worse starting point.

See: `model/tokenizer_utils.py`

### Unsloth for both SFT and GRPO
SFT uses `FastLanguageModel` + `SFTTrainer`. GRPO uses `PatchFastRL("GRPO", FastLanguageModel)` before importing `GRPOTrainer`, which applies Unsloth's kernel optimizations to the RL training loop. The model loading and LoRA application code is shared between both modes.

### GRPO reward design
Two reward functions are combined:

- **`correctness_reward`** (weight 1.0): Extracts text from `<answer>...</answer>` and does an exact string match against the ground truth solution. Binary 1.0 / 0.0.
- **`format_reward`** (weight 0.2): Awards a small bonus for structurally correct output — both `<think>` and `<answer>` tags present. Encourages the model to maintain format even when the answer is wrong.

The format reward is intentionally small so it doesn't dominate over correctness.

### LoRA target modules
Only `q_proj` and `v_proj` are targeted. This is conservative — adding `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` would increase trainable parameters and potentially improve quality at the cost of memory.

### beta=0.0 in GRPO
No reference model is loaded (`beta=0.0` disables KL divergence penalty). This saves significant VRAM. If the model starts drifting too far from coherent outputs during GRPO, increase `beta` to 0.01–0.1.

## Data Pipeline

```
Raw dataset (162k problems)
  → filter difficulty >= 9
  → shuffle with seed=42
  → take up to 20,000
  → format (SFT: text template | GRPO: message list + solution column)
  → 95/5 train/eval split
```

The same filtering and subsetting logic is duplicated between `data/preprocess.py` and `data/grpo_dataset.py`. If the dataset strategy changes, update both.

## Evaluation

Evaluation (`run.py --mode eval`) runs greedy decoding on 100 val examples and computes `\boxed{}` extraction accuracy. This matches the format used in math competition datasets. The eval uses the SFT val split, not the GRPO dataset format.

## Multi-GPU

Both SFT and GRPO run under DDP via `torchrun`. No code changes needed — `Trainer`/`GRPOTrainer` handle device placement automatically. Launch with:
```bash
torchrun --nproc_per_node=NUM_GPUS run.py --mode sft|grpo
```

GRPO does not support FSDP (Unsloth limitation as of early 2026). Use DDP only.

## Outputs

| Mode | Output dir |
|---|---|
| SFT | `./outputs/` |
| GRPO | `./outputs_grpo/` |

Checkpoints are saved every `eval_steps` (default 200) steps.
