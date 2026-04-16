import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from eval.evaluate import evaluate
from train.train import train

MASKING_CHOICES = ["c1", "c2", "c3", "c5", "c6"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sft", "eval", "grpo"], required=True)
    parser.add_argument(
        "--masking",
        choices=MASKING_CHOICES,
        default="c1",
        help=(
            "Loss masking strategy for SFT: "
            "c1=full, c2=answer-only, c3=explore-phase, "
            "c5=random-half-resampled, c6=random-half-frozen"
        ),
    )
    args = parser.parse_args()

    if args.mode == "sft":
        from config.config import Config
        config = Config()
        config.masking_strategy = args.masking
        train(config)
    elif args.mode == "grpo":
        from train.grpo import train_grpo
        train_grpo()
    else:
        evaluate()
