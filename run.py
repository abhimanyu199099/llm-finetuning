import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from eval.evaluate import evaluate
from train.train import train

MASKING_CHOICES = ["c1", "c2", "c3", "c5", "c6"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sft", "eval", "grpo", "bench"], required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to LoRA checkpoint directory for eval (e.g. outputs/checkpoint-400)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=100,
        help="Number of held-out examples to evaluate on",
    )
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
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=80000,
        help="Number of training examples to use from the dataset",
    )
    # bench mode args
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["math", "math500", "mmlu", "gpqa"],
        help="Specific benchmarks to run",
    )
    parser.add_argument(
        "--category",
        choices=["math", "coding", "qa"],
        help="Run all benchmarks in a category",
    )
    parser.add_argument(
        "--preset",
        choices=["easy", "hard"],
        help="easy: math+mmlu  hard: math+gpqa",
    )
    parser.add_argument(
        "--bench-batch-size",
        type=int,
        default=4,
        help="Batch size for benchmark generation",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples for benchmarks",
    )
    args = parser.parse_args()

    if args.mode == "sft":
        from config.config import Config
        config = Config()
        config.masking_strategy = args.masking
        config.train_subset_size = args.train_subset_size
        train(config)
    elif args.mode == "grpo":
        from train.grpo import train_grpo
        train_grpo()
    elif args.mode == "bench":
        from eval.bench import run_bench
        from config.config import Config
        run_bench(
            checkpoint_path=args.checkpoint,
            benchmarks=args.benchmarks,
            category=args.category,
            preset=args.preset,
            batch_size=args.bench_batch_size,
            num_fewshot=args.num_fewshot,
            model_name=Config().model_name,
        )
    else:
        evaluate(checkpoint_path=args.checkpoint, n_examples=args.n_examples,
                 train_subset_size=args.train_subset_size)

