import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from eval.evaluate import evaluate
from train.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        evaluate()
