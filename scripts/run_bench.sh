#!/bin/bash
#SBATCH --job-name=post_training_bench
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_bench_%j.out
#SBATCH --error=slurm_bench_%j.err

# Usage:
#   sbatch ./scripts/run_bench.sh --checkpoint outputs/checkpoint-2000 --preset easy
#   sbatch ./scripts/run_bench.sh --checkpoint outputs/checkpoint-2000 --category math
#   sbatch ./scripts/run_bench.sh --checkpoint outputs/checkpoint-2000 --benchmarks math500 aime24 gpqa
#   sbatch ./scripts/run_bench.sh --preset hard   # base model, no checkpoint

cd /nfs_home/users/dhruvk/team-post-training/llm-finetuning

source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate post-training

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "==== Benchmark Eval Job Start ===="
echo "Args: $@"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/bench_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

python run.py --mode bench "$@" | tee "${OUTPUT_DIR}/bench.log"

echo "==== Benchmark Eval job complete at $(date) ===="

