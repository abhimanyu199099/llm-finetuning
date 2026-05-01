#!/bin/bash
#SBATCH --job-name=post_training_sft
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_eval_%j.out
#SBATCH --error=slurm_eval_%j.err

set -e

cd /nfs_home/users/dhruvk/team-post-training/llm-finetuning

source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate post-training

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "==== Post-Training Eval Job Start ===="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/eval_c2_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

python run.py --mode eval --checkpoint outputs/checkpoint-2000 | tee "${OUTPUT_DIR}/eval_tiny.log"

echo "==== Eval job complete at $(date) ===="

