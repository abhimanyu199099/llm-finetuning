#!/bin/bash
#SBATCH --job-name=post_training_sft
#SBATCH --partition=gpu-3day
#SBATCH --qos=gpu-3day
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=slurm_sft_%j.out
#SBATCH --error=slurm_sft_%j.err

set -e

cd /nfs_home/users/dhruvk/team-post-training/llm-finetuning

source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate post-training

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "==== Post-Training Job Start ===="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/sft_c2_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

python run.py \
    --mode sft \
    --masking c2 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "==== SFT job complete at $(date) ===="
