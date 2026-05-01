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
#SBATCH --output=slurm_debug_%j.out
#SBATCH --error=slurm_debug_%j.err

set -e

cd /nfs_home/users/dhruvk/team-post-training/llm-finetuning

source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate post-training

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "==== Debug Job Start ===="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/debug_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

python -c "
from datasets import load_dataset
ds = load_dataset('open-thoughts/OpenThoughts-114k', split='train')
print('columns:', ds.column_names)
print('len:', len(ds))
print('first example keys:', list(ds[0].keys()))
" | tee "${OUTPUT_DIR}/debug.log"

echo "==== Debug job complete at $(date) ===="

