#!/bin/bash
#SBATCH --job-name=check-util
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_check_gpu_%j.out
#SBATCH --error=slurm_check_gpu_%j.err

set -e

cd /nfs_home/users/dhruvk/team-post-training/llm-finetuning


echo "==== Running nvidia-smi ===="
nvidia-smi

