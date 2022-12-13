#!/usr/bin/env bash

#SBATCH --cpus-per-task=4
#SBATCH --error=stderr/jhmdb_openpose_event/s0.01_mTrue_r8.txt
#SBATCH --gres=gpu:1
#SBATCH --job-name=s0.01_mTrue_r8
#SBATCH --mem=32GB
#SBATCH --output=summaries/evaluate/jhmdb_openpose_event/s0.01_mTrue_r8.txt
#SBATCH --partition=research
#SBATCH --time=1-00:00:00

./experiments/evaluate/jhmdb_openpose_event.py 0.01 -m -r 8
