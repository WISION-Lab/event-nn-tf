#!/usr/bin/env bash

#SBATCH --cpus-per-task=4
#SBATCH --error=stderr/jhmdb_openpose_event/s0.02_mTrue_r0_L.txt
#SBATCH --gres=gpu:1
#SBATCH --job-name=s0.02_mTrue_r0_L
#SBATCH --mem=32GB
#SBATCH --output=summaries/evaluate/jhmdb_openpose_event/s0.02_mTrue_r0_L.txt
#SBATCH --partition=research
#SBATCH --time=1-00:00:00

./experiments/evaluate/jhmdb_openpose_event.py 0.02 -m -s 352 480 -i _L
