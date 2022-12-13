#!/usr/bin/env bash

#SBATCH --cpus-per-task=4
#SBATCH --error=stderr/imagenet_vod_yolo_event_ops/s0.04_mTrue_r8_L.txt
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=s0.04_mTrue_r8_L
#SBATCH --mem=64GB
#SBATCH --output=summaries/evaluate/imagenet_vod_yolo_event_ops/s0.04_mTrue_r8_L.txt
#SBATCH --partition=research
#SBATCH --time=1-00:00:00

./experiments/evaluate/imagenet_vod_yolo_event_ops.py 0.04 -m -r 8 -s 320 544 -i _L
