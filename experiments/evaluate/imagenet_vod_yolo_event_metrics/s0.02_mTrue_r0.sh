#!/usr/bin/env bash

#SBATCH --cpus-per-task=4
#SBATCH --error=stderr/imagenet_vod_yolo_event_metrics/s0.02_mTrue_r0.txt
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=s0.02_mTrue_r0
#SBATCH --mem=32GB
#SBATCH --output=summaries/evaluate/imagenet_vod_yolo_event_metrics/s0.02_mTrue_r0.txt
#SBATCH --partition=research
#SBATCH --time=1-00:00:00

./experiments/evaluate/imagenet_vod_yolo_event_metrics.py 0.02 -m
