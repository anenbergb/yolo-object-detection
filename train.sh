#!/bin/bash

eval "$(conda shell.bash hook)"
# export CUDA_VISIBLE_DEVICES=1
conda activate pytorch-from-scratch

# ACCELERATE_DEBUG_MODE="1"
accelerate launch --gpu_ids 1, --num_processes 1 yolo/train.py \
--output-dir /media/bryan/ssd01/expr/yolo_from_scratch/debug02 \
--train-batch-size 32 --val-batch-size 32 \
--epochs 10 --lr-warmup-epochs 5