#!/bin/bash

eval "$(conda shell.bash hook)"
# export CUDA_VISIBLE_DEVICES=1
conda activate pytorch-from-scratch

export ACCELERATE_LOG_LEVEL="INFO"
accelerate launch yolo/train.py \
--output-dir /media/bryan/ssd01/expr/yolo_from_scratch/300-epochs \
--train-batch-size 32 --val-batch-size 32 \
--epochs 300 --lr-warmup-epochs 10 --eval-epochs 10

# DEBUG
# accelerate launch yolo/train.py \
# accelerate launch --gpu_ids 1, --num_processes 1 yolo/train.py \
# --output-dir /media/bryan/ssd01/expr/yolo_from_scratch/debug03-resume \
# --train-batch-size 32 --val-batch-size 32 \
# --epochs 10 --lr-warmup-epochs 5 --limit-train-iters 10  --limit-val-iters 20 \
# --resume-from-checkpoint /media/bryan/ssd01/expr/yolo_from_scratch/debug02/checkpoints/checkpoint_9 \
# --start-epoch 5