#!/bin/bash

eval "$(conda shell.bash hook)"
# export CUDA_VISIBLE_DEVICES=1
conda activate pytorch-from-scratch

export ACCELERATE_LOG_LEVEL="INFO"

# label smoothing: 0.1 (resulting model had low recall)
# accelerate launch yolo/train.py \
# accelerate launch --gpu_ids 0, --num_processes 1 yolo/train.py \
# --output-dir /media/bryan/ssd01/expr/yolo_from_scratch/300-epochs \
# --resume-from-checkpoint /media/bryan/ssd01/expr/yolo_from_scratch/300-epochs/checkpoints/checkpoint_83 \
# --start-epoch 84 \
# --train-batch-size 40 --val-batch-size 32 \
# --epochs 300 --lr-warmup-epochs 10 --eval-epochs 10


# DEBUG
# accelerate launch --gpu_ids 1, --num_processes 1 yolo/train.py \
# accelerate launch yolo/train.py \
# --output-dir /media/bryan/ssd01/expr/yolo_from_scratch/debug03-resume \
# --train-batch-size 32 --val-batch-size 32 \
# --epochs 10 --lr-warmup-epochs 5 --limit-train-iters 10  --limit-val-iters 20 \
# --resume-from-checkpoint /media/bryan/ssd01/expr/yolo_from_scratch/debug02/checkpoints/checkpoint_9 \
# --start-epoch 5

# accelerate launch yolo/train.py \
# --output-dir /media/bryan/ssd01/expr/yolo_from_scratch/debug04 \
# --train-batch-size 32 --val-batch-size 32 \
# --epochs 10 --lr-warmup-epochs 5


# no label smoothing, p=0.5 for photometric distortion
# accelerate launch --gpu_ids 0, --num_processes 1 yolo/train.py \
# --output-dir /media/bryan/ssd01/expr/yolo_from_scratch/100-epochs \
# --train-batch-size 40 --val-batch-size 40 \
# --epochs 100 --lr-warmup-epochs 10 --eval-epochs 10

# Update the Yolo loss function normalization such that the loss is normalized by the batch size,
# not the number of positive anchors (and negative anchors in the case of objectness)
accelerate launch --gpu_ids 0, --num_processes 1 yolo/train.py \
--output-dir /media/bryan/ssd01/expr/yolo_from_scratch/100-epochs-updatedloss \
--train-batch-size 40 --val-batch-size 40 \
--epochs 100 --lr-warmup-epochs 10 --eval-epochs 10