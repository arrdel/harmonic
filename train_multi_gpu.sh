#!/bin/bash
# HARMONIC Multi-GPU Training Script

# Configuration
GPUS="0,1"
NUM_GPUS=2
EPOCHS=200
BATCH_SIZE=32  # Per GPU, so total = 32 * 2 = 64
LEARNING_RATE=1e-4
DATA_DIR="/media/scratch/adele/harmonic/dataset"  # COCO dataset location

# Ensure checkpoints are clean
rm -rf checkpoints/*
mkdir -p checkpoints logs

echo "=========================================="
echo "HARMONIC Multi-GPU Training"
echo "=========================================="
echo "GPUs: $GPUS"
echo "Epochs: $EPOCHS"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Total Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "Learning Rate: $LEARNING_RATE"
echo "Data: $DATA_DIR"
echo "=========================================="
echo ""

# Run training with DDP
CUDA_VISIBLE_DEVICES=$GPUS \
conda run -n harmonic_env --no-capture-output python \
    train_harmonic.py \
    --train_data $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_workers 4 \
    --schedule_type conflict_aware \
    --save_every 20 \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Checkpoints: checkpoints/"
echo "Logs: logs/"
echo "=========================================="
