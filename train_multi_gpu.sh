#!/bin/bash
# HARMONIC Multi-GPU Training Script

# Configuration
GPUS="2,3"
NUM_GPUS=2
EPOCHS=200
BATCH_SIZE=32  # Per GPU, so total = 32 * 2 = 64
LEARNING_RATE=1e-4
DATA_DIR="/media/scratch/adele/harmonic/dataset"  # COCO dataset location
CHECKPOINT_DIR="/media/scratch/adele/harmonic/checkpoints"
LOG_DIR="/media/scratch/adele/harmonic/logs"

# Ensure directories exist (don't wipe existing checkpoints)
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

echo "=========================================="
echo "HARMONIC Multi-GPU Training"
echo "=========================================="
echo "GPUs: $GPUS"
echo "Epochs: $EPOCHS"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Total Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "Learning Rate: $LEARNING_RATE"
echo "Data: $DATA_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs: $LOG_DIR"
echo "=========================================="
echo ""

# Run training
CUDA_VISIBLE_DEVICES=$GPUS \
conda run -n harmonic_env --no-capture-output python \
    train_harmonic.py \
    --train_data "$DATA_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_workers 4 \
    --schedule_type conflict_aware \
    --save_every 20 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs: $LOG_DIR"
echo "=========================================="
