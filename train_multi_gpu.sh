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

# Redirect temp files to a local dir (scratch is SMB, can't use unix sockets)
# Use a dedicated subdir in /tmp to keep it organized
export TMPDIR="/tmp/harmonic_$$"
mkdir -p "$TMPDIR"

# Ensure directories exist (don't wipe existing checkpoints)
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

# Auto-resume from latest checkpoint if it exists
RESUME_ARG=""
if [ -f "$CHECKPOINT_DIR/latest.pt" ]; then
    echo "Found existing checkpoint, resuming training..."
    RESUME_ARG="--resume $CHECKPOINT_DIR/latest.pt"
fi

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
echo "TMPDIR: $TMPDIR"
echo "Resume: ${RESUME_ARG:-none (fresh start)}"
echo "=========================================="
echo ""

# Run training
# Export CUDA_VISIBLE_DEVICES so conda run inherits it
export CUDA_VISIBLE_DEVICES=$GPUS
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

conda run -n harmonic_env --no-capture-output \
    env CUDA_VISIBLE_DEVICES=$GPUS TMPDIR="$TMPDIR" python \
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
    $RESUME_ARG \
    2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs: $LOG_DIR"
echo "=========================================="
