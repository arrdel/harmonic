#!/bin/bash
# Complete HARMONIC Training Pipeline

set -e

HARMONIC_DIR="/home/adelechinda/home/semester_projects/spring_26/computer_vision/HARMONIC"
DATASET_DIR="/media/scratch/adele/harmonic/dataset"
TRAIN_DIR="$DATASET_DIR/train2017"

cd "$HARMONIC_DIR"

echo "=========================================="
echo "HARMONIC Training Pipeline"
echo "=========================================="
echo ""

# Step 1: Check if dataset is ready
echo "[1/3] Checking COCO dataset..."
if [ ! -d "$TRAIN_DIR" ]; then
    echo "  Dataset not extracted yet. Running extraction monitor..."
    bash scripts/wait_and_extract_coco.sh
    
    if [ $? -ne 0 ]; then
        echo "✗ Error: Dataset setup failed"
        exit 1
    fi
else
    NUM_IMAGES=$(ls -1 "$TRAIN_DIR"/*.jpg 2>/dev/null | wc -l || echo "0")
    echo "  ✓ Dataset ready: $NUM_IMAGES images"
fi

# Step 2: Verify captions
echo ""
echo "[2/3] Verifying captions..."
if [ ! -f "$TRAIN_DIR/captions.json" ] && [ -f "$DATASET_DIR/captions.json" ]; then
    echo "  Copying captions.json to train directory..."
    cp "$DATASET_DIR/captions.json" "$TRAIN_DIR/"
fi

if [ -f "$TRAIN_DIR/captions.json" ]; then
    echo "  ✓ Captions ready"
else
    echo "  ✗ Error: captions.json not found"
    exit 1
fi

# Step 3: Clean old checkpoints and start training
echo ""
echo "[3/3] Starting training..."
echo ""
read -p "Clean old checkpoints? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf checkpoints/*
    echo "  ✓ Cleaned checkpoints"
fi

echo ""
echo "=========================================="
echo "Launching Multi-GPU Training"
echo "=========================================="
echo "Dataset: $TRAIN_DIR"
echo "GPUs: 0, 1"
echo "Batch Size: 32 per GPU (64 total)"
echo "Epochs: 200"
echo "=========================================="
echo ""
echo "Training will run in the background."
echo "Monitor with: tail -f logs/training_*.log"
echo ""
read -p "Press Enter to start training..."

# Update train_multi_gpu.sh to use correct path
sed -i "s|DATA_DIR=.*|DATA_DIR=\"$TRAIN_DIR\"|g" train_multi_gpu.sh

# Start training
nohup bash train_multi_gpu.sh > /dev/null 2>&1 &

TRAIN_PID=$!
echo ""
echo "✓ Training started (PID: $TRAIN_PID)"
echo ""
echo "Commands:"
echo "  Monitor training:  tail -f logs/training_*.log"
echo "  Check GPU usage:   nvtop"
echo "  Stop training:     kill $TRAIN_PID"
echo ""
echo "=========================================="
