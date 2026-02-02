#!/bin/bash
# Monitor and extract COCO dataset when download completes

DATASET_DIR="/media/scratch/adele/harmonic/dataset"
ZIP_FILE="$DATASET_DIR/train2017.zip"
EXPECTED_SIZE=19336861798  # 18GB in bytes
TRAIN_DIR="$DATASET_DIR/train2017"

echo "=========================================="
echo "COCO Dataset Setup Monitor"
echo "=========================================="
echo ""

# Check if already extracted
if [ -d "$TRAIN_DIR" ] && [ -f "$TRAIN_DIR/captions.json" ]; then
    echo "✓ Dataset already extracted!"
    echo "  Location: $TRAIN_DIR"
    echo "  Ready for training!"
    exit 0
fi

# Monitor download progress
echo "Monitoring download progress..."
echo "Expected size: 18GB"
echo ""

while true; do
    if [ ! -f "$ZIP_FILE" ]; then
        echo "✗ Error: $ZIP_FILE not found"
        echo "  Download may have failed"
        exit 1
    fi
    
    CURRENT_SIZE=$(stat -f%z "$ZIP_FILE" 2>/dev/null || stat -c%s "$ZIP_FILE" 2>/dev/null)
    PROGRESS=$((CURRENT_SIZE * 100 / EXPECTED_SIZE))
    SIZE_GB=$(echo "scale=2; $CURRENT_SIZE / 1024 / 1024 / 1024" | bc)
    
    echo -ne "\rProgress: ${PROGRESS}% (${SIZE_GB} GB / 18 GB)   "
    
    # Check if download complete
    if [ "$CURRENT_SIZE" -ge "$EXPECTED_SIZE" ]; then
        echo ""
        echo ""
        echo "✓ Download complete!"
        break
    fi
    
    sleep 5
done

# Wait a bit to ensure file is fully written
sleep 3

# Extract the dataset
echo ""
echo "=========================================="
echo "Extracting images..."
echo "=========================================="
echo "This will take 5-10 minutes..."
echo ""

cd "$DATASET_DIR"
unzip -q train2017.zip

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Extraction complete!"
    
    # Move captions.json into train2017 directory
    if [ -f "$DATASET_DIR/captions.json" ] && [ ! -f "$TRAIN_DIR/captions.json" ]; then
        cp "$DATASET_DIR/captions.json" "$TRAIN_DIR/"
        echo "✓ Copied captions.json to train2017/"
    fi
    
    # Count images
    NUM_IMAGES=$(ls -1 "$TRAIN_DIR"/*.jpg 2>/dev/null | wc -l)
    
    echo ""
    echo "=========================================="
    echo "✅ COCO Dataset Ready!"
    echo "=========================================="
    echo "Location: $TRAIN_DIR"
    echo "Images: $NUM_IMAGES"
    echo "Captions: $TRAIN_DIR/captions.json"
    echo ""
    echo "You can now start training:"
    echo "  cd /home/adelechinda/home/semester_projects/spring_26/computer_vision/HARMONIC"
    echo "  bash train_multi_gpu.sh"
    echo "=========================================="
else
    echo ""
    echo "✗ Error: Extraction failed"
    echo "  Check disk space and permissions"
    exit 1
fi
