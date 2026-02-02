# HARMONIC Training Setup - Status

## ✅ Completed Setup

### 1. Dataset Download (In Progress)
- **Location**: `/media/scratch/adele/harmonic/dataset/`
- **Status**: Downloading COCO 2017 Train (~18GB)
  - Current: ~6.5GB downloaded (36%)
  - ETA: ~6-8 minutes remaining
- **Monitor**: Background process extracting when complete
  - Log: `tail -f logs/coco_setup.log`

### 2. Annotations Ready ✓
- **COCO Annotations**: Extracted
- **Captions Created**: 118,287 image-caption pairs
- **File**: `/media/scratch/adele/harmonic/dataset/captions.json`

### 3. Symlinks Configured ✓
- **HARMONIC data/coco** → `/media/scratch/adele/harmonic/dataset`

### 4. Training Scripts Ready ✓
- `train_multi_gpu.sh` - Multi-GPU training (2 GPUs)
- `launch_training.sh` - Complete training pipeline
- `scripts/wait_and_extract_coco.sh` - Auto-extraction monitor

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | COCO 2017 Train (118K images) |
| **GPUs** | 2 (GPU 0, GPU 1) |
| **Batch Size** | 32 per GPU (64 total) |
| **Epochs** | 200 |
| **Learning Rate** | 1e-4 |
| **Scheduler** | Cosine Annealing |
| **Schedule Type** | conflict_aware |

## 🚀 How to Start Training

### Option 1: Automatic (Recommended)
```bash
cd /home/adelechinda/home/semester_projects/spring_26/computer_vision/HARMONIC
bash launch_training.sh
```

This will:
1. Wait for dataset extraction (if not done)
2. Verify setup
3. Clean old checkpoints (optional)
4. Launch multi-GPU training

### Option 2: Manual
```bash
# Wait for download to complete
tail -f logs/coco_setup.log

# When you see "✅ COCO Dataset Ready!", run:
cd /home/adelechinda/home/semester_projects/spring_26/computer_vision/HARMONIC
bash train_multi_gpu.sh
```

## 📈 Monitoring Training

### GPU Usage
```bash
nvtop
```

### Training Progress
```bash
tail -f logs/training_*.log
```

### Checkpoints
- Saved every 20 epochs to `checkpoints/`
- Latest: `checkpoints/latest.pt`
- Best: `checkpoints/best.pt`

## 🔍 Current Status Check

```bash
# Check dataset download progress
ls -lh /media/scratch/adele/harmonic/dataset/train2017.zip

# Check if extraction is complete
ls -lh /media/scratch/adele/harmonic/dataset/train2017/ | head

# Check extraction monitor
tail logs/coco_setup.log
```

## ⏱️ Estimated Timeline

1. **Download**: ~10 more minutes (36% complete)
2. **Extraction**: ~5-10 minutes
3. **Total until ready**: ~15-20 minutes

## 🎯 Expected Training Time

- **Per Epoch**: ~45 minutes (118K images, batch size 64)
- **200 Epochs**: ~150 hours (~6 days)
- **Checkpoints saved**: Every 20 epochs

## 📝 Notes

- Loss function has been fixed (no more negative values)
- Training uses corrected loss with alignment + diversity terms
- All components verified working (tested with 5 epochs)
- Multi-GPU setup ready (no DDP needed for simple parallelism)

---

**Next Step**: Wait for extraction monitor to complete, then run `bash launch_training.sh`
