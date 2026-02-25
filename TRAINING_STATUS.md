# HARMONIC Training Setup - Status

## ✅ Environment Setup Complete (Feb 25, 2026)

### Conda Environment: `harmonic_env`
- **Python**: 3.10
- **PyTorch**: 2.5.1+cu121
- **CUDA**: 6x NVIDIA GeForce RTX 4090 (24GB each)
- **Key packages**: CLIP, guided-diffusion, transformers, accelerate, wandb, lpips, einops

### Activate with:
```bash
conda activate harmonic_env
```

## ✅ Dataset Ready

### COCO 2017 Train
- **Location**: `/media/scratch/adele/harmonic/dataset/`
- **Symlink**: `data/coco` → `/media/scratch/adele/harmonic/dataset/`
- **Images**: 118,288 training images (`train2017/`)
- **Captions**: 118,287 image-caption pairs (`captions.json`)
- **Annotations**: Full COCO annotations in `annotations/`

## ✅ Core Architecture Verified

### HARMONIC Module (8.4M parameters)
- **Semantic Conflict Detector**: Multi-head attention decomposition (8 heads)
- **Cross-Modal Attention Fusion**: Bidirectional cross-attention (2 layers)
- **Temporal Guidance Scheduler**: Conflict-aware cosine scheduling
- **Self-test**: ✅ All tests passed on CUDA

### HARMONIC Lite (1.3M parameters)
- Lightweight variant for faster inference
- **Self-test**: ✅ Passed

## ✅ External Dependencies
- **CLIP** (`CLIP/`): OpenAI CLIP ViT-B/32 - installed in editable mode
- **guided-diffusion** (`guided-diffusion/`): OpenAI guided diffusion - installed in editable mode

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | COCO 2017 Train (118K images) |
| **GPUs** | 2 (configurable, 6 available) |
| **Batch Size** | 32 per GPU (64 total) |
| **Epochs** | 200 |
| **Learning Rate** | 1e-4 |
| **Scheduler** | Cosine Annealing |
| **Schedule Type** | conflict_aware |

## 🚀 Next Steps

### 1. Download Pretrained Diffusion Models
```bash
conda activate harmonic_env
cd /home/adelechinda/home/projects/harmonic
mkdir -p /media/scratch/adele/harmonic/models

# 256x256 model (~2GB)
wget -O /media/scratch/adele/harmonic/models/256x256_diffusion_uncond.pt \
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"

# 512x512 model (~2GB)
wget -O /media/scratch/adele/harmonic/models/512x512_diffusion_uncond_finetune_008100.pt \
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion_uncond_finetune_008100.pt"
```

### 2. Start Training
```bash
conda activate harmonic_env
cd /home/adelechinda/home/projects/harmonic
bash train_multi_gpu.sh
```

### 3. Monitor Training
```bash
nvtop                           # GPU usage
tail -f logs/training_*.log     # Training progress
```

## ⏱️ Estimated Training Time
- **Per Epoch**: ~45 minutes (118K images, batch size 64, 2 GPUs)
- **Full Training (200 epochs)**: ~6.25 days
- **Checkpoints saved**: Every 20 epochs
