# HARMONIC
## Hierarchical Attention-based Reconciliation of Multimodal ONtologies for Image Creation

![HARMONIC Banner](assets/banner.svg)

A novel semantic conflict resolution framework for multimodal guided diffusion.

## 🚀 Quick Start

### Setup
```bash
cd HARMONIC

# Install dependencies
pip install -r requirements.txt

# Install CLIP
cd CLIP && pip install -e . && cd ..

# Install guided-diffusion (from MGAD)
pip install -e ../MGAD-multimodal-guided-artwork-diffusion/guided-diffusion
```

### Download Models
```bash
mkdir -p models
wget -O models/512x512_diffusion_uncond_finetune_008100.pt \
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion_uncond_finetune_008100.pt"
```

### Run
```bash
python run_harmonic.py \
    -t "A serene mountain landscape in impressionist style" \
    -i image_prompts/reference.jpg \
    -o results/
```

## 🔬 Key Innovation

Instead of discarding conflicting modalities, HARMONIC **reconciles** them:

1. **Semantic Conflict Detection** - Multi-head analysis of text-image alignment
2. **Cross-Modal Attention Fusion** - Extract complementary information from both
3. **Temporal Guidance Scheduling** - Dynamic weighting throughout diffusion

## 📁 Project Structure

```
HARMONIC/
├── harmonic/
│   ├── conflict_detector.py    # Semantic Conflict Detection
│   ├── fusion_module.py        # Cross-Modal Attention Fusion
│   ├── scheduler.py            # Temporal Guidance Scheduler
│   └── harmonic.py             # Main HARMONIC module (8.4M params)
├── run_harmonic.py             # Inference script
├── train_harmonic.py           # Training script
└── setup.py                    # Setup helper
```

## 📊 Training

```bash
python train_harmonic.py \
    --train_data /path/to/data \
    --epochs 100 \
    --batch_size 32 \
    --schedule_type conflict_aware
```
<!-- 
## Citation

```bibtex
@article{harmonic2026,
  title={HARMONIC: Semantic Conflict Resolution for Multimodal Guided Diffusion},
  author={Chinda, Adele},
  year={2026}
}
``` -->
