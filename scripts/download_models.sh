#!/bin/bash
# Download pretrained models for HARMONIC

set -e

echo "=========================================="
echo "HARMONIC Model Download Script"
echo "=========================================="

# Create models directory
mkdir -p models

# 1. Download 512x512 diffusion model
echo ""
echo "Downloading 512x512 diffusion model..."
if [ ! -f "models/512x512_diffusion_uncond_finetune_008100.pt" ]; then
    wget -O models/512x512_diffusion_uncond_finetune_008100.pt \
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion_uncond_finetune_008100.pt"
    echo "✓ Downloaded 512x512 model"
else
    echo "✓ 512x512 model already exists"
fi

# 2. Download 256x256 diffusion model
echo ""
echo "Downloading 256x256 diffusion model..."
if [ ! -f "models/256x256_diffusion_uncond.pt" ]; then
    wget -O models/256x256_diffusion_uncond.pt \
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
    echo "✓ Downloaded 256x256 model"
else
    echo "✓ 256x256 model already exists"
fi

# 3. Clone CLIP if not present
echo ""
echo "Setting up CLIP..."
if [ ! -d "CLIP" ]; then
    git clone https://github.com/openai/CLIP.git
    echo "✓ Cloned CLIP"
else
    echo "✓ CLIP already exists"
fi

# 4. Clone guided-diffusion if not present
echo ""
echo "Setting up guided-diffusion..."
if [ ! -d "guided-diffusion" ]; then
    git clone https://github.com/openai/guided-diffusion.git
    cd guided-diffusion
    pip install -e .
    cd ..
    echo "✓ Setup guided-diffusion"
else
    echo "✓ guided-diffusion already exists"
fi

# 5. Download secondary model (YFCC trained)
echo ""
echo "Downloading secondary model..."
if [ ! -f "models/yfcc_1.pth" ]; then
    # Note: You may need to find/train this model
    echo "⚠ Secondary model (yfcc_1.pth) not available for automatic download"
    echo "  See MGAD repository for training instructions"
else
    echo "✓ Secondary model already exists"
fi

# 6. Create symlinks for convenience
echo ""
echo "Creating symlinks..."
ln -sf models/512x512_diffusion_uncond_finetune_008100.pt 512x512_diffusion_uncond_finetune_008100.pt 2>/dev/null || true
ln -sf models/256x256_diffusion_uncond.pt 256x256_diffusion_uncond.pt 2>/dev/null || true

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "You can now run HARMONIC with:"
echo "  python run_harmonic.py -t 'your prompt' -i image.jpg -o results/"
