#!/usr/bin/env python3
"""
HARMONIC Setup Script

Sets up the environment for HARMONIC by:
1. Cloning CLIP and guided-diffusion if not present
2. Installing dependencies
3. Downloading pretrained models
"""

import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()


def run_cmd(cmd, cwd=None):
    """Run a shell command."""
    print(f"  > {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    Error: {result.stderr}")
        return False
    return True


def setup_clip():
    """Clone and setup CLIP."""
    clip_path = SCRIPT_DIR / 'CLIP'
    if clip_path.exists():
        print("✓ CLIP already exists")
        return True
    
    print("Cloning CLIP...")
    if run_cmd("git clone https://github.com/openai/CLIP.git", cwd=SCRIPT_DIR):
        print("✓ CLIP cloned successfully")
        return True
    return False


def setup_guided_diffusion():
    """Clone and setup guided-diffusion."""
    gd_path = SCRIPT_DIR / 'guided-diffusion'
    
    # Check if we can use MGAD's guided-diffusion
    mgad_gd = SCRIPT_DIR.parent / 'MGAD-multimodal-guided-artwork-diffusion' / 'guided-diffusion'
    if mgad_gd.exists():
        print(f"✓ Using guided-diffusion from MGAD: {mgad_gd}")
        return True
    
    if gd_path.exists():
        print("✓ guided-diffusion already exists")
        return True
    
    print("Cloning guided-diffusion...")
    if run_cmd("git clone https://github.com/openai/guided-diffusion.git", cwd=SCRIPT_DIR):
        print("Installing guided-diffusion...")
        run_cmd("pip install -e .", cwd=gd_path)
        print("✓ guided-diffusion setup complete")
        return True
    return False


def setup_models():
    """Download pretrained models."""
    models_dir = SCRIPT_DIR / 'models'
    models_dir.mkdir(exist_ok=True)
    
    models = {
        '512x512_diffusion_uncond_finetune_008100.pt': 
            'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion_uncond_finetune_008100.pt',
        '256x256_diffusion_uncond.pt':
            'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
    }
    
    for model_name, url in models.items():
        model_path = models_dir / model_name
        if model_path.exists():
            print(f"✓ {model_name} already exists")
        else:
            print(f"Downloading {model_name}...")
            print("  This may take a while (~2GB each)...")
            if run_cmd(f"wget -O {model_path} '{url}'"):
                print(f"✓ Downloaded {model_name}")
            else:
                print(f"✗ Failed to download {model_name}")
                print(f"  Manual download: {url}")


def install_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    req_file = SCRIPT_DIR / 'requirements.txt'
    if req_file.exists():
        run_cmd(f"pip install -r {req_file}")
        print("✓ Dependencies installed")
    else:
        print("✗ requirements.txt not found")


def verify_setup():
    """Verify the setup is complete."""
    print("\n" + "="*50)
    print("Verifying setup...")
    print("="*50)
    
    # Check CLIP
    clip_ok = False
    try:
        sys.path.insert(0, str(SCRIPT_DIR / 'CLIP'))
        import clip
        clip_ok = True
        print("✓ CLIP import successful")
    except ImportError:
        print("✗ CLIP import failed")
    
    # Check guided-diffusion
    gd_ok = False
    try:
        gd_path = SCRIPT_DIR / 'guided-diffusion'
        mgad_gd = SCRIPT_DIR.parent / 'MGAD-multimodal-guided-artwork-diffusion' / 'guided-diffusion'
        if gd_path.exists():
            sys.path.insert(0, str(gd_path))
        elif mgad_gd.exists():
            sys.path.insert(0, str(mgad_gd))
        
        from guided_diffusion.script_util import create_model_and_diffusion
        gd_ok = True
        print("✓ guided_diffusion import successful")
    except ImportError:
        print("✗ guided_diffusion import failed")
    
    # Check PyTorch
    torch_ok = False
    try:
        import torch
        torch_ok = True
        cuda_str = "CUDA available" if torch.cuda.is_available() else "CPU only"
        print(f"✓ PyTorch {torch.__version__} ({cuda_str})")
    except ImportError:
        print("✗ PyTorch not found")
    
    # Check models
    models_dir = SCRIPT_DIR / 'models'
    model_512 = models_dir / '512x512_diffusion_uncond_finetune_008100.pt'
    if model_512.exists():
        print(f"✓ 512x512 model found ({model_512.stat().st_size / 1e9:.2f} GB)")
    else:
        print("✗ 512x512 model not found (optional)")
    
    print("\n" + "="*50)
    if clip_ok and gd_ok and torch_ok:
        print("Setup complete! You can now run HARMONIC.")
        print("\nExample:")
        print('  python run_harmonic.py -t "a beautiful sunset" -o output/')
    else:
        print("Setup incomplete. Please fix the errors above.")
    print("="*50)


def main():
    print("="*60)
    print("HARMONIC Setup")
    print("="*60)
    print()
    
    # 1. Setup CLIP
    print("\n[1/4] Setting up CLIP...")
    setup_clip()
    
    # 2. Setup guided-diffusion
    print("\n[2/4] Setting up guided-diffusion...")
    setup_guided_diffusion()
    
    # 3. Install dependencies
    print("\n[3/4] Installing dependencies...")
    install_dependencies()
    
    # 4. Download models (optional, large files)
    print("\n[4/4] Downloading pretrained models...")
    response = input("Download pretrained models? (~4GB total) [y/N]: ").strip().lower()
    if response == 'y':
        setup_models()
    else:
        print("Skipping model download. You can run this later:")
        print("  bash scripts/download_models.sh")
    
    # Verify
    verify_setup()


if __name__ == '__main__':
    main()
