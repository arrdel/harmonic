#!/usr/bin/env python3
"""
HARMONIC: Main Entry Point

Run multimodal guided diffusion with intelligent semantic conflict resolution.
"""

import argparse
import gc
import io
import math
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import requests
from tqdm import tqdm

# Add paths for CLIP and guided-diffusion
# First check for local clones, then check MGAD structure
SCRIPT_DIR = Path(__file__).parent.resolve()
CLIP_PATH = SCRIPT_DIR / 'CLIP'
GUIDED_DIFF_PATH = SCRIPT_DIR / 'guided-diffusion'

if CLIP_PATH.exists():
    sys.path.insert(0, str(CLIP_PATH))
if GUIDED_DIFF_PATH.exists():
    sys.path.insert(0, str(GUIDED_DIFF_PATH))

# Also check MGAD folder (reuse existing guided-diffusion)
PARENT_DIR = SCRIPT_DIR.parent
MGAD_PATH = PARENT_DIR / 'MGAD-multimodal-guided-artwork-diffusion'
if (MGAD_PATH / 'guided-diffusion').exists():
    sys.path.insert(0, str(MGAD_PATH / 'guided-diffusion'))

try:
    import clip
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please run: bash scripts/download_models.sh")
    print("Or clone CLIP and guided-diffusion manually")

from harmonic import HARMONIC


def parse_args():
    parser = argparse.ArgumentParser(description='HARMONIC: Multimodal Guided Diffusion with Conflict Resolution')
    
    # Prompts
    parser.add_argument("-t", "--text_prompt", type=str, required=True,
                        help="Text prompt for generation")
    parser.add_argument("-i", "--image_prompt", type=str, default=None,
                        help="Image prompt path or URL")
    parser.add_argument("--init_image", type=str, default=None,
                        help="Initial image for img2img")
    
    # HARMONIC settings
    parser.add_argument("--harmonic_variant", type=str, default="full",
                        choices=["full", "lite", "multi_scale", "phase_aware"],
                        help="HARMONIC variant to use")
    parser.add_argument("--schedule_type", type=str, default="conflict_aware",
                        choices=["linear", "cosine", "sigmoid", "conflict_aware", "learned"],
                        help="Guidance schedule type")
    
    # Diffusion settings
    parser.add_argument("--diffusion_steps", type=int, default=1000,
                        help="Number of diffusion steps")
    parser.add_argument("--skip_timesteps", type=int, default=0,
                        help="Skip timesteps for init image")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Output image size")
    
    # Guidance settings
    parser.add_argument("--clip_guidance_scale", type=float, default=5000,
                        help="CLIP guidance scale")
    parser.add_argument("--tv_scale", type=float, default=150,
                        help="Total variation loss scale")
    parser.add_argument("--range_scale", type=float, default=50,
                        help="Range loss scale")
    
    # Cutouts
    parser.add_argument("--cutn", type=int, default=16,
                        help="Number of cutouts for CLIP")
    parser.add_argument("--cut_pow", type=float, default=0.5,
                        help="Cutout power")
    
    # Output
    parser.add_argument("-o", "--output", type=str, default="./results",
                        help="Output directory")
    parser.add_argument("--save_frequency", type=int, default=100,
                        help="Save intermediate results every N steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    
    return parser.parse_args()


def fetch(url_or_path):
    """Fetch image from URL or local path."""
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


class MakeCutouts(nn.Module):
    """Generate random cutouts from an image for CLIP."""
    
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    """Spherical distance loss for CLIP embeddings."""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """Total variation loss for smoothness."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    """Loss to keep values in valid range."""
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def main():
    args = parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # Load CLIP
    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load('ViT-B/32', device=device, jit=False)
    clip_model.eval().requires_grad_(False)
    clip_size = clip_model.visual.input_resolution
    
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    
    # Load diffusion model
    print("Loading diffusion model...")
    model_config = model_and_diffusion_defaults()
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': args.diffusion_steps,
        'rescale_timesteps': True,
        'timestep_respacing': str(args.diffusion_steps),
        'image_size': args.image_size,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })
    
    model, diffusion = create_model_and_diffusion(**model_config)
    
    # Load pretrained weights
    if args.image_size == 512:
        model_path = '512x512_diffusion_uncond_finetune_008100.pt'
    else:
        model_path = '256x256_diffusion_uncond.pt'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print(f"Warning: Model weights not found at {model_path}")
        print("Download from: https://github.com/openai/guided-diffusion")
    
    model.requires_grad_(False).eval().to(device)
    if model_config['use_fp16']:
        model.convert_to_fp16()
    
    # Create HARMONIC
    print(f"Creating HARMONIC ({args.harmonic_variant})...")
    use_multi_scale = args.harmonic_variant == 'multi_scale'
    use_phase_aware = args.harmonic_variant == 'phase_aware'
    
    harmonic = HARMONIC(
        embed_dim=512,
        num_heads=8,
        max_timesteps=args.diffusion_steps,
        schedule_type=args.schedule_type,
        use_multi_scale_conflict=use_multi_scale,
        use_phase_aware_scheduler=use_phase_aware,
        device=str(device)
    )
    print(f"HARMONIC parameters: {sum(p.numel() for p in harmonic.parameters()):,}")
    
    # Setup cutouts
    make_cutouts = MakeCutouts(clip_size, args.cutn, args.cut_pow)
    
    # Encode text prompt
    print(f"Encoding text prompt: '{args.text_prompt}'")
    text_tokens = clip.tokenize([args.text_prompt]).to(device)
    text_embed = clip_model.encode_text(text_tokens).float()
    
    # Encode image prompt if provided
    if args.image_prompt:
        print(f"Encoding image prompt: '{args.image_prompt}'")
        img = Image.open(fetch(args.image_prompt)).convert('RGB')
        img = TF.resize(img, min(args.image_size, *img.size))
        img_batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        image_embed = clip_model.encode_image(normalize(img_batch)).float().mean(0, keepdim=True)
    else:
        # If no image prompt, use text embed (HARMONIC will handle this)
        image_embed = text_embed.clone()
        print("No image prompt provided, using text-only guidance")
    
    # Get initial conflict assessment
    harmonic_diag = harmonic.get_diagnostics(text_embed, image_embed)
    conflict = harmonic_diag['conflict_breakdown']['overall']['conflict_score']
    print(f"Initial conflict score: {conflict:.4f}")
    
    # Load init image if provided
    init = None
    if args.init_image:
        init = Image.open(fetch(args.init_image)).convert('RGB')
        init = init.resize((args.image_size, args.image_size), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
    
    # Guidance function using HARMONIC
    cur_t = None
    
    def cond_fn(x, t, y=None):
        nonlocal cur_t
        
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            
            # Get current prediction
            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
            alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device)
            sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device)
            
            # Denoise estimate
            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={})
            pred = out['pred_xstart']
            
            # Get CLIP embeddings of current generation
            clip_in = normalize(make_cutouts(pred.add(1).div(2)))
            current_embeds = clip_model.encode_image(clip_in).float()
            
            # Use HARMONIC for guidance
            harmonic_out = harmonic(
                text_embed.expand(n, -1),
                image_embed.expand(n, -1),
                torch.tensor([cur_t], device=device)
            )
            
            guidance = harmonic_out['guidance']
            w_text = harmonic_out['w_text']
            w_img = harmonic_out['w_img']
            
            # Compute loss
            # Spherical distance to harmonized guidance
            clip_loss = spherical_dist_loss(
                current_embeds,
                guidance.repeat(current_embeds.shape[0] // n, 1)
            ).mean()
            
            # Regularization losses
            tv_l = tv_loss(pred).sum()
            range_l = range_loss(pred).sum()
            
            # Total loss
            loss = (
                clip_loss * args.clip_guidance_scale +
                tv_l * args.tv_scale +
                range_l * args.range_scale
            )
            
            # Compute gradient
            grad = -torch.autograd.grad(loss, x)[0]
            
            return grad
    
    # Run diffusion
    print("\nStarting generation...")
    print(f"  Steps: {args.diffusion_steps}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    
    cur_t = diffusion.num_timesteps - args.skip_timesteps - 1
    
    samples = diffusion.p_sample_loop_progressive(
        model,
        (args.batch_size, 3, args.image_size, args.image_size),
        clip_denoised=True,
        model_kwargs={},
        cond_fn=cond_fn,
        progress=True,
        skip_timesteps=args.skip_timesteps,
        init_image=init,
    )
    
    # Process samples
    for step, sample in enumerate(tqdm(samples, desc="Generating")):
        cur_t -= 1
        
        if step % args.save_frequency == 0 or cur_t <= 0:
            for k, image in enumerate(sample['pred_xstart']):
                img_pil = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                
                filename = os.path.join(
                    args.output,
                    f"harmonic_step{step:04d}_b{k}.png"
                )
                img_pil.save(filename)
                
                if cur_t <= 0:
                    # Save final with metadata
                    final_name = os.path.join(args.output, f"final_{k}.png")
                    img_pil.save(final_name)
                    print(f"\nSaved final image: {final_name}")
    
    print("\nGeneration complete!")
    print(f"Results saved to: {args.output}")
    
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
