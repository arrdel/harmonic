#!/usr/bin/env python3
"""
HARMONIC Training Script

Train the HARMONIC conflict resolution modules on paired text-image data.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
CLIP_PATH = SCRIPT_DIR / 'CLIP'
GUIDED_DIFF_PATH = SCRIPT_DIR / 'guided-diffusion'
MGAD_GD_PATH = SCRIPT_DIR.parent / 'MGAD-multimodal-guided-artwork-diffusion' / 'guided-diffusion'

if CLIP_PATH.exists():
    sys.path.insert(0, str(CLIP_PATH))
if GUIDED_DIFF_PATH.exists():
    sys.path.insert(0, str(GUIDED_DIFF_PATH))
if MGAD_GD_PATH.exists():
    sys.path.insert(0, str(MGAD_GD_PATH))

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Run setup.py first.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from harmonic import (
    SemanticConflictDetector,
    CrossModalAttentionFusion,
    TemporalGuidanceScheduler,
    HARMONIC
)


class TextImageDataset(Dataset):
    """Dataset for paired text-image training."""
    
    def __init__(self, data_dir, transform=None, clip_preprocess=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.clip_preprocess = clip_preprocess
        
        # Find all image files
        self.image_files = []
        self.captions = {}
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.image_files.extend(self.data_dir.glob(f'**/{ext}'))
        
        # Look for captions file
        captions_file = self.data_dir / 'captions.json'
        if captions_file.exists():
            with open(captions_file) as f:
                self.captions = json.load(f)
        
        # Filter to only images with captions
        if self.captions:
            self.image_files = [f for f in self.image_files if f.name in self.captions]
        
        print(f"Found {len(self.image_files)} image-caption pairs")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.zeros(3, 224, 224)
        
        # Get caption
        caption = self.captions.get(img_path.name, "")
        
        # Prepare CLIP inputs on CPU (encoding happens in training loop on GPU)
        if self.clip_preprocess is not None:
            clip_image = self.clip_preprocess(Image.open(img_path))  # CPU tensor
        else:
            clip_image = torch.zeros(3, 224, 224)
        
        text_tokens = clip.tokenize([caption], truncate=True).squeeze(0) if CLIP_AVAILABLE else torch.zeros(77, dtype=torch.long)
        
        return {
            'image': image_tensor,
            'clip_image': clip_image,
            'text_tokens': text_tokens,
            'caption': caption,
            'path': str(img_path)
        }


class HARMONICTrainer:
    """Trainer for HARMONIC modules."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize CLIP
        if CLIP_AVAILABLE:
            print("Loading CLIP model...")
            self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device)
            self.clip_model.eval()
        else:
            self.clip_model = None
            self.clip_preprocess = None
        
        # Initialize HARMONIC
        print("Initializing HARMONIC modules...")
        self.harmonic = HARMONIC(
            embed_dim=512,
            num_heads=args.num_heads,
            max_timesteps=args.num_timesteps,
            schedule_type=args.schedule_type,
            device=str(self.device)
        ).to(self.device)
        
        # Wrap in DataParallel if multiple GPUs visible
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"Using DataParallel across {self.num_gpus} GPUs")
            self.harmonic = nn.DataParallel(self.harmonic)
        
        # Keep reference to underlying module for saving/loading
        self.harmonic_module = self.harmonic.module if self.num_gpus > 1 else self.harmonic
        
        # Optimizer
        self.optimizer = AdamW(
            self.harmonic.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate * 0.01
        )
        
        # Setup logging
        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # WandB
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.init(
                project="harmonic",
                name=args.run_name or f"harmonic-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=vars(args)
            )
    
    def create_dataloaders(self):
        """Create training and validation dataloaders."""
        transform = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Training dataset - CLIP encoding happens in train loop, not workers
        train_dataset = TextImageDataset(
            self.args.train_data,
            transform=transform,
            clip_preprocess=self.clip_preprocess,
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        # Validation dataset (if provided)
        if self.args.val_data:
            val_dataset = TextImageDataset(
                self.args.val_data,
                transform=transform,
                clip_preprocess=self.clip_preprocess,
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        else:
            self.val_loader = None
    
    def compute_loss(self, text_embed, image_embed, timestep):
        """
        Compute training loss for HARMONIC.
        
        The loss encourages:
        1. Smooth fusion of embeddings
        2. Appropriate conflict detection
        3. Temporal consistency
        """
        # Get HARMONIC guidance - forward() returns a dictionary
        output = self.harmonic(text_embed, image_embed, timestep, return_all=True)
        guidance_embed = output['guidance']
        info = {
            'conflict_score': output['conflict_score'].mean().item(),
            'text_weight': output['w_text'].mean().item(),
            'image_weight': output['w_img'].mean().item()
        }
        
        # Loss 1: Reconstruction - guidance should preserve information from both modalities
        # Target: weighted combination based on conflict (low conflict = simple average)
        target_embed = (text_embed + image_embed) / 2
        recon_loss = F.mse_loss(guidance_embed, target_embed)
        
        # Loss 2: Alignment - guidance should be similar to both inputs
        # Encourages the model to extract useful information from both
        text_align = 1 - F.cosine_similarity(guidance_embed, text_embed).mean()
        image_align = 1 - F.cosine_similarity(guidance_embed, image_embed).mean()
        align_loss = (text_align + image_align) / 2
        
        # Loss 3: Diversity - different inputs should produce different outputs
        batch_size = text_embed.shape[0]
        if batch_size > 1:
            # Compare guidance embeddings within batch
            # Normalize embeddings
            guidance_norm = F.normalize(guidance_embed, dim=-1)
            # Compute pairwise similarities
            sim_matrix = torch.mm(guidance_norm, guidance_norm.t())
            # We want off-diagonal elements to be low (diverse outputs)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=guidance_embed.device)
            diversity_loss = sim_matrix[mask].mean().clamp(min=0)
        else:
            diversity_loss = torch.tensor(0.0, device=guidance_embed.device)
        
        # Loss 4: Temporal smoothness - similar timesteps should give similar weights
        if timestep > 0:
            prev_timestep = timestep - 1
            with torch.no_grad():
                prev_output = self.harmonic(text_embed, image_embed, prev_timestep, return_all=True)
            temporal_loss = F.mse_loss(
                torch.stack([output['w_text'], output['w_img']]),
                torch.stack([prev_output['w_text'].detach(), prev_output['w_img'].detach()])
            )
        else:
            temporal_loss = torch.tensor(0.0, device=guidance_embed.device)
        
        # Combined loss - all components should be non-negative
        total_loss = (
            self.args.recon_weight * recon_loss +
            0.5 * align_loss +  # New alignment term
            self.args.contrast_weight * diversity_loss +  # Renamed from contrast
            self.args.temporal_weight * temporal_loss
        )
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'align_loss': align_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'temporal_loss': temporal_loss.item(),
            'conflict_score': info['conflict_score'],
            'text_weight': info['text_weight'],
            'image_weight': info['image_weight']
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.harmonic.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            # Encode with CLIP on GPU (main process, not in DataLoader workers)
            with torch.no_grad():
                clip_images = batch['clip_image'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                image_embed = self.clip_model.encode_image(clip_images).float()
                text_embed = self.clip_model.encode_text(text_tokens).float()
            
            # Sample random timestep
            timestep = torch.randint(0, self.args.num_timesteps, (1,)).item()
            
            # Forward pass
            loss, metrics = self.compute_loss(text_embed, image_embed, timestep)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.harmonic.parameters(), 
                    self.args.grad_clip
                )
            
            self.optimizer.step()
            
            # Update stats
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'conflict': f"{metrics['conflict_score']:.3f}"
            })
            
            # Log to wandb
            if WANDB_AVAILABLE and self.args.use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/recon_loss': metrics['recon_loss'],
                    'train/align_loss': metrics['align_loss'],
                    'train/diversity_loss': metrics['diversity_loss'],
                    'train/temporal_loss': metrics['temporal_loss'],
                    'train/conflict_score': metrics['conflict_score'],
                    'train/text_weight': metrics['text_weight'],
                    'train/image_weight': metrics['image_weight'],
                    'train/timestep': timestep
                })
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model."""
        if self.val_loader is None:
            return None
        
        self.harmonic.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Encode with CLIP on GPU
            clip_images = batch['clip_image'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)
            image_embed = self.clip_model.encode_image(clip_images).float()
            text_embed = self.clip_model.encode_text(text_tokens).float()
            
            timestep = self.args.num_timesteps // 2  # Use middle timestep for validation
            
            loss, metrics = self.compute_loss(text_embed, image_embed, timestep)
            total_loss += loss.item()
            num_batches += 1
        
        val_loss = total_loss / max(num_batches, 1)
        
        if WANDB_AVAILABLE and self.args.use_wandb:
            wandb.log({'val/loss': val_loss, 'epoch': epoch})
        
        return val_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.harmonic_module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'args': vars(self.args)
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save periodic checkpoint
        if (epoch + 1) % self.args.save_every == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch+1:04d}.pt'
            torch.save(checkpoint, epoch_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.harmonic_module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch']
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting HARMONIC Training")
        print("="*60)
        
        # Create dataloaders
        self.create_dataloaders()
        
        # Resume from checkpoint if provided
        start_epoch = 0
        if self.args.resume:
            start_epoch = self.load_checkpoint(self.args.resume) + 1
        
        best_loss = float('inf')
        
        for epoch in range(start_epoch, self.args.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Print stats
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Save checkpoint - track best by val_loss if available, else train_loss
            current_loss = val_loss if val_loss is not None else train_loss
            is_best = current_loss < best_loss
            if is_best:
                best_loss = current_loss
            self.save_checkpoint(epoch, train_loss, is_best)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best validation loss: {best_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print("="*60)
        
        if WANDB_AVAILABLE and self.args.use_wandb:
            wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description='Train HARMONIC')
    
    # Data
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data directory')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for CLIP')
    
    # Model
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--hidden_dim', type=int, default=2048,
                        help='Hidden dimension')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--schedule_type', type=str, default='conflict_aware',
                        choices=['linear', 'cosine', 'adaptive', 'conflict_aware'],
                        help='Guidance schedule type')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Loss weights
    parser.add_argument('--recon_weight', type=float, default=1.0,
                        help='Reconstruction loss weight')
    parser.add_argument('--contrast_weight', type=float, default=0.5,
                        help='Contrastive loss weight')
    parser.add_argument('--temporal_weight', type=float, default=0.1,
                        help='Temporal smoothness loss weight')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for this run')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check for CLIP
    if not CLIP_AVAILABLE:
        print("Error: CLIP is required for training.")
        print("Please run: python setup.py")
        sys.exit(1)
    
    # Create trainer
    trainer = HARMONICTrainer(args)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
