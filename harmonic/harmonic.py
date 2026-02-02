"""
HARMONIC: Hierarchical Attention-based Reconciliation of Multimodal ONtologies for Image Creation

Main module that integrates:
1. Semantic Conflict Detection
2. Cross-Modal Attention Fusion
3. Temporal Guidance Scheduling

This module provides the unified interface for multimodal guidance
with intelligent conflict resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from .conflict_detector import SemanticConflictDetector, MultiScaleConflictDetector
from .fusion_module import CrossModalAttentionFusion, AdaptiveFusion
from .scheduler import TemporalGuidanceScheduler, PhaseAwareScheduler


class HARMONIC(nn.Module):
    """
    Complete HARMONIC module: Hierarchical Attention-based Reconciliation
    of Multimodal ONtologies for Image Creation.
    
    Integrates all three components for intelligent multimodal guidance:
    - Detects semantic conflicts between text and image prompts
    - Fuses complementary information from both modalities
    - Dynamically schedules guidance based on timestep and conflict
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_fusion_layers: int = 2,
        max_timesteps: int = 1000,
        schedule_type: str = "conflict_aware",
        use_multi_scale_conflict: bool = False,
        use_phase_aware_scheduler: bool = False,
        dropout: float = 0.1,
        device: str = "cuda"
    ):
        """
        Args:
            embed_dim: Dimension of CLIP embeddings (512 for ViT-B/32)
            num_heads: Number of attention heads
            num_fusion_layers: Number of cross-attention layers in fusion
            max_timesteps: Maximum diffusion timesteps
            schedule_type: Type of guidance schedule
            use_multi_scale_conflict: Use multi-scale conflict detection
            use_phase_aware_scheduler: Use phase-aware scheduling
            dropout: Dropout rate
            device: Device to run on
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_timesteps = max_timesteps
        self.device = device
        
        # Component 1: Semantic Conflict Detector
        if use_multi_scale_conflict:
            self.conflict_detector = MultiScaleConflictDetector(
                embed_dim=embed_dim,
                scales=[4, 8, 16],
                dropout=dropout
            )
        else:
            self.conflict_detector = SemanticConflictDetector(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # Component 2: Cross-Modal Attention Fusion
        self.fusion_module = CrossModalAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_fusion_layers,
            dropout=dropout
        )
        
        # Component 3: Temporal Guidance Scheduler
        if use_phase_aware_scheduler:
            self.scheduler = PhaseAwareScheduler(
                max_timesteps=max_timesteps,
                schedule_type=schedule_type
            )
        else:
            self.scheduler = TemporalGuidanceScheduler(
                max_timesteps=max_timesteps,
                schedule_type=schedule_type
            )
        
        # Learnable combination weights for final guidance
        self.guidance_combiner = nn.Sequential(
            nn.Linear(embed_dim * 3 + 3, embed_dim),  # [text, image, fused, w_text, w_img, conflict]
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Final output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(
        self,
        text_embed: torch.Tensor,
        image_embed: torch.Tensor,
        timestep: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass of HARMONIC.
        
        Args:
            text_embed: (batch, embed_dim) CLIP text embedding
            image_embed: (batch, embed_dim) CLIP image embedding
            timestep: Current diffusion timestep
            return_all: Whether to return all intermediate values
            
        Returns:
            Dictionary containing:
                - guidance: (batch, embed_dim) final harmonized guidance
                - conflict_score: (batch,) detected conflict level
                - w_text: (batch,) text weight for this timestep
                - w_img: (batch,) image weight for this timestep
                - fused_embed: (batch, embed_dim) cross-modal fused embedding
                - aspect_similarities: (batch, num_heads) per-aspect similarity (optional)
        """
        batch_size = text_embed.shape[0]
        
        # Ensure inputs are on the right device
        text_embed = text_embed.to(self.device)
        image_embed = image_embed.to(self.device)
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep], device=self.device)
        else:
            timestep = timestep.to(self.device)
        
        # Step 1: Detect semantic conflict
        conflict_result = self.conflict_detector(text_embed, image_embed)
        if isinstance(conflict_result, tuple):
            conflict_score, aspect_similarities = conflict_result
        else:
            conflict_score = conflict_result
            aspect_similarities = None
        
        # Step 2: Cross-modal fusion
        fused_embed = self.fusion_module(text_embed, image_embed, conflict_score)
        
        # Step 3: Temporal scheduling
        w_text, w_img = self.scheduler(timestep, conflict_score)
        
        # Ensure weights are the right shape (batch_size,)
        if w_text.dim() == 0:
            w_text = w_text.expand(batch_size)
        elif w_text.dim() > 1:
            w_text = w_text.squeeze()
            if w_text.dim() == 0:
                w_text = w_text.expand(batch_size)
        
        if w_img.dim() == 0:
            w_img = w_img.expand(batch_size)
        elif w_img.dim() > 1:
            w_img = w_img.squeeze()
            if w_img.dim() == 0:
                w_img = w_img.expand(batch_size)
        
        # Ensure batch dimension matches
        if w_text.shape[0] != batch_size:
            w_text = w_text[:1].expand(batch_size)
        if w_img.shape[0] != batch_size:
            w_img = w_img[:1].expand(batch_size)
        
        # Expand weights for broadcasting with embeddings
        w_text_expanded = w_text.unsqueeze(-1)  # (batch, 1)
        w_img_expanded = w_img.unsqueeze(-1)    # (batch, 1)
        
        # Compute weighted individual embeddings
        weighted_text = w_text_expanded * text_embed
        weighted_img = w_img_expanded * image_embed
        
        # Combine: Fused embedding gets stronger influence when conflict is high
        conflict_expanded = conflict_score.unsqueeze(-1)
        fused_weight = conflict_expanded * 0.5  # Max 50% fused when full conflict
        
        # Build input for guidance combiner - ensure all have shape (batch, dim)
        w_text_for_cat = w_text.unsqueeze(-1)  # (batch, 1)
        w_img_for_cat = w_img.unsqueeze(-1)    # (batch, 1)
        conflict_for_cat = conflict_score.unsqueeze(-1)  # (batch, 1)
        
        combiner_input = torch.cat([
            weighted_text,
            weighted_img,
            fused_embed,
            w_text_for_cat,
            w_img_for_cat,
            conflict_for_cat
        ], dim=-1)
        
        # Compute combined guidance
        combined = self.guidance_combiner(combiner_input)
        
        # Final guidance with residual paths
        # When conflict is low: rely more on weighted combination
        # When conflict is high: rely more on learned fusion
        final_guidance = (
            (1 - fused_weight) * (weighted_text + weighted_img) +
            fused_weight * combined
        )
        
        # Output projection
        final_guidance = self.output_proj(final_guidance)
        
        # Build output dictionary
        output = {
            'guidance': final_guidance,
            'conflict_score': conflict_score,
            'w_text': w_text,
            'w_img': w_img,
            'fused_embed': fused_embed
        }
        
        if return_all and aspect_similarities is not None:
            output['aspect_similarities'] = aspect_similarities
            output['weighted_text'] = weighted_text
            output['weighted_img'] = weighted_img
        
        return output
    
    def compute_guidance_loss(
        self,
        current_embed: torch.Tensor,
        text_embed: torch.Tensor,
        image_embed: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the CLIP guidance loss using HARMONIC.
        This replaces the fixed-weight loss in original MGAD.
        
        Args:
            current_embed: (batch*cutn, embed_dim) CLIP embeddings of current generation
            text_embed: (num_text_prompts, embed_dim) CLIP text embeddings
            image_embed: (num_image_prompts, embed_dim) CLIP image embeddings
            timestep: Current diffusion timestep
            
        Returns:
            loss: Scalar guidance loss
        """
        # Get mean embeddings for conflict detection
        text_mean = text_embed.mean(dim=0, keepdim=True)
        image_mean = image_embed.mean(dim=0, keepdim=True) if image_embed.numel() > 0 else text_mean
        
        # Get HARMONIC guidance
        harmonic_out = self.forward(text_mean, image_mean, timestep)
        guidance = harmonic_out['guidance']
        
        # Compute spherical distance loss
        current_norm = F.normalize(current_embed, dim=-1)
        guidance_norm = F.normalize(guidance, dim=-1)
        
        # Spherical distance
        dist = (current_norm - guidance_norm).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
        
        return dist.mean()
    
    def get_diagnostics(
        self,
        text_embed: torch.Tensor,
        image_embed: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Get diagnostic information about the conflict and fusion.
        Useful for debugging and visualization.
        """
        # Conflict breakdown
        if hasattr(self.conflict_detector, 'get_conflict_breakdown'):
            conflict_breakdown = self.conflict_detector.get_conflict_breakdown(
                text_embed, image_embed
            )
        else:
            conflict_score, _ = self.conflict_detector(text_embed, image_embed)
            conflict_breakdown = {'overall': {'conflict_score': conflict_score.mean().item()}}
        
        # Schedule information
        if hasattr(self.scheduler, 'get_full_schedule'):
            conflict_score = conflict_breakdown.get('overall', {}).get('conflict_score', 0.5)
            schedule = self.scheduler.get_full_schedule(conflict_score=conflict_score)
        else:
            schedule = None
        
        return {
            'conflict_breakdown': conflict_breakdown,
            'schedule': schedule,
            'embed_dim': self.embed_dim,
            'max_timesteps': self.max_timesteps
        }


class HARMONICLite(nn.Module):
    """
    Lightweight version of HARMONIC for faster inference.
    Uses simpler components with fewer parameters.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        max_timesteps: int = 1000,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_timesteps = max_timesteps
        self.device = device
        
        # Simple conflict detection
        self.conflict_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Simple fusion
        self.fusion = AdaptiveFusion(embed_dim)
        
        # Simple scheduler (just cosine)
        self.register_buffer('_dummy', torch.tensor(0))
        
    def forward(
        self,
        text_embed: torch.Tensor,
        image_embed: torch.Tensor,
        timestep: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Simple conflict detection
        concat = torch.cat([text_embed, image_embed], dim=-1)
        conflict_score = self.conflict_proj(concat).squeeze(-1)
        
        # Simple cosine schedule
        t_norm = timestep.float() / self.max_timesteps
        w_text = torch.cos(3.14159 / 2 * (1 - t_norm))
        w_img = 1 - w_text
        
        # Simple fusion
        fused = self.fusion(text_embed, image_embed, conflict_score)
        
        # Combine
        guidance = w_text.unsqueeze(-1) * text_embed + w_img.unsqueeze(-1) * image_embed
        guidance = (1 - conflict_score.unsqueeze(-1) * 0.5) * guidance + \
                   conflict_score.unsqueeze(-1) * 0.5 * fused
        
        return {
            'guidance': guidance,
            'conflict_score': conflict_score,
            'w_text': w_text.expand(text_embed.shape[0]),
            'w_img': w_img.expand(text_embed.shape[0])
        }


def create_harmonic(
    variant: str = "full",
    embed_dim: int = 512,
    max_timesteps: int = 1000,
    device: str = "cuda",
    **kwargs
) -> nn.Module:
    """
    Factory function to create HARMONIC variants.
    
    Args:
        variant: One of 'full', 'lite', 'multi_scale', 'phase_aware'
        embed_dim: Embedding dimension
        max_timesteps: Max diffusion timesteps
        device: Device to use
        **kwargs: Additional arguments passed to constructor
        
    Returns:
        HARMONIC module instance
    """
    if variant == "lite":
        return HARMONICLite(embed_dim, max_timesteps, device).to(device)
    
    elif variant == "multi_scale":
        return HARMONIC(
            embed_dim=embed_dim,
            max_timesteps=max_timesteps,
            use_multi_scale_conflict=True,
            device=device,
            **kwargs
        ).to(device)
    
    elif variant == "phase_aware":
        return HARMONIC(
            embed_dim=embed_dim,
            max_timesteps=max_timesteps,
            use_phase_aware_scheduler=True,
            device=device,
            **kwargs
        ).to(device)
    
    else:  # "full" or default
        return HARMONIC(
            embed_dim=embed_dim,
            max_timesteps=max_timesteps,
            device=device,
            **kwargs
        ).to(device)


if __name__ == "__main__":
    # Test the complete HARMONIC module
    print("Testing HARMONIC module...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create HARMONIC
    harmonic = create_harmonic(variant="full", device=device)
    print(f"Created HARMONIC with {sum(p.numel() for p in harmonic.parameters()):,} parameters")
    
    # Test inputs
    batch_size = 4
    text_embed = torch.randn(batch_size, 512).to(device)
    image_embed = torch.randn(batch_size, 512).to(device)
    timestep = torch.tensor([500]).to(device)
    
    # Forward pass
    output = harmonic(text_embed, image_embed, timestep, return_all=True)
    
    print(f"\nOutput shapes:")
    print(f"  guidance: {output['guidance'].shape}")
    print(f"  conflict_score: {output['conflict_score'].shape}")
    print(f"  w_text: {output['w_text'].shape}")
    print(f"  w_img: {output['w_img'].shape}")
    print(f"  fused_embed: {output['fused_embed'].shape}")
    
    print(f"\nConflict score: {output['conflict_score'].mean().item():.4f}")
    print(f"Text weight: {output['w_text'].mean().item():.4f}")
    print(f"Image weight: {output['w_img'].mean().item():.4f}")
    
    # Test diagnostics
    diagnostics = harmonic.get_diagnostics(text_embed, image_embed)
    print(f"\nDiagnostics:")
    print(f"  Overall conflict: {diagnostics['conflict_breakdown']['overall']['conflict_score']:.4f}")
    
    # Test lite version
    print("\nTesting HARMONIC Lite...")
    harmonic_lite = create_harmonic(variant="lite", device=device)
    print(f"Created HARMONIC Lite with {sum(p.numel() for p in harmonic_lite.parameters()):,} parameters")
    
    output_lite = harmonic_lite(text_embed, image_embed, timestep)
    print(f"Lite output guidance shape: {output_lite['guidance'].shape}")
    
    print("\n✓ All HARMONIC tests passed!")
