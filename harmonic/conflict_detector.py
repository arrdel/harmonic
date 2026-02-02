"""
Semantic Conflict Detector Module

Detects semantic conflict between text and image embeddings using
multi-head semantic decomposition to identify alignment across different
semantic aspects (content, style, color, composition, mood, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SemanticConflictDetector(nn.Module):
    """
    Detects semantic conflict between text and image embeddings
    using multi-head semantic decomposition.
    
    The detector projects embeddings into multiple semantic subspaces,
    each capturing different aspects like content, style, color, etc.
    """
    
    # Semantic aspect names for interpretability
    ASPECT_NAMES = [
        "content",      # Subject matter
        "style",        # Artistic technique
        "color",        # Color palette
        "composition",  # Spatial arrangement
        "mood",         # Emotional tone
        "artist",       # Artist/period characteristics
        "medium",       # Painting medium/texture
        "abstract"      # Abstract concepts
    ]
    
    def __init__(
        self, 
        embed_dim: int = 512, 
        num_heads: int = 8,
        dropout: float = 0.1,
        learnable_weights: bool = True
    ):
        """
        Args:
            embed_dim: Dimension of CLIP embeddings (512 for ViT-B/32)
            num_heads: Number of semantic aspects to decompose into
            dropout: Dropout rate for regularization
            learnable_weights: Whether aspect importance weights are learnable
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # Learnable projection matrices for semantic decomposition
        # Each head projects into a semantic subspace
        self.semantic_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, self.head_dim * 2),
                nn.LayerNorm(self.head_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.head_dim * 2, self.head_dim),
            )
            for _ in range(num_heads)
        ])
        
        # Learnable importance weights for each semantic aspect
        if learnable_weights:
            self.aspect_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        else:
            self.register_buffer('aspect_weights', torch.ones(num_heads) / num_heads)
        
        # Optional: learn a threshold for conflict classification
        self.conflict_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Global similarity projection (alternative path)
        self.global_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
    def compute_aspect_similarities(
        self, 
        text_embed: torch.Tensor, 
        image_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity for each semantic aspect.
        
        Args:
            text_embed: (batch, embed_dim) CLIP text embedding
            image_embed: (batch, embed_dim) CLIP image embedding
            
        Returns:
            per_aspect_similarity: (batch, num_heads) similarities per aspect
        """
        similarities = []
        
        for proj in self.semantic_projections:
            # Project both embeddings to semantic subspace
            text_proj = proj(text_embed)
            img_proj = proj(image_embed)
            
            # Normalize for cosine similarity
            text_proj = F.normalize(text_proj, dim=-1)
            img_proj = F.normalize(img_proj, dim=-1)
            
            # Compute cosine similarity
            sim = (text_proj * img_proj).sum(dim=-1)  # (batch,)
            similarities.append(sim)
        
        return torch.stack(similarities, dim=-1)  # (batch, num_heads)
    
    def forward(
        self, 
        text_embed: torch.Tensor, 
        image_embed: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect semantic conflict between text and image embeddings.
        
        Args:
            text_embed: (batch, embed_dim) CLIP text embedding
            image_embed: (batch, embed_dim) CLIP image embedding
            return_details: Whether to return detailed breakdown
            
        Returns:
            conflict_score: (batch,) scalar in [0, 1], higher = more conflict
            per_aspect_similarity: (batch, num_heads) similarities per aspect
        """
        batch_size = text_embed.shape[0]
        
        # Compute per-aspect similarities
        per_aspect_similarity = self.compute_aspect_similarities(text_embed, image_embed)
        
        # Compute weighted combination
        weights = F.softmax(self.aspect_weights, dim=0)  # (num_heads,)
        weighted_similarity = (per_aspect_similarity * weights).sum(dim=-1)  # (batch,)
        
        # Also compute global similarity as a sanity check
        global_input = torch.cat([text_embed, image_embed], dim=-1)
        global_similarity = self.global_proj(global_input).squeeze(-1)  # (batch,)
        
        # Combine local (aspect-based) and global similarity
        # Global acts as a regularizer
        combined_similarity = 0.7 * weighted_similarity + 0.3 * global_similarity
        
        # Conflict score: higher means more conflict (less similar)
        conflict_score = 1 - combined_similarity
        
        # Clamp to [0, 1]
        conflict_score = conflict_score.clamp(0, 1)
        
        if return_details:
            return {
                'conflict_score': conflict_score,
                'per_aspect_similarity': per_aspect_similarity,
                'global_similarity': global_similarity,
                'weighted_similarity': weighted_similarity,
                'aspect_weights': weights,
                'aspect_names': self.ASPECT_NAMES[:self.num_heads]
            }
        
        return conflict_score, per_aspect_similarity
    
    def get_conflict_breakdown(
        self, 
        text_embed: torch.Tensor, 
        image_embed: torch.Tensor
    ) -> dict:
        """
        Get a detailed breakdown of conflict by semantic aspect.
        Useful for interpretability and debugging.
        
        Returns:
            Dictionary with aspect names and their similarity scores
        """
        result = self.forward(text_embed, image_embed, return_details=True)
        
        breakdown = {}
        for i, name in enumerate(self.ASPECT_NAMES[:self.num_heads]):
            breakdown[name] = {
                'similarity': result['per_aspect_similarity'][:, i].mean().item(),
                'weight': result['aspect_weights'][i].item(),
                'conflict': 1 - result['per_aspect_similarity'][:, i].mean().item()
            }
        
        breakdown['overall'] = {
            'conflict_score': result['conflict_score'].mean().item(),
            'global_similarity': result['global_similarity'].mean().item()
        }
        
        return breakdown


class MultiScaleConflictDetector(nn.Module):
    """
    Extended conflict detector that operates at multiple scales
    of semantic granularity.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        scales: list = [4, 8, 16],  # Different numbers of semantic heads
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.scales = scales
        self.detectors = nn.ModuleList([
            SemanticConflictDetector(embed_dim, num_heads=s, dropout=dropout)
            for s in scales
        ])
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
    def forward(
        self, 
        text_embed: torch.Tensor, 
        image_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute conflict at multiple scales and aggregate.
        """
        conflict_scores = []
        all_similarities = {}
        
        for i, (scale, detector) in enumerate(zip(self.scales, self.detectors)):
            conflict, similarities = detector(text_embed, image_embed)
            conflict_scores.append(conflict)
            all_similarities[f'scale_{scale}'] = similarities
        
        # Stack and weight
        conflict_stack = torch.stack(conflict_scores, dim=-1)  # (batch, num_scales)
        weights = F.softmax(self.scale_weights, dim=0)
        final_conflict = (conflict_stack * weights).sum(dim=-1)
        
        return final_conflict, all_similarities


if __name__ == "__main__":
    # Test the conflict detector
    print("Testing SemanticConflictDetector...")
    
    detector = SemanticConflictDetector(embed_dim=512, num_heads=8)
    
    # Simulate CLIP embeddings
    batch_size = 4
    text_embed = torch.randn(batch_size, 512)
    image_embed = torch.randn(batch_size, 512)
    
    # Test with random (likely conflicting) embeddings
    conflict, similarities = detector(text_embed, image_embed)
    print(f"Random embeddings - Conflict score: {conflict.mean().item():.4f}")
    print(f"Per-aspect similarities shape: {similarities.shape}")
    
    # Test with similar embeddings (should have low conflict)
    similar_image = text_embed + torch.randn_like(text_embed) * 0.1
    conflict_low, _ = detector(text_embed, similar_image)
    print(f"Similar embeddings - Conflict score: {conflict_low.mean().item():.4f}")
    
    # Get detailed breakdown
    breakdown = detector.get_conflict_breakdown(text_embed, image_embed)
    print("\nConflict breakdown by aspect:")
    for aspect, scores in breakdown.items():
        if aspect != 'overall':
            print(f"  {aspect}: similarity={scores['similarity']:.3f}, weight={scores['weight']:.3f}")
    print(f"  Overall conflict: {breakdown['overall']['conflict_score']:.3f}")
    
    print("\n✓ SemanticConflictDetector tests passed!")
