"""
Cross-Modal Attention Fusion Module

Implements bidirectional cross-attention between text and image embeddings
to extract complementary information from both modalities instead of
discarding one when conflicts arise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class CrossModalAttention(nn.Module):
    """
    Single-direction cross-modal attention.
    Query from one modality attends to key/value from another.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch, embed_dim) - the modality asking the question
            key: (batch, embed_dim) - the modality being queried
            value: (batch, embed_dim) - the information to extract
            
        Returns:
            output: (batch, embed_dim) attended output
            attention: (batch, num_heads, 1, 1) attention weights (optional)
        """
        batch_size = query.shape[0]
        
        # Project and reshape for multi-head attention
        # Shape: (batch, num_heads, 1, head_dim)
        Q = self.q_proj(query).view(batch_size, self.num_heads, 1, self.head_dim)
        K = self.k_proj(key).view(batch_size, self.num_heads, 1, self.head_dim)
        V = self.v_proj(value).view(batch_size, self.num_heads, 1, self.head_dim)
        
        # Scaled dot-product attention
        # For single token, attention is essentially a weighted combination
        attn_weights = (Q @ K.transpose(-2, -1)) * self.scale  # (batch, heads, 1, 1)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = attn_weights @ V  # (batch, heads, 1, head_dim)
        output = output.view(batch_size, self.embed_dim)  # (batch, embed_dim)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion module that extracts
    complementary information from both text and image modalities.
    
    Key insight: Instead of choosing one modality, we extract the
    INTERSECTION of compatible semantic concepts from BOTH modalities.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            dropout: Dropout rate
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        # Text-to-Image attention: What in the image matches the text?
        self.text_to_image_attn = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Image-to-Text attention: What in the text matches the image?
        self.image_to_text_attn = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norms for each attention layer
        self.text_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        self.image_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        # Gated fusion mechanism
        # Takes cross-attended features + conflict score to produce fusion gate
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Final fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(
        self,
        text_embed: torch.Tensor,
        image_embed: torch.Tensor,
        conflict_score: torch.Tensor,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Fuse text and image embeddings using bidirectional cross-attention.
        
        Args:
            text_embed: (batch, embed_dim) CLIP text embedding
            image_embed: (batch, embed_dim) CLIP image embedding
            conflict_score: (batch,) conflict score in [0, 1]
            return_intermediates: Whether to return intermediate features
            
        Returns:
            fused_embed: (batch, embed_dim) harmonized embedding
        """
        batch_size = text_embed.shape[0]
        
        # Initialize with input embeddings
        text_feat = text_embed
        image_feat = image_embed
        
        # Store intermediates for analysis
        intermediates = {'t2i_attentions': [], 'i2t_attentions': []}
        
        # Apply cross-attention layers
        for i in range(self.num_layers):
            # Text-to-Image: Query with text, attend to image
            t2i_out, t2i_attn = self.text_to_image_attn[i](
                query=text_feat,
                key=image_feat,
                value=image_feat,
                return_attention=True
            )
            
            # Image-to-Text: Query with image, attend to text
            i2t_out, i2t_attn = self.image_to_text_attn[i](
                query=image_feat,
                key=text_feat,
                value=text_feat,
                return_attention=True
            )
            
            # Apply residual and norm
            if self.use_residual:
                text_feat = self.text_norms[i](text_feat + t2i_out)
                image_feat = self.image_norms[i](image_feat + i2t_out)
            else:
                text_feat = self.text_norms[i](t2i_out)
                image_feat = self.image_norms[i](i2t_out)
            
            intermediates['t2i_attentions'].append(t2i_attn)
            intermediates['i2t_attentions'].append(i2t_attn)
        
        # Gated fusion based on conflict score
        # Higher conflict -> more balanced fusion
        # Lower conflict -> favor the more informative modality
        gate_input = torch.cat([
            text_feat,
            image_feat,
            conflict_score.unsqueeze(-1)
        ], dim=-1)  # (batch, embed_dim * 2 + 1)
        
        gate = self.gate_network(gate_input)  # (batch, embed_dim)
        
        # Fuse features with learned gate
        # When conflict is high, gate learns to balance both
        # When conflict is low, gate can favor the dominant modality
        weighted_text = gate * text_feat
        weighted_image = (1 - gate) * image_feat
        
        # Concatenate and project
        concat_feat = torch.cat([weighted_text, weighted_image], dim=-1)
        fused = self.fusion_proj(concat_feat)
        
        # Final projection with residual from both inputs
        output = self.out_proj(fused)
        
        # Add skip connection from original embeddings (scaled by inverse conflict)
        # When conflict is low, we trust the original embeddings more
        skip_weight = (1 - conflict_score).unsqueeze(-1) * 0.3
        output = output + skip_weight * (text_embed + image_embed) / 2
        
        if return_intermediates:
            intermediates['text_feat'] = text_feat
            intermediates['image_feat'] = image_feat
            intermediates['gate'] = gate
            intermediates['fused'] = fused
            return output, intermediates
        
        return output


class AdaptiveFusion(nn.Module):
    """
    Simplified adaptive fusion for when full cross-attention is too expensive.
    Uses learned mixing based on conflict score.
    """
    
    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Learn mixing coefficients from embeddings and conflict
        self.mix_network = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # [text_weight, image_weight, fusion_weight]
            nn.Softmax(dim=-1)
        )
        
        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(
        self,
        text_embed: torch.Tensor,
        image_embed: torch.Tensor,
        conflict_score: torch.Tensor
    ) -> torch.Tensor:
        """
        Simple adaptive fusion based on conflict score.
        """
        # Compute fusion embedding
        fused = self.fusion_proj(torch.cat([text_embed, image_embed], dim=-1))
        
        # Compute mixing weights
        mix_input = torch.cat([
            text_embed, image_embed, conflict_score.unsqueeze(-1)
        ], dim=-1)
        weights = self.mix_network(mix_input)  # (batch, 3)
        
        # Weighted combination
        output = (
            weights[:, 0:1] * text_embed +
            weights[:, 1:2] * image_embed +
            weights[:, 2:3] * fused
        )
        
        return output


if __name__ == "__main__":
    # Test the fusion module
    print("Testing CrossModalAttentionFusion...")
    
    fusion = CrossModalAttentionFusion(embed_dim=512, num_heads=8, num_layers=2)
    
    # Simulate inputs
    batch_size = 4
    text_embed = torch.randn(batch_size, 512)
    image_embed = torch.randn(batch_size, 512)
    conflict_score = torch.rand(batch_size)  # Random conflict scores
    
    # Test forward pass
    fused = fusion(text_embed, image_embed, conflict_score)
    print(f"Fused embedding shape: {fused.shape}")
    
    # Test with intermediates
    fused, intermediates = fusion(text_embed, image_embed, conflict_score, return_intermediates=True)
    print(f"Gate values range: [{intermediates['gate'].min().item():.3f}, {intermediates['gate'].max().item():.3f}]")
    
    # Test adaptive fusion
    print("\nTesting AdaptiveFusion...")
    adaptive = AdaptiveFusion(embed_dim=512)
    output = adaptive(text_embed, image_embed, conflict_score)
    print(f"Adaptive fusion output shape: {output.shape}")
    
    print("\n✓ CrossModalAttentionFusion tests passed!")
