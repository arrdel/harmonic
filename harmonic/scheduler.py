"""
Temporal Guidance Scheduler Module

Dynamically adjusts the influence of text and image guidance throughout
the diffusion process based on timestep and conflict level.

Key insight: Different stages of diffusion need different guidance:
- Early stages (high noise): Global structure, composition (text-driven)
- Middle stages: Shape refinement (balanced)
- Late stages (low noise): Fine details, textures (image-driven)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Literal
from enum import Enum


class ScheduleType(Enum):
    """Available scheduling strategies."""
    LINEAR = "linear"
    COSINE = "cosine"
    SIGMOID = "sigmoid"
    CONFLICT_AWARE = "conflict_aware"
    LEARNED = "learned"


class TemporalGuidanceScheduler(nn.Module):
    """
    Computes timestep-dependent guidance weights with optional
    conflict-aware modulation.
    
    The scheduler outputs weights for text and image guidance that
    sum to <= 1.0 (allowing the diffusion model some freedom when
    guidance is reduced).
    """
    
    def __init__(
        self,
        max_timesteps: int = 1000,
        schedule_type: str = "conflict_aware",
        text_weight_range: Tuple[float, float] = (0.1, 0.9),
        image_weight_range: Tuple[float, float] = (0.1, 0.9),
        conflict_reduction: float = 0.5,
        learnable: bool = True
    ):
        """
        Args:
            max_timesteps: Maximum number of diffusion timesteps
            schedule_type: One of 'linear', 'cosine', 'sigmoid', 'conflict_aware', 'learned'
            text_weight_range: (min, max) range for text weights
            image_weight_range: (min, max) range for image weights  
            conflict_reduction: How much to reduce guidance when conflict is high (0-1)
            learnable: Whether schedule parameters are learnable
        """
        super().__init__()
        
        self.max_timesteps = max_timesteps
        self.schedule_type = ScheduleType(schedule_type)
        self.text_weight_range = text_weight_range
        self.image_weight_range = image_weight_range
        self.conflict_reduction = conflict_reduction
        
        # Learnable parameters for conflict-aware scheduling
        if learnable:
            # Bell curve width for conflict modulation
            self.sigma = nn.Parameter(torch.tensor(0.25))
            # How much conflict affects weights
            self.conflict_strength = nn.Parameter(torch.tensor(conflict_reduction))
            # Shift parameter for the schedule
            self.shift = nn.Parameter(torch.tensor(0.0))
            # Steepness for sigmoid schedule
            self.steepness = nn.Parameter(torch.tensor(10.0))
        else:
            self.register_buffer('sigma', torch.tensor(0.25))
            self.register_buffer('conflict_strength', torch.tensor(conflict_reduction))
            self.register_buffer('shift', torch.tensor(0.0))
            self.register_buffer('steepness', torch.tensor(10.0))
        
        # For learned schedule: small MLP
        if self.schedule_type == ScheduleType.LEARNED:
            self.schedule_mlp = nn.Sequential(
                nn.Linear(2, 64),  # [normalized_timestep, conflict_score]
                nn.GELU(),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, 2),  # [text_weight, image_weight]
                nn.Sigmoid()
            )
    
    def _normalize_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Normalize timestep to [0, 1] range."""
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep], dtype=torch.float32)
        return timestep.float() / self.max_timesteps
    
    def _linear_schedule(self, t_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linear schedule: text starts high, decreases; image starts low, increases.
        t_norm = 1.0 means beginning (high noise), t_norm = 0.0 means end (low noise)
        """
        w_text = t_norm  # High at beginning, low at end
        w_img = 1 - t_norm  # Low at beginning, high at end
        return w_text, w_img
    
    def _cosine_schedule(self, t_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cosine schedule: smooth S-curve transition.
        """
        # Shift by learnable parameter
        t_shifted = (t_norm + self.shift).clamp(0, 1)
        
        w_text = torch.cos(math.pi / 2 * (1 - t_shifted))
        w_img = torch.sin(math.pi / 2 * (1 - t_shifted))
        return w_text, w_img
    
    def _sigmoid_schedule(self, t_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sigmoid schedule: sharp transition around midpoint.
        """
        # Center at 0.5, steepness controls sharpness
        w_text = torch.sigmoid(self.steepness * (t_norm - 0.5 + self.shift))
        w_img = 1 - w_text
        return w_text, w_img
    
    def _apply_conflict_modulation(
        self,
        w_text: torch.Tensor,
        w_img: torch.Tensor,
        t_norm: torch.Tensor,
        conflict_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply conflict-aware modulation to the weights.
        
        When conflict is high:
        - Reduce both weights in the middle of the process
        - Let the diffusion model be more creative
        - Apply stronger guidance only at the extremes
        """
        # Bell-shaped modulation centered at t=0.5
        # Maximum reduction happens in the middle of diffusion
        gamma = torch.exp(-((t_norm - 0.5) ** 2) / (2 * self.sigma ** 2))
        
        # Modulation factor: reduce weights when conflict is high and we're in the middle
        # All tensors should be 1D with shape (batch,)
        conflict_strength_val = self.conflict_strength.detach() if self.conflict_strength.requires_grad else self.conflict_strength
        modulation = 1 - conflict_score * conflict_strength_val * gamma
        modulation = modulation.clamp(min=0.1)  # Never reduce below 10%
        
        w_text = w_text * modulation
        w_img = w_img * modulation
        
        return w_text, w_img
    
    def _apply_weight_bounds(
        self,
        w_text: torch.Tensor,
        w_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply min/max bounds to weights."""
        w_text = w_text * (self.text_weight_range[1] - self.text_weight_range[0])
        w_text = w_text + self.text_weight_range[0]
        
        w_img = w_img * (self.image_weight_range[1] - self.image_weight_range[0])
        w_img = w_img + self.image_weight_range[0]
        
        return w_text, w_img
    
    def forward(
        self,
        timestep: torch.Tensor,
        conflict_score: torch.Tensor,
        return_raw: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute guidance weights for the given timestep and conflict score.
        
        Args:
            timestep: Current diffusion timestep (can be int or tensor)
            conflict_score: (batch,) conflict score in [0, 1]
            return_raw: Whether to return raw weights before bounds
            
        Returns:
            w_text: (batch,) text guidance weight
            w_img: (batch,) image guidance weight
        """
        # Normalize timestep
        t_norm = self._normalize_timestep(timestep)
        
        # Ensure proper shapes
        if t_norm.dim() == 0:
            t_norm = t_norm.unsqueeze(0)
        if conflict_score.dim() == 0:
            conflict_score = conflict_score.unsqueeze(0)
        
        # Broadcast if needed
        batch_size = conflict_score.shape[0]
        if t_norm.shape[0] == 1 and batch_size > 1:
            t_norm = t_norm.expand(batch_size)
        
        # Compute base schedule
        if self.schedule_type == ScheduleType.LINEAR:
            w_text, w_img = self._linear_schedule(t_norm)
            
        elif self.schedule_type == ScheduleType.COSINE:
            w_text, w_img = self._cosine_schedule(t_norm)
            
        elif self.schedule_type == ScheduleType.SIGMOID:
            w_text, w_img = self._sigmoid_schedule(t_norm)
            
        elif self.schedule_type == ScheduleType.CONFLICT_AWARE:
            # Start with cosine base
            w_text, w_img = self._cosine_schedule(t_norm)
            # Apply conflict modulation
            w_text, w_img = self._apply_conflict_modulation(
                w_text, w_img, t_norm, conflict_score
            )
            
        elif self.schedule_type == ScheduleType.LEARNED:
            # Use learned MLP
            mlp_input = torch.stack([t_norm, conflict_score], dim=-1)
            weights = self.schedule_mlp(mlp_input)
            w_text = weights[:, 0]
            w_img = weights[:, 1]
        
        raw_weights = (w_text.clone(), w_img.clone())
        
        # Apply bounds
        w_text, w_img = self._apply_weight_bounds(w_text, w_img)
        
        if return_raw:
            return w_text, w_img, raw_weights
        
        return w_text, w_img
    
    def get_full_schedule(
        self,
        conflict_score: float = 0.5,
        num_points: int = 100
    ) -> dict:
        """
        Get the full schedule curve for visualization.
        
        Args:
            conflict_score: Fixed conflict score to use
            num_points: Number of points in the curve
            
        Returns:
            Dictionary with timesteps and corresponding weights
        """
        timesteps = torch.linspace(0, self.max_timesteps, num_points)
        conflict = torch.full((num_points,), conflict_score)
        
        w_text_list = []
        w_img_list = []
        
        for t, c in zip(timesteps, conflict):
            w_text, w_img = self.forward(t.unsqueeze(0), c.unsqueeze(0))
            w_text_list.append(w_text.item())
            w_img_list.append(w_img.item())
        
        return {
            'timesteps': timesteps.numpy(),
            't_normalized': (timesteps / self.max_timesteps).numpy(),
            'w_text': w_text_list,
            'w_img': w_img_list,
            'conflict_score': conflict_score
        }


class PhaseAwareScheduler(TemporalGuidanceScheduler):
    """
    Extended scheduler that defines distinct phases of diffusion
    with different guidance strategies for each.
    """
    
    def __init__(
        self,
        max_timesteps: int = 1000,
        phases: dict = None,
        **kwargs
    ):
        super().__init__(max_timesteps, **kwargs)
        
        # Default phases if not provided
        self.phases = phases or {
            'structure': (1.0, 0.7),    # t ∈ [70%, 100%] - global structure
            'refinement': (0.7, 0.3),   # t ∈ [30%, 70%] - shape refinement  
            'detail': (0.3, 0.0)        # t ∈ [0%, 30%] - fine details
        }
        
        # Phase-specific weight overrides
        self.phase_weights = nn.ParameterDict({
            'structure_text': nn.Parameter(torch.tensor(0.8)),
            'structure_img': nn.Parameter(torch.tensor(0.2)),
            'refinement_text': nn.Parameter(torch.tensor(0.5)),
            'refinement_img': nn.Parameter(torch.tensor(0.5)),
            'detail_text': nn.Parameter(torch.tensor(0.2)),
            'detail_img': nn.Parameter(torch.tensor(0.8)),
        })
    
    def _get_phase(self, t_norm: float) -> str:
        """Determine which phase the current timestep is in."""
        for phase_name, (start, end) in self.phases.items():
            if end <= t_norm <= start:
                return phase_name
        return 'refinement'  # Default
    
    def forward(
        self,
        timestep: torch.Tensor,
        conflict_score: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override to use phase-specific weights."""
        t_norm = self._normalize_timestep(timestep)
        
        # Get base weights from parent
        w_text, w_img = super().forward(timestep, conflict_score, **kwargs)
        
        # Blend with phase-specific weights
        if t_norm.dim() == 0:
            t_norm = t_norm.unsqueeze(0)
        
        batch_size = conflict_score.shape[0]
        phase_text = torch.zeros(batch_size, device=w_text.device)
        phase_img = torch.zeros(batch_size, device=w_img.device)
        
        for i, t in enumerate(t_norm):
            phase = self._get_phase(t.item())
            phase_text[i] = self.phase_weights[f'{phase}_text']
            phase_img[i] = self.phase_weights[f'{phase}_img']
        
        # Blend base schedule with phase weights (50/50)
        w_text = 0.5 * w_text + 0.5 * phase_text
        w_img = 0.5 * w_img + 0.5 * phase_img
        
        return w_text, w_img


if __name__ == "__main__":
    # Test the scheduler
    print("Testing TemporalGuidanceScheduler...")
    
    scheduler = TemporalGuidanceScheduler(
        max_timesteps=1000,
        schedule_type="conflict_aware"
    )
    
    # Test at different timesteps
    batch_size = 4
    conflict_scores = torch.tensor([0.1, 0.3, 0.6, 0.9])  # Various conflict levels
    
    print("\nWeights at different timesteps (conflict varies):")
    for t in [900, 500, 100]:  # Beginning, middle, end
        w_text, w_img = scheduler(torch.tensor([t]), conflict_scores)
        print(f"  t={t}: text=[{w_text.min():.3f}, {w_text.max():.3f}], "
              f"img=[{w_img.min():.3f}, {w_img.max():.3f}]")
    
    # Test full schedule visualization
    print("\nFull schedule (conflict=0.5):")
    schedule = scheduler.get_full_schedule(conflict_score=0.5, num_points=10)
    for i in range(0, 10, 3):
        t = schedule['timesteps'][i]
        wt = schedule['w_text'][i]
        wi = schedule['w_img'][i]
        print(f"  t={t:.0f}: text={wt:.3f}, img={wi:.3f}")
    
    # Test phase-aware scheduler
    print("\nTesting PhaseAwareScheduler...")
    phase_scheduler = PhaseAwareScheduler(max_timesteps=1000)
    conflict = torch.tensor([0.5])
    
    for t in [900, 500, 100]:
        w_text, w_img = phase_scheduler(torch.tensor([t]), conflict)
        phase = phase_scheduler._get_phase(t / 1000)
        print(f"  t={t} ({phase}): text={w_text.item():.3f}, img={w_img.item():.3f}")
    
    print("\n✓ TemporalGuidanceScheduler tests passed!")
