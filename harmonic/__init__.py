"""
HARMONIC: Hierarchical Attention-based Reconciliation of Multimodal ONtologies for Image Creation
"""

from .conflict_detector import SemanticConflictDetector
from .fusion_module import CrossModalAttentionFusion
from .scheduler import TemporalGuidanceScheduler
from .harmonic import HARMONIC

__version__ = "0.1.0"
__all__ = [
    "SemanticConflictDetector",
    "CrossModalAttentionFusion", 
    "TemporalGuidanceScheduler",
    "HARMONIC"
]
