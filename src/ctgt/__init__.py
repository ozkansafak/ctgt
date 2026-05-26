"""CTGT — LLM hallucination detection.

Three methods, ordered from baseline to production:

  kuhn_2024/    Farquhar / Kuhn (Nature 2024, 1561 citations)
                Samples M outputs → NLI clustering → entropy over meanings.
                Black-box: only logprobs needed. M=10 LLM calls per query.

  kossen_2024/  Kossen et al. SEPs (2024, 161 citations)
                Linear probe on frozen last-token hidden states.
                White-box: reads hidden states. O(d) per query at inference.

  inside_2024/  Chen et al. INSIDE (ICLR 2024, 356 citations) — stub
                EigenScore on mid-layer covariance + forward-hook clipping.
                Not yet implemented; see inside_2024/inside.py.
"""

# ── Baseline: Kuhn / Farquhar 2024 ──────────────────────────────────────────
from .kuhn_2024.detection import DetectionResult, SemanticEntropyDetector
from .kuhn_2024.entailment import EntailmentClustering
from .kuhn_2024.scoring import predictive_entropy, semantic_entropy

# ── Shared infrastructure ────────────────────────────────────────────────────
from .sampling import LLMSampler, Sample

# ── Production: Kossen 2024 ──────────────────────────────────────────────────
from .kossen_2024.probe import SEProbe

# ── Future: Chen et al. INSIDE 2024 (stub, not implemented) ─────────────────
# from .inside_2024.inside import INSIDEDetector

__all__ = [
    # Kuhn baseline
    "SemanticEntropyDetector",
    "DetectionResult",
    "EntailmentClustering",
    "semantic_entropy",
    "predictive_entropy",
    # Shared
    "LLMSampler",
    "Sample",
    # Kossen production
    "SEProbe",
]
