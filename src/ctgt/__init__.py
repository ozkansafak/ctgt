"""CTGT — LLM hallucination detection.

Three methods, ordered from baseline to production:

Baseline   Kuhn / Farquhar (2024) — Semantic Entropy
           Samples M outputs → NLI clustering → entropy over meanings.
           Black-box: only needs logprobs. M=10 LLM calls per query.

Production Kossen et al. (2024) — Semantic Entropy Probes (SEPs)
           Linear probe on frozen last-token hidden states.
           White-box: reads hidden states. O(d) per query at inference.

Future     Chen et al. INSIDE (ICLR 2024) — EigenScore + Feature Clipping
           EigenScore over mid-layer embedding covariance + activation
           clipping via forward hooks. Not yet implemented; see inside.py.
"""

# ── Baseline: Kuhn / Farquhar 2024 ──────────────────────────────────────────
from .detection import DetectionResult, SemanticEntropyDetector
from .entailment import EntailmentClustering
from .sampling import LLMSampler, Sample
from .scoring import predictive_entropy, semantic_entropy

# ── Production: Kossen 2024 ──────────────────────────────────────────────────
from .probe import SEProbe

# ── Future: Chen et al. INSIDE 2024 (stub, not implemented) ─────────────────
# from .inside import INSIDEDetector

__all__ = [
    # Kuhn baseline
    "SemanticEntropyDetector",
    "DetectionResult",
    "LLMSampler",
    "Sample",
    "EntailmentClustering",
    "semantic_entropy",
    "predictive_entropy",
    # Kossen production
    "SEProbe",
]
