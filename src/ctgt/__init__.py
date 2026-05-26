"""CTGT — LLM hallucination detection via semantic entropy."""

from .detection import DetectionResult, SemanticEntropyDetector
from .entailment import EntailmentClustering
from .sampling import LLMSampler, Sample
from .scoring import predictive_entropy, semantic_entropy

__all__ = [
    "SemanticEntropyDetector",
    "DetectionResult",
    "LLMSampler",
    "Sample",
    "EntailmentClustering",
    "semantic_entropy",
    "predictive_entropy",
]
