"""Top-level hallucination detector combining sampling, clustering, and scoring."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .entailment import EntailmentClustering
from .sampling import LLMSampler, Sample
from .scoring import predictive_entropy, semantic_entropy


@dataclass
class DetectionResult:
    """Full output of a single semantic-entropy analysis."""

    semantic_entropy: float
    predictive_entropy: float
    n_clusters: int
    samples: list[Sample]
    clusters: list[list[int]]
    is_uncertain: bool
    time_llm_s: float = 0.0   # wall time for LLM sampling
    time_nli_s: float = 0.0   # wall time for DeBERTa clustering

    @property
    def answers(self) -> list[str]:
        return [s.text for s in self.samples]

    @property
    def most_common_answer(self) -> str:
        """Return the answer whose semantic cluster has the highest total probability."""
        if not self.clusters or not self.samples:
            return ""
        lps = [s.log_prob_normalized for s in self.samples]

        def _logsumexp(vals: list[float]) -> float:
            m = max(vals)
            return m + math.log(sum(math.exp(v - m) for v in vals))

        best_cluster = max(self.clusters, key=lambda c: _logsumexp([lps[i] for i in c]))
        return self.samples[best_cluster[0]].text


class SemanticEntropyDetector:
    """
    Hallucination detector based on semantic entropy (Kuhn et al., ICLR 2023).

    The detector samples N completions from the LLM, clusters them by
    meaning using bidirectional NLI entailment, and computes entropy over
    the resulting meaning distribution.  High semantic entropy signals that
    the model is uncertain *about which fact to state*, which is a strong
    indicator of hallucination or unreliable generation.

    Usage:
        detector = SemanticEntropyDetector()
        result = detector.analyze("What is the capital of Australia?")
        print(result.semantic_entropy, result.is_uncertain)
    """

    # log(2) ≈ 0.69: entropy of a 50/50 split between two distinct meanings
    DEFAULT_THRESHOLD = math.log(2)

    def __init__(
        self,
        sampler: LLMSampler | None = None,
        clusterer: EntailmentClustering | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.sampler = sampler or LLMSampler()
        self.clusterer = clusterer or EntailmentClustering()
        self.threshold = threshold

    def analyze(
        self,
        question: str,
        n_samples: int = 10,
        temperature: float = 0.5,
        max_new_tokens: int = 128,
        length_normalize: bool = True,
    ) -> DetectionResult:
        """
        Run the full semantic-entropy pipeline for one question.

        Args:
            question: The prompt / question to evaluate.
            n_samples: Number of stochastic completions to draw (10 is the
                paper's recommended minimum; more → better estimates).
            temperature: Sampling temperature.  0.5 balances accuracy and
                diversity for most QA tasks (Kuhn et al., 2023, Fig. 3b).
            max_new_tokens: Maximum completion length per sample.
            length_normalize: Divide log-probs by sequence length before
                computing entropy (recommended for longer-answer tasks).

        Returns:
            DetectionResult with entropy scores, cluster assignments, and a
            binary `is_uncertain` flag based on `self.threshold`.
        """
        t0 = time.perf_counter()
        samples = self.sampler.sample(
            question,
            n=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        t1 = time.perf_counter()
        clusters = self.clusterer.cluster(
            [s.text for s in samples], context=question
        )
        t2 = time.perf_counter()

        se = semantic_entropy(samples, clusters, length_normalize=length_normalize)
        pe = predictive_entropy(samples, length_normalize=length_normalize)

        return DetectionResult(
            semantic_entropy=se,
            predictive_entropy=pe,
            n_clusters=len(clusters),
            samples=samples,
            clusters=clusters,
            is_uncertain=se > self.threshold,
            time_llm_s=t1 - t0,
            time_nli_s=t2 - t1,
        )

    # ------------------------------------------------------------------
    # Backward-compatible interface (kept for callers of the original stub)
    # ------------------------------------------------------------------

    def detect(self, prompt: str, output: str = "") -> list[str]:
        """Return a list of hallucination signal strings for `prompt`."""
        result = self.analyze(prompt)
        signals: list[str] = []
        if result.is_uncertain:
            signals.append("high_semantic_entropy")
        if result.n_clusters >= len(result.samples) // 2:
            signals.append("many_distinct_meanings")
        if result.predictive_entropy > self.threshold:
            signals.append("high_predictive_entropy")
        return signals

    def score(self, prompt: str, output: str = "") -> float:
        """Return the semantic entropy for `prompt` (0 = certain, higher = uncertain)."""
        return self.analyze(prompt).semantic_entropy
