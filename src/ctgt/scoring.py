"""Entropy-based uncertainty scores for hallucination detection."""
from __future__ import annotations

import math

from .sampling import Sample


def _logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def semantic_entropy(
    samples: list[Sample],
    clusters: list[list[int]],
    length_normalize: bool = True,
) -> float:
    """
    Compute semantic entropy over the distribution of meanings.

        SE(x) = -∑_c p(c|x) log p(c|x)

    where p(c|x) ∝ ∑_{s∈c} exp(log_prob_s).

    This collapses semantically equivalent answers before computing entropy,
    so paraphrases do not inflate the uncertainty estimate (Kuhn et al., 2023).

    Args:
        samples: Model samples with their log-probabilities.
        clusters: Semantic equivalence classes (lists of indices into `samples`).
        length_normalize: Use per-token log-probs to correct for sequence length
            bias.  Recommended for longer answers (CoQA); less impactful for
            short factual answers (TriviaQA).

    Returns:
        Semantic entropy in nats.  0 = all samples share one meaning (certain).
        log(N) = all N samples have distinct meanings (maximally uncertain).
    """
    lps = [
        s.log_prob_normalized if length_normalize else s.log_prob for s in samples
    ]
    cluster_log_probs = [_logsumexp([lps[i] for i in c]) for c in clusters]
    log_z = _logsumexp(cluster_log_probs)
    normalized = [lp - log_z for lp in cluster_log_probs]
    return float(-sum(math.exp(lp) * lp for lp in normalized))


def predictive_entropy(
    samples: list[Sample],
    length_normalize: bool = True,
) -> float:
    """
    Baseline entropy computed over raw sequences (no semantic collapsing).

    Treats every distinct token sequence as a different outcome, so
    paraphrases inflate uncertainty.  Provided for comparison with SE.
    """
    lps = [
        s.log_prob_normalized if length_normalize else s.log_prob for s in samples
    ]
    log_z = _logsumexp(lps)
    normalized = [lp - log_z for lp in lps]
    return float(-sum(math.exp(lp) * lp for lp in normalized))
