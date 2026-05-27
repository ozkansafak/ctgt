"""Semantic Entropy Probe (Kossen et al., 2024) — approximates SE from hidden states.

Instead of M=10 costly LLM generations, a linear probe on the last-input-token
hidden state at a single forward pass approximates semantic entropy at O(d) cost.

Training workflow
-----------------
1. Collect SE scores via the full pipeline (modal run collect_sep_training_data)
2. Fit probe: ``SEProbe().fit(hidden_states_by_layer, se_scores)``
3. Save: ``probe.save("outputs/sep_probe.pkl")``

Inference workflow
------------------
1. Load: ``probe = SEProbe.load("outputs/sep_probe.pkl")``
2. Extract hidden state from LLM: ``hs = sampler.extract_hidden_states(question)``
3. Score: ``probe.predict_proba(np.array(hs[str(probe.best_layer)]))``
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def _otsu_threshold(scores: np.ndarray) -> float:
    """Pick the threshold that minimizes total within-class variance (Otsu 1979)."""
    best_t, best_var = float(scores.mean()), float("inf")
    for t in np.linspace(scores.min(), scores.max(), 200):
        lo = scores[scores <= t]
        hi = scores[scores > t]
        if len(lo) == 0 or len(hi) == 0:
            continue
        var = len(lo) * float(lo.var()) + len(hi) * float(hi.var())
        if var < best_var:
            best_var, best_t = var, float(t)
    return best_t


class SEProbe:
    """Linear probe that predicts high semantic entropy from a single LLM forward pass.

    Attributes
    ----------
    best_layer : int
        Layer index with highest train-AUROC; used for inference.
    threshold : float
        Otsu threshold that converts continuous SE scores to binary labels.
    clf : LogisticRegression
        Fitted classifier on the best layer's hidden states.
    layer_aurocs : dict[int, float]
        Train AUROC for each layer examined during grid search.
    """

    def __init__(self) -> None:
        self.best_layer: int | None = None
        self.threshold: float | None = None
        self.clf: LogisticRegression | None = None
        self.layer_aurocs: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        hidden_states: dict[int, list[list[float]]],
        se_scores: list[float],
    ) -> dict[int, float]:
        """Grid-search over layers; return layer → train-AUROC mapping.

        Args:
            hidden_states: ``{layer_idx: matrix of shape (n_questions, hidden_dim)}``.
            se_scores: Ground-truth semantic entropy per question.

        Returns:
            Dict mapping each layer to its train AUROC.
        """
        arr = np.array(se_scores, dtype=np.float32)
        self.threshold = _otsu_threshold(arr)
        labels = (arr > self.threshold).astype(int)

        if len(np.unique(labels)) < 2:
            raise ValueError(
                f"All {len(labels)} questions share the same SE label "
                f"(threshold={self.threshold:.3f}). "
                "Collect more diverse data or adjust the threshold manually."
            )

        self.layer_aurocs = {}
        for layer_idx, X_list in hidden_states.items():
            X = np.array(X_list, dtype=np.float32)
            X = StandardScaler().fit_transform(X)
            clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
            clf.fit(X, labels)
            proba = clf.predict_proba(X)[:, 1]
            self.layer_aurocs[layer_idx] = float(roc_auc_score(labels, proba))

        self.best_layer = max(self.layer_aurocs, key=self.layer_aurocs.get)
        X_best = np.array(hidden_states[self.best_layer], dtype=np.float32)
        self.scaler = StandardScaler().fit(X_best)
        self.clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
        self.clf.fit(self.scaler.transform(X_best), labels)

        return self.layer_aurocs

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, hidden_state: np.ndarray) -> float:
        """Return P(high SE) from a single hidden state vector.

        Args:
            hidden_state: 1-D array of shape ``(hidden_dim,)``.

        Returns:
            Float in [0, 1]; values above 0.5 indicate high uncertainty.
        """
        if self.clf is None:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        h = self.scaler.transform(hidden_state.reshape(1, -1))
        return float(self.clf.predict_proba(h)[0, 1])

    def is_uncertain(self, hidden_state: np.ndarray, cutoff: float = 0.5) -> bool:
        return self.predict_proba(hidden_state) > cutoff

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> "SEProbe":
        obj = pickle.loads(Path(path).read_bytes())
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected SEProbe")
        return obj
