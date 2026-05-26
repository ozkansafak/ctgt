"""NLI-based semantic equivalence clustering for hallucination detection."""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EntailmentClustering:
    """
    Groups a list of answers into semantic equivalence classes via
    bidirectional NLI entailment, following Kuhn et al. (2023).

    Two answers are semantically equivalent iff each entails the other
    in the context of the question.  We compare each new answer against
    one representative per existing cluster (greedy, O(M * C) comparisons
    instead of O(M^2) in the worst case, exploiting transitivity).
    """

    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-large"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Robustly locate the "entailment" output dimension
        id2label: dict[int, str] = self.model.config.id2label
        matches = [i for i, lbl in id2label.items() if "entail" in lbl.lower()]
        if not matches:
            raise ValueError(
                f"Cannot find 'entailment' label in {model_name}. "
                f"id2label={id2label}"
            )
        self._entailment_idx: int = matches[0]

    @torch.no_grad()
    def _entails(self, premise: str, hypothesis: str) -> bool:
        """Returns True if premise logically implies hypothesis."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)
        logits = self.model(**inputs).logits
        return logits.argmax(dim=-1).item() == self._entailment_idx

    def _equivalent(self, s1: str, s2: str, context: str) -> bool:
        """Bidirectional entailment: s1 ⟺ s2 given the question context."""
        if context:
            p1 = f"{context} {s1}"
            p2 = f"{context} {s2}"
        else:
            p1, p2 = s1, s2
        return self._entails(p1, p2) and self._entails(p2, p1)

    def cluster(self, answers: list[str], context: str = "") -> list[list[int]]:
        """
        Partition answer indices into semantic equivalence classes.

        Returns a list of clusters; each cluster is a list of indices into
        `answers`.  The first element of each cluster is the representative
        used for future comparisons.
        """
        clusters: list[list[int]] = []

        for i, answer in enumerate(answers):
            placed = False
            for cluster in clusters:
                rep = answers[cluster[0]]
                if self._equivalent(rep, answer, context):
                    cluster.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])

        return clusters
