"""
Modal deployment for the semantic-entropy hallucination detector.

Commands
--------
Single question (interactive):
    modal run modal_app.py::app.analyze --question "Who painted the Mona Lisa?"

TriviaQA benchmark (parallelised across Modal workers):
    modal run modal_app.py::app.benchmark --n-questions 200 --n-samples 10

Persistent web endpoint:
    modal deploy modal_app.py

Model defaults are intentionally small for cheap prototyping.
Swap LLM_MODEL to a larger model (e.g. Qwen/Qwen2.5-7B-Instruct) once the
pipeline is validated.
"""
from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Model selection  — small by default for cheap prototyping
# ---------------------------------------------------------------------------
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # ~3 GB on disk; fits on T4
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"


# ---------------------------------------------------------------------------
# Image: pre-download weights at build time so cold starts are fast
# ---------------------------------------------------------------------------
def _download_weights() -> None:
    """
    Cache model weights in the image layer at build time.

    Uses snapshot_download (pure HTTP, no torch import) so this step works
    in the CPU-only build environment before torch is available.
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading {LLM_MODEL} …")
    snapshot_download(LLM_MODEL)

    print(f"Downloading {NLI_MODEL} …")
    snapshot_download(NLI_MODEL)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers>=4.43.0",
        "accelerate>=0.31.0",
        "sentencepiece",
        "protobuf",
        "rouge-score>=0.1.2",
        "datasets>=2.20.0",
        "scikit-learn>=1.5.0",
        "tqdm>=4.66.0",
    )
    .run_function(_download_weights)
    # Add local package to the image — replaces modal.Mount (removed in Modal 1.x)
    .add_local_python_source("ctgt")
)

app = modal.App("hallucination-detector", image=image)


# ---------------------------------------------------------------------------
# Interactive single-question service
# ---------------------------------------------------------------------------
@app.cls(gpu="T4", timeout=300, scaledown_window=300)
class DetectorService:
    """
    GPU service for interactive single-question analysis.

    Models are loaded once when the container starts (@modal.enter) and reused
    across all requests, so per-question latency is just inference time.
    """

    @modal.enter()
    def load(self) -> None:
        from ctgt.detection import SemanticEntropyDetector
        from ctgt.entailment import EntailmentClustering
        from ctgt.sampling import LLMSampler

        sampler = LLMSampler(model_name=LLM_MODEL)
        clusterer = EntailmentClustering(model_name=NLI_MODEL)
        self.detector = SemanticEntropyDetector(sampler=sampler, clusterer=clusterer)

    @modal.method()
    def analyze(
        self,
        question: str,
        n_samples: int = 10,
        temperature: float = 0.5,
        max_new_tokens: int = 128,
    ) -> dict:
        result = self.detector.analyze(
            question,
            n_samples=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return {
            "question": question,
            "semantic_entropy": result.semantic_entropy,
            "predictive_entropy": result.predictive_entropy,
            "n_clusters": result.n_clusters,
            "is_uncertain": result.is_uncertain,
            "most_common_answer": result.most_common_answer,
            "answers": result.answers,
        }


# ---------------------------------------------------------------------------
# Benchmark worker — module-level cache so each container loads models once
# ---------------------------------------------------------------------------
_detector = None


def _get_detector():
    """Lazy singleton: models load once per container and stay warm."""
    global _detector
    if _detector is None:
        from ctgt.detection import SemanticEntropyDetector
        from ctgt.entailment import EntailmentClustering
        from ctgt.sampling import LLMSampler

        sampler = LLMSampler(model_name=LLM_MODEL)
        clusterer = EntailmentClustering(model_name=NLI_MODEL)
        _detector = SemanticEntropyDetector(sampler=sampler, clusterer=clusterer)
    return _detector


@app.function(gpu="T4", timeout=600)
def score_question(question: str, n_samples: int = 10) -> dict:
    """
    Score one question.  Modal keeps containers warm between .map() calls so
    _get_detector() only pays the model-load cost on the first question per
    container, not on every call.
    """
    detector = _get_detector()
    result = detector.analyze(question, n_samples=n_samples)
    return {
        "question": question,
        "semantic_entropy": result.semantic_entropy,
        "predictive_entropy": result.predictive_entropy,
        "n_clusters": result.n_clusters,
        "most_common_answer": result.most_common_answer,
    }


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def analyze(question: str = "What is the capital of France?", n_samples: int = 10):
    """Analyze a single question and pretty-print results."""
    service = DetectorService()
    r = service.analyze.remote(question, n_samples=n_samples)

    print(f"\nQuestion : {r['question']}")
    print(f"SE       : {r['semantic_entropy']:.3f}  (PE: {r['predictive_entropy']:.3f})")
    print(f"Clusters : {r['n_clusters']}")
    print(f"Uncertain: {r['is_uncertain']}")
    print(f"Best ans : {r['most_common_answer']}")
    print("\nAll samples:")
    for i, ans in enumerate(r["answers"]):
        print(f"  [{i+1:2d}] {ans}")


@app.local_entrypoint()
def benchmark(n_questions: int = 200, n_samples: int = 10):
    """
    Evaluate SE vs PE on TriviaQA (closed-book).

    Questions are scored in parallel across Modal T4 workers; Modal autoscales
    and keeps containers warm, so model-load cost is amortised across questions.
    AUROC is computed locally once all results arrive.
    """
    import json
    from pathlib import Path

    from datasets import load_dataset
    from rouge_score import rouge_scorer as rs
    from sklearn.metrics import roc_auc_score

    print(f"Loading TriviaQA validation[:{n_questions}] …")
    dataset = load_dataset(
        "mandarjoshi/trivia_qa", "rc.nocontext", split=f"validation[:{n_questions}]"
    )
    items = list(dataset)
    questions = [item["question"] for item in items]
    aliases_list = [item["answer"]["normalized_aliases"] for item in items]

    print(f"Dispatching {len(questions)} questions to Modal (n_samples={n_samples}) …")
    raw = list(
        score_question.map(
            questions,
            kwargs={"n_samples": n_samples},
            order_outputs=True,
        )
    )

    # Score correctness locally (cheap, no GPU needed)
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
    rows = []
    for r, aliases in zip(raw, aliases_list):
        correct = any(
            scorer.score(alias, r["most_common_answer"])["rougeL"].fmeasure > 0.3
            for alias in aliases
        )
        rows.append({**r, "is_correct": correct})

    labels_wrong = [1 - int(r["is_correct"]) for r in rows]
    se_auroc = roc_auc_score(labels_wrong, [r["semantic_entropy"] for r in rows])
    pe_auroc = roc_auc_score(labels_wrong, [r["predictive_entropy"] for r in rows])
    accuracy = 1 - sum(labels_wrong) / len(labels_wrong)

    print(f"\n{'='*56}")
    print(f" TriviaQA closed-book  n={len(rows)}  accuracy={accuracy:.1%}")
    print(f"   Semantic entropy AUROC   : {se_auroc:.3f}")
    print(f"   Predictive entropy AUROC : {pe_auroc:.3f}")
    print(f"   SE improvement           : {se_auroc - pe_auroc:+.3f}")
    print(f"{'='*56}")

    out = Path("benchmark_results.json")
    out.write_text(
        json.dumps(
            {
                "llm_model": LLM_MODEL,
                "nli_model": NLI_MODEL,
                "n_questions": len(rows),
                "n_samples": n_samples,
                "accuracy": accuracy,
                "se_auroc": se_auroc,
                "pe_auroc": pe_auroc,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"Full results → {out}")
