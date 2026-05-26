"""
Modal deployment for LLM hallucination detection.

Implements two methods:
  Kuhn / Farquhar 2024 (1561 citations) — Semantic Entropy baseline
    modal run modal_app.py::app.benchmark --n-questions 300
    modal run modal_app.py::app.benchmark --n-questions 300 --llm-model NousResearch/Meta-Llama-3.1-8B-Instruct
    modal run modal_app.py::app.analyze  --question "Who painted the Mona Lisa?"

  Kossen et al. 2024 (161 citations) — Semantic Entropy Probes
    modal run modal_app.py::app.collect_sep_training_data --n-questions 300
    modal run modal_app.py::app.collect_sep_training_data --n-questions 300 --llm-model Qwen/Qwen2.5-1.5B-Instruct
    python train_probe.py
    modal run modal_app.py::app.upload_probe
    modal run modal_app.py::app.analyze_fast --question "Who painted the Mona Lisa?"

Persistent web endpoint:
    modal deploy modal_app.py
"""
from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"   # default
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"

# All LLMs pre-baked into the image so cold starts are fast for any model.
_ALL_LLM_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",  # public mirror, no HF token needed
    "Qwen/Qwen2.5-1.5B-Instruct",
]


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

    for model in _ALL_LLM_MODELS:
        print(f"Downloading {model} …")
        snapshot_download(model)

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
@app.cls(gpu="A10G", timeout=300, scaledown_window=300)
class DetectorService:
    """
    GPU service for interactive single-question analysis.

    Models are loaded once when the container starts (@modal.enter) and reused
    across all requests, so per-question latency is just inference time.
    """

    @modal.enter()
    def load(self) -> None:
        from ctgt.kuhn_2024.detection import SemanticEntropyDetector
        from ctgt.kuhn_2024.entailment import EntailmentClustering
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
# Benchmark worker — per-model cache so each container loads models once
# ---------------------------------------------------------------------------
_detectors: dict = {}


def _get_detector(llm_model: str = LLM_MODEL):
    """Lazy cache keyed by model name: loads once per (container, model) pair."""
    if llm_model not in _detectors:
        from ctgt.kuhn_2024.detection import SemanticEntropyDetector
        from ctgt.kuhn_2024.entailment import EntailmentClustering
        from ctgt.sampling import LLMSampler

        sampler = LLMSampler(model_name=llm_model)
        clusterer = EntailmentClustering(model_name=NLI_MODEL)
        _detectors[llm_model] = SemanticEntropyDetector(sampler=sampler, clusterer=clusterer)
    return _detectors[llm_model]


@app.function(gpu="A10G", timeout=600)
def score_question(question: str, n_samples: int = 10, temperature: float = 0.5, llm_model: str = LLM_MODEL) -> dict:
    """
    Score one question.  Modal keeps containers warm between .map() calls so
    _get_detector() only pays the model-load cost on the first question per
    container, not on every call.
    """
    detector = _get_detector(llm_model)
    result = detector.analyze(question, n_samples=n_samples, temperature=temperature)
    return {
        "question": question,
        "semantic_entropy": result.semantic_entropy,
        "predictive_entropy": result.predictive_entropy,
        "n_clusters": result.n_clusters,
        "most_common_answer": result.most_common_answer,
        "time_llm_s": result.time_llm_s,
        "time_nli_s": result.time_nli_s,
    }


# ---------------------------------------------------------------------------
# SEP: data collection + fast inference
# ---------------------------------------------------------------------------
probe_volume = modal.Volume.from_name("sep-probes", create_if_missing=True)


@app.function(gpu="A10G", timeout=600)
def score_question_with_states(
    question: str,
    n_samples: int = 10,
    temperature: float = 0.5,
    n_last_layers: int = 12,
    llm_model: str = LLM_MODEL,
) -> dict:
    """Score one question (full SE pipeline) and also extract LLM hidden states.

    The hidden states are used to train a Semantic Entropy Probe (Kossen et al., 2024)
    that later approximates SE with a single forward pass and a linear probe.

    Args:
        n_last_layers: How many of the final transformer layers to return.
                       Reduces payload size; later layers are most informative.
    """
    detector = _get_detector(llm_model)
    result = detector.analyze(question, n_samples=n_samples, temperature=temperature)

    n_layers = detector.sampler.model.config.num_hidden_layers  # transformer blocks only
    layers = list(range(n_layers + 1 - n_last_layers, n_layers + 1))  # +1 for embedding offset
    hidden_states = detector.sampler.extract_hidden_states(question, layers=layers)

    return {
        "question": question,
        "semantic_entropy": result.semantic_entropy,
        "predictive_entropy": result.predictive_entropy,
        "n_clusters": result.n_clusters,
        "most_common_answer": result.most_common_answer,
        "time_llm_s": result.time_llm_s,
        "time_nli_s": result.time_nli_s,
        "hidden_states": hidden_states,
    }


@app.function(gpu="A10G", timeout=120, volumes={"/probes": probe_volume})
def analyze_fast(question: str) -> dict:
    """Hallucination detection via a single forward pass + linear probe.

    Replaces M=10 LLM generations + NLI clustering with one forward pass and a
    matrix multiply — approximately 5-10x faster at inference time.

    Requires a trained probe uploaded via ``upload_probe``.
    """
    import pickle
    from pathlib import Path

    import numpy as np

    probe_path = Path("/probes/sep_probe.pkl")
    if not probe_path.exists():
        raise FileNotFoundError(
            "No probe found at /probes/sep_probe.pkl. "
            "Run collect_sep_training_data → train_probe.py → upload_probe first."
        )

    probe = pickle.loads(probe_path.read_bytes())
    sampler = _get_detector().sampler
    hidden_states = sampler.extract_hidden_states(question, layers=[probe.best_layer])
    hs = np.array(hidden_states[str(probe.best_layer)], dtype=np.float32)

    score = probe.predict_proba(hs)
    return {
        "question": question,
        "probe_score": score,
        "is_uncertain": score > 0.5,
        "best_layer": probe.best_layer,
        "method": "sep",
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
def benchmark(n_questions: int = 200, n_samples: int = 10, temperature: float = 0.5, llm_model: str = LLM_MODEL):
    """
    Evaluate SE vs PE on TriviaQA (closed-book).

    Questions are scored in parallel across Modal T4 workers; Modal autoscales
    and keeps containers warm, so model-load cost is amortised across questions.
    AUROC is computed locally once all results arrive.

    Example:
        modal run modal_app.py::app.benchmark --n-questions 300
        modal run modal_app.py::app.benchmark --n-questions 300 --llm-model NousResearch/Meta-Llama-3.1-8B-Instruct
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

    import time as _time
    print(f"Dispatching {len(questions)} questions to Modal (model={llm_model}, n_samples={n_samples}, temperature={temperature}) …")
    _t_start = _time.perf_counter()
    raw = list(
        score_question.map(
            questions,
            kwargs={"n_samples": n_samples, "temperature": temperature, "llm_model": llm_model},
            order_outputs=True,
        )
    )
    wall_time_s = _time.perf_counter() - _t_start

    # Score correctness locally (cheap, no GPU needed)
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
    rows = []
    for r, aliases in zip(raw, aliases_list):
        correct = any(
            scorer.score(alias, r["most_common_answer"])["rougeL"].fmeasure > 0.3
            for alias in aliases
        )
        rows.append({**r, "is_correct": correct})

    import math
    labels_wrong = [1 - int(r["is_correct"]) for r in rows]
    se_scores = [r["semantic_entropy"] for r in rows]
    pe_scores = [r["predictive_entropy"] for r in rows]
    valid_se = [(lw, s) for lw, s in zip(labels_wrong, se_scores) if not math.isnan(s)]
    valid_pe = [(lw, s) for lw, s in zip(labels_wrong, pe_scores) if not math.isnan(s)]
    nan_count = sum(1 for s in se_scores if math.isnan(s))
    if nan_count:
        print(f"  Warning: {nan_count}/{len(rows)} questions had NaN entropy — excluded from AUROC")
    se_auroc = roc_auc_score(*zip(*valid_se)) if valid_se else float("nan")
    pe_auroc = roc_auc_score(*zip(*valid_pe)) if valid_pe else float("nan")
    accuracy = 1 - sum(labels_wrong) / len(labels_wrong)

    avg_llm = sum(r["time_llm_s"] for r in raw) / len(raw)
    avg_nli = sum(r["time_nli_s"] for r in raw) / len(raw)

    print(f"\n{'='*56}")
    print(f" TriviaQA closed-book  n={len(rows)}  accuracy={accuracy:.1%}")
    print(f"   Semantic entropy AUROC   : {se_auroc:.3f}")
    print(f"   Predictive entropy AUROC : {pe_auroc:.3f}")
    print(f"   SE improvement           : {se_auroc - pe_auroc:+.3f}")
    print(f"   Avg LLM sampling time    : {avg_llm:.1f}s  (M={n_samples} samples)")
    print(f"   Avg NLI clustering time  : {avg_nli:.1f}s")
    print(f"   Avg total per question   : {avg_llm + avg_nli:.1f}s")
    print(f"   Total wall time          : {wall_time_s:.0f}s ({wall_time_s/60:.1f} min)")
    print(f"{'='*56}")

    from datetime import datetime
    model_slug = llm_model.split("/")[-1].lower()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname      = f"{model_slug}_q{len(rows)}_s{n_samples}_t{temperature}_{timestamp}.json"
    Path("outputs").mkdir(exist_ok=True)
    out = Path("outputs") / fname

    payload = json.dumps(
        {
            "llm_model": llm_model,
            "nli_model": NLI_MODEL,
            "n_questions": len(rows),
            "n_samples": n_samples,
            "temperature": temperature,
            "accuracy": accuracy,
            "se_auroc": se_auroc,
            "pe_auroc": pe_auroc,
            "avg_llm_s": avg_llm,
            "avg_nli_s": avg_nli,
            "wall_time_s": wall_time_s,
            "rows": rows,
        },
        indent=2,
    )
    out.write_text(payload)
    Path("benchmark_results.json").write_text(payload)  # kept for results.py default
    print(f"Full results → {out}")


@app.local_entrypoint()
def collect_sep_training_data(
    n_questions: int = 200,
    n_samples: int = 10,
    temperature: float = 0.5,
    llm_model: str = LLM_MODEL,
):
    """Collect SE scores + LLM hidden states for SEP probe training.

    Runs the full SE pipeline on TriviaQA and saves hidden states alongside
    entropy scores.  Feed the output JSON to train_probe.py to fit the probe.

    Example:
        modal run modal_app.py::app.collect_sep_training_data --n-questions 300
        modal run modal_app.py::app.collect_sep_training_data --n-questions 300 --llm-model Qwen/Qwen2.5-1.5B-Instruct
    """
    import json
    import time as _time
    from datetime import datetime
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

    print(
        f"Dispatching {len(questions)} questions with hidden state extraction "
        f"(model={llm_model}, n_samples={n_samples}, temperature={temperature}) …"
    )
    _t0 = _time.perf_counter()
    raw = list(
        score_question_with_states.map(
            questions,
            kwargs={"n_samples": n_samples, "temperature": temperature, "llm_model": llm_model},
            order_outputs=True,
        )
    )
    wall_time_s = _time.perf_counter() - _t0

    scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
    rows = []
    for r, aliases in zip(raw, aliases_list):
        correct = any(
            scorer.score(alias, r["most_common_answer"])["rougeL"].fmeasure > 0.3
            for alias in aliases
        )
        rows.append({**r, "is_correct": correct})

    import math
    labels_wrong = [1 - int(r["is_correct"]) for r in rows]
    se_scores = [r["semantic_entropy"] for r in rows]
    pe_scores = [r["predictive_entropy"] for r in rows]
    # Filter NaNs for AUROC (some models produce NaN entropy on degenerate outputs)
    valid_se = [(lw, s) for lw, s in zip(labels_wrong, se_scores) if not math.isnan(s)]
    valid_pe = [(lw, s) for lw, s in zip(labels_wrong, pe_scores) if not math.isnan(s)]
    nan_count = sum(1 for s in se_scores if math.isnan(s))
    if nan_count:
        print(f"  Warning: {nan_count}/{len(rows)} questions had NaN entropy — excluded from AUROC")
    se_auroc = roc_auc_score(*zip(*valid_se)) if valid_se else float("nan")
    pe_auroc = roc_auc_score(*zip(*valid_pe)) if valid_pe else float("nan")
    accuracy = 1 - sum(labels_wrong) / len(labels_wrong)

    model_slug = llm_model.split("/")[-1].lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"sep_data_{model_slug}_q{n_questions}_s{n_samples}_t{temperature}_{timestamp}.json"
    Path("outputs").mkdir(exist_ok=True)
    out = Path("outputs") / fname

    payload = json.dumps(
        {
            "llm_model": llm_model,
            "nli_model": NLI_MODEL,
            "n_questions": len(rows),
            "n_samples": n_samples,
            "temperature": temperature,
            "accuracy": accuracy,
            "se_auroc": se_auroc,
            "pe_auroc": pe_auroc,
            "wall_time_s": wall_time_s,
            "rows": rows,
        },
        indent=2,
    )
    out.write_text(payload)
    print(f"\nSE AUROC: {se_auroc:.3f} | PE AUROC: {pe_auroc:.3f} | Accuracy: {accuracy:.1%}")
    print(f"Wall time: {wall_time_s:.0f}s ({wall_time_s / 60:.1f} min)")
    print(f"SEP training data → {out}")
    print("Next: python train_probe.py --input", out)


@app.local_entrypoint()
def upload_probe(path: str = "outputs/sep_probe.pkl"):
    """Upload a trained SEP probe to Modal volume for fast inference.

    Example:
        modal run modal_app.py::app.upload_probe --path outputs/sep_probe.pkl
    """
    from pathlib import Path

    data = Path(path).read_bytes()
    with probe_volume.batch_upload() as batch:
        batch.put_bytes(data, "sep_probe.pkl")
    print(f"Probe uploaded from {path} → Modal volume 'sep-probes'")
