"""
Local benchmark runner — no Modal required.

Evaluates semantic entropy vs predictive entropy on a subset of TriviaQA
(closed-book) and reports AUROC, mirroring the Kuhn et al. (2023) evaluation.

Usage:
    python benchmark.py                          # 50 questions, 10 samples each
    python benchmark.py --n-questions 200        # larger run
    python benchmark.py --n-questions 10 --n-samples 5  # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from rouge_score import rouge_scorer as rs
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ctgt import SemanticEntropyDetector
from ctgt.entailment import EntailmentClustering
from ctgt.sampling import LLMSampler


def is_correct(prediction: str, aliases: list[str], threshold: float = 0.3) -> bool:
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
    return any(
        scorer.score(alias, prediction)["rougeL"].fmeasure > threshold
        for alias in aliases
    )


def run(
    n_questions: int,
    n_samples: int,
    temperature: float,
    llm_model: str,
    nli_model: str,
    out_path: Path,
) -> None:
    print(f"Loading TriviaQA (closed-book, validation[:{n_questions}]) …")
    dataset = load_dataset(
        "mandarjoshi/trivia_qa", "rc.nocontext", split=f"validation[:{n_questions}]"
    )

    print(f"Loading LLM  : {llm_model}")
    print(f"Loading NLI  : {nli_model}")
    sampler = LLMSampler(model_name=llm_model)
    clusterer = EntailmentClustering(model_name=nli_model)
    detector = SemanticEntropyDetector(sampler=sampler, clusterer=clusterer)

    rows: list[dict] = []
    t0 = time.perf_counter()

    for item in tqdm(dataset, desc="Scoring"):
        question = item["question"]
        aliases: list[str] = item["answer"]["normalized_aliases"]

        result = detector.analyze(
            question, n_samples=n_samples, temperature=temperature
        )

        correct = is_correct(result.most_common_answer, aliases)
        rows.append(
            {
                "question": question,
                "gold": aliases[0] if aliases else "",
                "prediction": result.most_common_answer,
                "is_correct": correct,
                "semantic_entropy": result.semantic_entropy,
                "predictive_entropy": result.predictive_entropy,
                "n_clusters": result.n_clusters,
            }
        )

    elapsed = time.perf_counter() - t0
    labels = [int(r["is_correct"]) for r in rows]

    # AUROC: negate entropy so that "more uncertain" → higher score →
    # predicts incorrectness; sklearn expects higher score = more likely positive.
    # We flip the convention: predict *incorrectness*, so positive class = wrong.
    labels_wrong = [1 - l for l in labels]
    se_auroc = roc_auc_score(labels_wrong, [r["semantic_entropy"] for r in rows])
    pe_auroc = roc_auc_score(labels_wrong, [r["predictive_entropy"] for r in rows])
    accuracy = sum(labels) / len(labels)

    summary = {
        "n_questions": len(rows),
        "n_samples": n_samples,
        "temperature": temperature,
        "llm_model": llm_model,
        "nli_model": nli_model,
        "accuracy": accuracy,
        "se_auroc": se_auroc,
        "pe_auroc": pe_auroc,
        "elapsed_s": round(elapsed, 1),
        "rows": rows,
    }

    print(f"\n{'='*56}")
    print(f" TriviaQA closed-book benchmark")
    print(f"   n={len(rows)}  accuracy={accuracy:.1%}  time={elapsed:.0f}s")
    print(f"   Semantic entropy AUROC    : {se_auroc:.3f}")
    print(f"   Predictive entropy AUROC  : {pe_auroc:.3f}")
    print(f"   SE improvement            : {se_auroc - pe_auroc:+.3f}")
    print(f"{'='*56}")

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TriviaQA semantic entropy benchmark")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-large")
    parser.add_argument("--out", default="benchmark_results.json")
    args = parser.parse_args()

    run(
        n_questions=args.n_questions,
        n_samples=args.n_samples,
        temperature=args.temperature,
        llm_model=args.llm_model,
        nli_model=args.nli_model,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
