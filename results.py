"""
Visualise benchmark results from benchmark_results.json.

Usage:
    python results.py                          # reads benchmark_results.json
    python results.py --input my_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def print_table(rows: list[dict]) -> None:
    correct = [r for r in rows if r["is_correct"]]
    wrong   = [r for r in rows if not r["is_correct"]]

    def mean(vals): return sum(vals) / len(vals) if vals else 0.0

    print("\n── Per-question breakdown ──────────────────────────────────────")
    print(f"{'Question':<55} {'SE':>6} {'Clusters':>8} {'Correct':>7}")
    print("─" * 80)
    for r in rows:
        q = r["question"][:54]
        mark = "✓" if r["is_correct"] else "✗"
        print(f"{q:<55} {r['semantic_entropy']:>6.3f} {r['n_clusters']:>8} {mark:>7}")

    print("\n── Aggregate ───────────────────────────────────────────────────")
    print(f"  {'Metric':<35} {'Correct answers':>16} {'Wrong answers':>14}")
    print(f"  {'─'*35} {'─'*16} {'─'*14}")
    print(f"  {'Avg semantic entropy':<35} {mean([r['semantic_entropy'] for r in correct]):>16.3f} "
          f"{mean([r['semantic_entropy'] for r in wrong]):>14.3f}")
    print(f"  {'Avg predictive entropy':<35} {mean([r['predictive_entropy'] for r in correct]):>16.3f} "
          f"{mean([r['predictive_entropy'] for r in wrong]):>14.3f}")
    print(f"  {'Avg # semantic clusters':<35} {mean([r['n_clusters'] for r in correct]):>16.2f} "
          f"{mean([r['n_clusters'] for r in wrong]):>14.2f}")


def print_summary(data: dict) -> None:
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║         Semantic Entropy Benchmark Results           ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  LLM model   : {data['llm_model']:<37}║")
    print(f"║  NLI model   : {data['nli_model']:<37}║")
    print(f"║  Questions   : {data['n_questions']:<37}║")
    print(f"║  Samples/q   : {data['n_samples']:<37}║")
    print("╠══════════════════════════════════════════════════════╣")
    acc = f"{data['accuracy']:.1%}"
    se  = f"{data['se_auroc']:.3f}"
    pe  = f"{data['pe_auroc']:.3f}"
    gain = f"{data['se_auroc'] - data['pe_auroc']:+.3f}"
    print(f"║  Accuracy    : {acc:<36}║")
    print(f"║  SE  AUROC   : {se:<36}║")
    print(f"║  PE  AUROC   : {pe:<36}║")
    print(f"║  SE gain     : {gain:<36}║")
    print("╚══════════════════════════════════════════════════════╝")


def plot_auroc(data: dict, path: Path = Path("benchmark_results.json")) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import roc_curve

        rows = data["rows"]
        labels_wrong = [1 - int(r["is_correct"]) for r in rows]
        se_scores    = [r["semantic_entropy"]     for r in rows]
        pe_scores    = [r["predictive_entropy"]   for r in rows]

        model_short = data["llm_model"].split("/")[-1]
        model_short = model_short.replace("-Instruct-v0.3", "").replace("-Instruct", "")
        n_q   = data["n_questions"]
        n_s   = data["n_samples"]
        temp  = data["temperature"]
        acc   = data["accuracy"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Semantic Entropy (Kuhn 2023) · "
            f"{model_short} · N={n_q} · M={n_s} · "
            f"temp={temp} · Acc={acc:.0%}",
            fontsize=10,
        )

        # ROC curves
        ax = axes[0]
        for scores, label, color, marker, fillstyle in [
            (se_scores, f"Semantic Entropy (AUROC={data['se_auroc']:.3f})", "#2196F3", "o", "full"),
            (pe_scores, f"Predictive Entropy (AUROC={data['pe_auroc']:.3f})", "#FF5722", "o", "none"),
        ]:
            fpr, tpr, _ = roc_curve(labels_wrong, scores)
            ax.plot(fpr, tpr, color=color, lw=2)
            ax.plot(fpr, tpr, marker=marker, fillstyle=fillstyle,
                    color=color, linestyle="none", ms=8, label=label)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        # SE distribution: correct vs wrong
        ax = axes[1]
        correct_se = [r["semantic_entropy"] for r in rows if r["is_correct"]]
        wrong_se   = [r["semantic_entropy"] for r in rows if not r["is_correct"]]
        bins = np.linspace(0, max(se_scores) + 0.1, 25)
        ax.hist(wrong_se,   bins=bins, alpha=0.6,  label="Wrong",   color="#F44336")
        ax.hist(correct_se, bins=bins, alpha=0.9,  label="Correct", color="#4CAF50",
                edgecolor="black", linewidth=0.8)
        ax.set_xlabel("Semantic Entropy")
        ax.set_ylabel("Count")
        ax.set_title("SE Distribution: Correct vs Wrong Answers")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        out = path.with_name(path.stem + "_plots.png")
        plt.savefig(out, dpi=150)
        print(f"\nPlots saved to {out}")
        plt.show()

    except ImportError:
        print("\nInstall matplotlib to generate plots:  pip install matplotlib")


def _latest_output() -> Path:
    """Return the most recently modified file in outputs/, or fall back to benchmark_results.json."""
    candidates = sorted(Path("outputs").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    return Path("benchmark_results.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None,
                        help="Path to results JSON. Defaults to latest file in outputs/.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    path = Path(args.input) if args.input else _latest_output()
    if not path.exists():
        print(f"No results file found at {path}. Run the benchmark first:")
        print("  modal run modal_app.py::app.benchmark --n-questions 200")
        return
    print(f"Reading {path}")

    data = json.loads(path.read_text())
    print_summary(data)
    print_table(data["rows"])
    if not args.no_plot:
        plot_auroc(data, path)


if __name__ == "__main__":
    main()
