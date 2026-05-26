"""
Train and evaluate a Semantic Entropy Probe (Kossen et al., 2024).

Reads a SEP training data JSON produced by:
    modal run modal_app.py::app.collect_sep_training_data

Usage:
    python train_probe.py --input outputs/sep_data_*.json
    python train_probe.py --input outputs/sep_data_*.json --test-size 0.2 --output outputs/sep_probe.pkl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from ctgt.kossen_2024.probe import SEProbe


def _latest_sep_data() -> Path:
    candidates = sorted(Path("outputs").glob("sep_data_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "No SEP training data found in outputs/. "
        "Run: modal run modal_app.py::app.collect_sep_training_data"
    )


def _plot_sep(data: dict, probe, rows: list, sep_scores_cv: "np.ndarray", sep_hall_cv: float, out_path: "Path") -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import roc_curve

        se_scores = [r["semantic_entropy"] for r in rows]
        pe_scores = [r["predictive_entropy"] for r in rows]
        labels_wrong = [1 - int(r["is_correct"]) for r in rows]

        model_short = data["llm_model"].split("/")[-1]
        model_short = model_short.replace("-Instruct-v0.3", "").replace("-Instruct", "")
        n_q = data["n_questions"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Semantic Entropy Probes (Kossen 2024) · {model_short} · N={n_q} · layer {probe.best_layer}",
            fontsize=12,
        )

        ax = axes[0]
        for scores, label, color in [
            (se_scores,      f"SE oracle (AUROC={data['se_auroc']:.3f})",  "#2196F3"),
            (sep_scores_cv,  f"SEP probe (AUROC={sep_hall_cv:.3f})",       "#4CAF50"),
            (pe_scores,      f"PE       (AUROC={data['pe_auroc']:.3f})",   "#FF5722"),
        ]:
            fpr, tpr, _ = roc_curve(labels_wrong, scores)
            ax.plot(fpr, tpr, lw=2, label=label)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        ax = axes[1]
        correct_sep = [sep_scores_cv[i] for i, r in enumerate(rows) if r["is_correct"]]
        wrong_sep   = [sep_scores_cv[i] for i, r in enumerate(rows) if not r["is_correct"]]
        bins = np.linspace(0, 1, 25)
        ax.hist(wrong_sep,   bins=bins, alpha=0.6, label="Wrong",   color="#F44336",
                edgecolor="black", linewidth=0.8)
        ax.hist(correct_sep, bins=bins, alpha=0.9, label="Correct", color="#4CAF50")
        ax.axvline(0.5, color="k", linestyle="--", lw=1)
        ax.set_xlabel("P(uncertain) — SEP probe score")
        ax.set_ylabel("Count")
        ax.set_title("SEP Distribution: Correct vs Wrong Answers")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"\nSEP plots saved to {out_path}")
        plt.show()
    except ImportError:
        print("\nInstall matplotlib to generate plots:  pip install matplotlib")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to SEP training data JSON")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction held out for evaluation")
    parser.add_argument("--output", default=None, help="Where to save the fitted probe (default: outputs/sep_probe_<model>.pkl)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    path = Path(args.input) if args.input else _latest_sep_data()
    print(f"Reading {path}")
    data = json.loads(path.read_text())
    rows = data["rows"]
    print(f"  {len(rows)} questions | LLM: {data['llm_model']}")

    model_slug = data["llm_model"].split("/")[-1].lower()
    default_out = f"outputs/sep_probe_{model_slug}.pkl"
    out = Path(args.output) if args.output else Path(default_out)

    se_scores = [r["semantic_entropy"] for r in rows]

    # JSON serialises int keys as strings; normalise to int before handing to probe
    sample_hs = rows[0]["hidden_states"]
    layer_keys_str = sorted(sample_hs.keys(), key=int)
    layer_keys_int = [int(k) for k in layer_keys_str]
    hidden_dim = len(sample_hs[layer_keys_str[0]])
    print(f"  {len(layer_keys_int)} layers extracted | hidden_dim={hidden_dim}")

    # Train / test split on question indices
    indices = list(range(len(rows)))
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=42)
    print(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")

    train_se = [se_scores[i] for i in train_idx]
    test_se  = [se_scores[i] for i in test_idx]

    # Build per-layer hidden state matrices (int keys for probe.fit)
    train_hs = {
        int(k): [rows[i]["hidden_states"][k] for i in train_idx]
        for k in layer_keys_str
    }
    test_hs = {
        int(k): [rows[i]["hidden_states"][k] for i in test_idx]
        for k in layer_keys_str
    }

    # Fit probe — grid search over layers on training data
    probe = SEProbe()
    train_aurocs = probe.fit(train_hs, train_se)

    # Print layer grid search results
    print("\n── Layer grid search (train AUROC) ─────────────────────────────")
    for layer_idx in sorted(train_aurocs.keys()):
        marker = " ◄ best" if layer_idx == probe.best_layer else ""
        bar_len = int(train_aurocs[layer_idx] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  Layer {layer_idx:3d}: {bar} {train_aurocs[layer_idx]:.3f}{marker}")

    # ── 5-fold cross-validation on all 200 questions ─────────────────
    # Uses all data to estimate AUROC, avoiding the variance of a single split.
    # Two metrics reported:
    #   (A) Hallucination detection: probe vs is_correct ground truth
    #   (B) SE approximation: probe vs SE-binarised labels (Kossen 2024 metric)
    print("\n── 5-fold cross-validation (all 200 questions) ─────────────────")

    all_se    = np.array(se_scores, dtype=np.float32)
    # Binarise SE with the threshold already fitted on the train split
    se_labels_all   = (all_se > probe.threshold).astype(int)
    hall_labels_all = [1 - int(r["is_correct"]) for r in rows]

    X_all = np.array(
        [rows[i]["hidden_states"][str(probe.best_layer)] for i in range(len(rows))],
        dtype=np.float32,
    )

    def safe_auroc(labels, scores):
        labels = np.array(labels)
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sep_hall_fold, se_hall_fold, sep_se_fold = [], [], []
    sep_scores_cv = np.zeros(len(rows))

    for fold_train, fold_test in skf.split(X_all, hall_labels_all):
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X_all[fold_train], se_labels_all[fold_train])
        proba = clf.predict_proba(X_all[fold_test])[:, 1]
        sep_scores_cv[fold_test] = proba

        sep_hall_fold.append(safe_auroc([hall_labels_all[i] for i in fold_test], proba))
        se_hall_fold.append( safe_auroc([hall_labels_all[i] for i in fold_test], all_se[fold_test]))
        sep_se_fold.append(  safe_auroc(se_labels_all[fold_test], proba))

    def mean_nan(vals): return float(np.nanmean(vals))

    sep_hall_cv = mean_nan(sep_hall_fold)
    se_hall_cv  = mean_nan(se_hall_fold)
    sep_se_cv   = mean_nan(sep_se_fold)

    print(f"  SE threshold (γ*)  : {probe.threshold:.3f}")
    print(f"  Best layer         : {probe.best_layer}  (train AUROC={train_aurocs[probe.best_layer]:.3f})")
    print()
    print(f"  (A) Hallucination detection  [label = is_correct from dataset]")
    print(f"      SE  AUROC (oracle, 5-fold): {se_hall_cv:.3f}  ← 10× slower, needs M samples + NLI")
    print(f"      SEP AUROC (probe, 5-fold) : {sep_hall_cv:.3f}  ← single forward pass")
    if not (np.isnan(sep_hall_cv) or np.isnan(se_hall_cv)):
        print(f"      Gap                       : {sep_hall_cv - se_hall_cv:+.3f}")
    print()
    print(f"  (B) SE approximation  [label = SE > γ* = {probe.threshold:.3f}]")
    print(f"      SEP AUROC (5-fold)         : {sep_se_cv:.3f}  (Kossen 2024 metric)")
    print()
    print(f"  Inference speedup  : ~10× fewer LLM forward passes")
    print(f"  Compute class      : O(d) = O({hidden_dim}) vs O(M · NLI calls)")

    out.parent.mkdir(exist_ok=True)
    probe.save(out)
    print(f"\nProbe saved → {out}")
    print("Next: modal run modal_app.py::app.upload_probe --path", out)

    if not args.no_plot:
        plot_out = path.with_name(path.stem + "_sep_plots.png")
        _plot_sep(data, probe, rows, sep_scores_cv, sep_hall_cv, plot_out)


if __name__ == "__main__":
    main()
