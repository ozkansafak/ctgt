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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ctgt.probe import SEProbe


def _latest_sep_data() -> Path:
    candidates = sorted(Path("outputs").glob("sep_data_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "No SEP training data found in outputs/. "
        "Run: modal run modal_app.py::app.collect_sep_training_data"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to SEP training data JSON")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction held out for evaluation")
    parser.add_argument("--output", default="outputs/sep_probe.pkl", help="Where to save the fitted probe")
    args = parser.parse_args()

    path = Path(args.input) if args.input else _latest_sep_data()
    print(f"Reading {path}")
    data = json.loads(path.read_text())
    rows = data["rows"]
    print(f"  {len(rows)} questions | LLM: {data['llm_model']}")

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

    # Evaluate on test set
    print("\n── Test set evaluation ─────────────────────────────────────────")
    X_test = np.array(test_hs[probe.best_layer], dtype=np.float32)
    se_test = np.array(test_se, dtype=np.float32)
    labels_test = (se_test > probe.threshold).astype(int)

    if len(np.unique(labels_test)) < 2:
        print("  Warning: test set is single-class — AUROC undefined. Run with more data.")
        sep_test_auroc = float("nan")
        se_test_auroc  = float("nan")
    else:
        probe_proba   = probe.clf.predict_proba(X_test)[:, 1]
        sep_test_auroc = roc_auc_score(labels_test, probe_proba)
        # Oracle: SE score itself predicting high-SE label
        se_test_auroc  = roc_auc_score(labels_test, se_test)

    print(f"  SE threshold (γ*)  : {probe.threshold:.3f}")
    print(f"  Best layer         : {probe.best_layer}  (train AUROC={train_aurocs[probe.best_layer]:.3f})")
    print(f"  SEP test AUROC     : {sep_test_auroc:.3f}")
    print(f"  SE  test AUROC     : {se_test_auroc:.3f}  ← oracle (10× slower at inference)")
    if not (np.isnan(sep_test_auroc) or np.isnan(se_test_auroc)):
        print(f"  Gap vs oracle      : {sep_test_auroc - se_test_auroc:+.3f}")
    print(f"\n  Inference speedup  : ~{10:.0f}× fewer LLM forward passes")
    print(f"  Compute class      : O(d) = O({hidden_dim}) matrix multiply vs O(M·d) clustering")

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    probe.save(out)
    print(f"\nProbe saved → {out}")
    print("Next: modal run modal_app.py::app.upload_probe --path", out)


if __name__ == "__main__":
    main()
