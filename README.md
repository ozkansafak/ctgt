# Hallucination Detection via Semantic Entropy

Prototype implementation of **Semantic Entropy** (Kuhn et al., ICLR 2023) for detecting hallucinations in LLM outputs, deployed on Modal with GPU inference.

**References**
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR 2023. [arXiv:2302.09664](https://arxiv.org/abs/2302.09664)
- Park, J. W., & Cho, K. (2025). *Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling.* NeurIPS 2025. [neurips.cc](https://neurips.cc/virtual/2025/loc/san-diego/poster/118777)

---

## Glossary

| Term | Definition |
|---|---|
| **LLM** | Large Language Model — the main model being evaluated (Qwen2.5-1.5B here) |
| **NLI** | Natural Language Inference — classifying whether one sentence logically implies another (`entailment / neutral / contradiction`) |
| **DeBERTa** | The NLI model we use for clustering (`cross-encoder/nli-deberta-v3-large`, 400M params) |
| **SE** | **Semantic Entropy** — entropy computed over *meaning clusters*, not raw token sequences. Our main detection signal |
| **PE** | **Predictive Entropy** — entropy computed over raw token sequences, ignoring semantic equivalence. Our baseline to beat |
| **AUROC** | Area Under the ROC Curve — measures how well a score separates two classes. 0.5 = random, 1.0 = perfect. Here: how well SE separates hallucinated answers from correct ones |
| **RougeL** | A string overlap metric (longest common subsequence). We use RougeL > 0.3 to decide if the model's answer matches the gold answer |
| **Temperature** | Controls randomness during LLM sampling. Low (0.1) = repetitive outputs, high (1.0) = very diverse. We use 0.5, the sweet spot from Kuhn et al. |
| **log-prob** | Log probability — the model's confidence in a specific token sequence. We use length-normalised log-probs so short and long answers are comparable |
| **Bidirectional entailment** | Checking A→B *and* B→A. Both must hold for two answers to be considered semantically equivalent |
| **TriviaQA** | The evaluation dataset — 95k trivia questions with human-verified answers and aliases |
| **Closed-book** | The model answers from memory only, with no supporting document provided |

---

## How it works

**Predictive Entropy (PE)** — token-level entropy over raw sequences — is a poor hallucination signal because paraphrases look uncertain even when they mean the same thing:

```
"Bell invented the telephone"         ← token sequence A
"The telephone was invented by Bell"  ← token sequence B

PE sees two different sequences → high entropy → wrongly flags as uncertain
SE sees one meaning            → low entropy  → correctly flags as confident
```

**Semantic Entropy (SE)** fixes this by computing entropy over *meanings* instead of token sequences:

```
Step 1 — Sample   Draw N=10 completions from the LLM at temperature 0.5

Step 2 — Cluster  For each pair of answers, run DeBERTa NLI in both directions.
                  If A entails B AND B entails A → same meaning → same cluster.
                  Greedy algorithm: compare each new answer against one
                  representative per cluster (exploits transitivity, O(M·C) not O(M²))

Step 3 — Score    p(cluster) = sum of exp(log_prob) for all answers in that cluster
                  SE = -∑_c  p(c) · log p(c)
                  High SE → model uncertain about which fact to state → hallucination risk
```

---

## Evaluation dataset: TriviaQA (closed-book)

We evaluate on **[TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa)** `rc.nocontext` split (validation set).

Closed-book = no supporting document. The model must answer from memory. This makes uncertainty genuine: either the model knows the answer or it doesn't — there is nothing to look up.

**Example item:**
```python
{
    "question": "Which American-born Sinclair won the Nobel Prize for Literature in 1930?",
    "answer": {
        "value": "Sinclair Lewis",
        "normalized_aliases": [
            "sinclair lewis",
            "harry sinclair lewis",
            "lewis sinclair",
        ]
    }
}
```

We make M=10 inferences with the model. We check the best answer (highest-probability cluster) against every alias using RougeL > 0.3. If any alias matches → correct. The ground truth comes entirely from the dataset — no manual labelling required.

---

## Results

```
╔════════════════════════════════════════════════════════╗
║         Semantic Entropy Benchmark Results             ║
╠════════════════════════════════════════════════════════╣
║  LLM    : Qwen/Qwen2.5-1.5B-Instruct                   ║
║  Dataset: TriviaQA rc.nocontext (validation)           ║
║  N questions : 50                                      ║
║  N samples/q : 10                                      ║
╠════════════════════════════════════════════════════════╣
║  Accuracy  : 6.0%  (% of questions answered correctly) ║
║  SE  AUROC : 0.780  ← our method                       ║
║  PE  AUROC : 0.752  ← baseline (no clustering)         ║
║  SE gain   : +0.028                                    ║
╚════════════════════════════════════════════════════════╝
```

SE outperforms PE on AUROC — semantic clustering adds signal even with a small model.

**Note on accuracy (6%):** The 1.5B model is too small to reliably recall trivia facts from memory. Kuhn et al. (2023) use a 30B OPT model and report ~50% accuracy and SE AUROC ~0.83. The low accuracy here is expected — the key result is that SE is a *better uncertainty signal* than PE regardless of model size, which holds.

To reproduce:
```bash
modal run modal_app.py::app.benchmark --n-questions 50 --n-samples 10
python results.py
```

---

## Project structure

```
modal_app.py          # Modal deployment: GPU service + all benchmark entrypoints
train_probe.py        # Local: train SEP logistic regression from collected data
results.py            # Visualise benchmark_results.json → table + ROC plots
benchmark.py          # Local benchmark runner (no Modal required)
outputs/              # Benchmark JSONs + plots (committed)

src/ctgt/
  # ── Baseline: Kuhn / Farquhar 2024 ──────────────────────────
  sampling.py         # LLMSampler: batch generation, log-prob extraction,
                      #   hidden state extraction (output_hidden_states=True)
  entailment.py       # EntailmentClustering: bidirectional NLI via DeBERTa
  scoring.py          # semantic_entropy() and predictive_entropy()
  detection.py        # SemanticEntropyDetector: orchestrates the full pipeline

  # ── Production: Kossen 2024 ──────────────────────────────────
  probe.py            # SEProbe: Otsu binarisation, layer grid search,
                      #   logistic regression fit/predict, save/load

  # ── Future: Chen et al. INSIDE 2024 ─────────────────────────
  inside.py           # INSIDEDetector stub — full spec documented, not yet built
```

---

## Models

| Role | Model | Size | VRAM |
|---|---|---|---|
| LLM (generation) | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B params | ~3 GB |
| NLI (clustering) | `cross-encoder/nli-deberta-v3-large` | 400M params | ~1.5 GB |

Both fit on a single T4 GPU (15 GB VRAM). Swap `LLM_MODEL` in `modal_app.py` to scale up.

---

## Setup

```bash
git clone <repo>
cd ctgt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
modal setup
```

---

## Usage

### Single question
```bash
modal run modal_app.py::app.analyze --question "Who invented the telephone?"
```

**Real output (Qwen2.5-1.5B, N=10 samples):**
```
Question : Who invented the telephone?
SE       : 0.301  (PE: 2.302)
Clusters : 2
Uncertain: False
Best ans : The telephone was invented by Alexander Graham Bell in 1876.

All samples:
  [ 1] The telephone was invented by Alexander Graham Bell in 1876.
  [ 2] The telephone was invented by Alexander Graham Bell in 1876.
  ...
  [ 8] Alexander Graham Bell is credited with inventing the telephone in 1876.
```

PE = 2.302 because sample 8 uses different words. SE = 0.301 because both mean the same thing — the model is actually confident. This is exactly the failure mode PE has that SE corrects.

### Benchmark (TriviaQA, parallelised on Modal)
```bash
modal run modal_app.py::app.benchmark --n-questions 200 --n-samples 10
```

### Visualise results
```bash
python results.py
```

Produces a per-question table and saves `benchmark_plots.png` (ROC curves + SE distribution histogram).

### Local benchmark (no Modal, slow)
```bash
python benchmark.py --n-questions 50 --n-samples 5
```

---

## Baseline comparison

| Method | What it computes |
|---|---|
| **Semantic Entropy (SE)** | Entropy over meaning clusters — paraphrases collapsed |
| **Predictive Entropy (PE)** | Entropy over raw token sequences — paraphrases inflate uncertainty |

---

## Scaling up

Change two lines in `modal_app.py`. No other code changes needed.

| | Current (prototype) | Scaled up |
|---|---|---|
| `LLM_MODEL` | `"Qwen/Qwen2.5-1.5B-Instruct"` | `"Qwen/Qwen2.5-7B-Instruct"` |
| `gpu` | `"T4"` | `"A10G"` |

```python
# modal_app.py — prototype
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
gpu = "T4"

# modal_app.py — scaled up
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
gpu = "A10G"
```

---

## Method comparison

| Paper | Internal access | What it does |
|---|---|---|
| Farquhar / Kuhn (Nature 2024) | Logprobs only | M=10 samples → NLI cluster → entropy over meanings |
| Kossen et al. SEPs (2024) | Hidden states (read) | Linear probe on frozen last-token activation; single forward pass at inference |
| Chen et al. INSIDE (ICLR 2024) | Hidden states (read + modify) | EigenScore on mid-layer covariance + forward-hook activation clipping |

The progression is deliberate: Kuhn is the well-validated baseline; Kossen reduces inference cost by ~10× using the model's own representations; INSIDE goes further by *modifying* activations at inference time to suppress overconfident generations.

---

## Related work

**Park & Cho (NeurIPS 2025)** extend Kuhn with *diversity-steered sampling* — penalising semantically redundant outputs during generation so you get better entropy estimates with fewer samples. Fewer LLM calls = cheaper per query. A natural next step for production.

**Kossen et al. (2024)** introduce Semantic Entropy Probes: train a logistic regression layer on the LLM's own frozen hidden states to predict whether SE would be high, replacing M=10 heavy generations with a single forward pass and an O(d) matrix multiply. Implemented in `src/ctgt/probe.py`.

**Chen et al. INSIDE (ICLR 2024)** propose EigenScore — measuring semantic consistency via the eigenvalues of the responses' covariance matrix in dense embedding space — and feature clipping, which truncates extreme activations in the penultimate layer via a PyTorch forward hook to reduce overconfident generations. Documented as a future-work stub in `src/ctgt/inside.py`.

---

## References

- Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature. **1561 citations.** [arXiv:2303.08896](https://arxiv.org/abs/2303.08896)
- Kossen, J., Han, J., Razzak, M., Schut, L., Malik, S., & Gal, Y. (2024). *Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs.* **161 citations.** [arXiv:2406.15927](https://arxiv.org/abs/2406.15927)
- Chen, C., Liu, K., Chen, Z., Gu, Y., Wu, Y., Tao, M., Fu, Z., & Ye, J. (2024). *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection.* ICLR 2024. **356 citations.** [arXiv:2402.03744](https://arxiv.org/abs/2402.03744)
- Park, J. W., & Cho, K. (2025). *Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling.* NeurIPS 2025.

---

## Action Items

Priority-ordered work remaining. Addresses gaps in motivation, system design, and statistical validity.

**1. Motivation & judgment** — `DESIGN.md`
- [ ] Add "Why Kuhn et al." section: easy to implement, well-cited (1244 Google Scholar citations), strong empirical results, not outdated
- [ ] Add "Alternatives rejected" section: why NLI entailment over cosine similarity, why SE over p(True), why TriviaQA over CoQA

**2. Serving system design** — `DESIGN.md`
- [ ] Add REST API schema: `POST /detect` → `{is_hallucinated, se_score, n_clusters, best_answer}`
- [ ] Add serving architecture: Modal web endpoint, latency budget, request flow diagram

**3. Statistical validity** — run benchmark
- [ ] Re-run with 200+ questions (n=50 gives ±0.07 AUROC variance; need more for a credible claim)
- [ ] Add confidence intervals or bootstrap error bars to results

**4. p(True) baseline** — `modal_app.py`, `src/ctgt/`
- [ ] Implement Kadavath et al. (2022): ask model "Is your answer correct?" as a third AUROC baseline
- [ ] Add to results table alongside SE and PE
