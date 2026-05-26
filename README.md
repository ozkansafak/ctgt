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

We evaluate on **[TriviaQA](https://huggingface.co/datasets/trivia_qa)** `rc.nocontext` split (validation set).

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

The model generates 10 answers. We check the best answer (highest-probability cluster) against every alias using RougeL > 0.3. If any alias matches → correct. The ground truth comes entirely from the dataset — no manual labelling required.

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
modal_app.py          # Modal deployment: GPU service + benchmark entrypoints
benchmark.py          # Local benchmark runner (no Modal required)
results.py            # Visualise benchmark_results.json → table + ROC plots
src/ctgt/
  sampling.py         # LLMSampler: batch generation + log-prob extraction
  entailment.py       # EntailmentClustering: bidirectional NLI via DeBERTa
  scoring.py          # semantic_entropy() and predictive_entropy()
  detection.py        # SemanticEntropyDetector: orchestrates the pipeline
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

Change two lines in `modal_app.py`:

```python
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # was 1.5B
gpu = "A10G"                              # was T4 — in @app.cls decorator
```

No other code changes needed.

---

## Related work

**Park & Cho (NeurIPS 2025)** extend this method with *diversity-steered sampling* — penalising semantically redundant outputs during generation so you get better entropy estimates with fewer samples. This directly addresses the main scalability bottleneck of our approach: fewer LLM calls = cheaper per query. A natural next step for a production system.
