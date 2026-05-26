# Technical Design Doc: Hallucination Detection via Semantic Entropy

---

## 1. Introduction

LLMs hallucinate — they produce confident-sounding outputs that are factually wrong. The model doesn't say "I'm not sure." It just generates tokens.

One natural signal is token-level probability. But this fails because natural language has many ways to express the same meaning. A model confident about a fact may still generate varied phrasings across samples, inflating apparent uncertainty.

> "Bell invented the telephone" and "The telephone was invented by Bell" mean the same thing but count as two different outcomes under token entropy.

This project implements and benchmarks three papers that measure hallucination risk by testing whether the model's answers agree with themselves — at increasing depths of model access.

---

## 2. Methods

### 2.1 Semantic Entropy — Kuhn / Farquhar (2024)  *(implemented)*

Standard **Predictive Entropy (PE)** measures spread at the token level:

```
PE = -∑_s  p(s|x) · log p(s|x)
```

This is inflated by paraphrases. **Semantic Entropy (SE)** collapses paraphrases into meaning clusters first, then computes entropy over those clusters:

```
SE = -∑_c  p(c|x) · log p(c|x),   p(c|x) = ∑_{s ∈ c} p(s|x)
```

High SE means the model's answers genuinely disagree on *meaning* — a strong hallucination signal.

**Step 1 — Sample**

Draw M=10 completions at temperature 0.5. Record each completion's length-normalised log-probability:

```
log_prob = (1/n) · ∑_i log p(token_i | context, tokens_{<i})
```

Length normalisation makes short and long answers comparable.

**Step 2 — Cluster by meaning**

Use **DeBERTa-large** (400M parameters, fine-tuned on Multi-NLI) to check semantic equivalence. Both directions must hold:

```
A entails B  AND  B entails A  →  same cluster
```

A one-way implication is not enough: "Paris is in France" entails "Paris exists" but they are not equivalent. Greedy transitivity keeps comparisons at O(M·C) rather than O(M²).

**Step 3 — Entropy over clusters**

Sum probabilities within each cluster, then compute entropy:

```
SE = -∑_c  p(c) · log p(c)
```

SE = 0: all samples share one meaning (certain). SE ≈ log(10) = 2.3: every sample has a distinct meaning (maximally uncertain).

---

### 2.2 Semantic Entropy Probes — Kossen (2024)  *(implemented)*

Kuhn needs M=10 samples plus NLI per question. Kossen's insight: **the model already encodes its uncertainty in the hidden states before generating a single output token**. A cheap linear probe can read it out.

**Training (one-time, offline):**

1. Run Kuhn's pipeline on N questions → one SE score per question. This is the *teacher*.
2. Binarize SE scores → 0/1 labels using an Otsu threshold γ* (minimises within-class variance).
3. For the same N questions, run one additional forward pass → extract the hidden state at the **last input token position** at every layer. At that position the transformer has attended to the entire question; its hidden state is the model's compressed state of knowledge just before it commits to an answer.
4. Grid-search layers: train a logistic regression per layer, pick the layer with the highest validation AUROC (layer 21 for Mistral-7B, d=4096).
5. Refit the final logistic regression on all training data at the best layer.

**Inference (per question):**

1. One forward pass through the frozen LLM.
2. Slice `hidden_state[best_layer][-1]` — shape `(d,)`.
3. `P(uncertain) = sigmoid(W · h + b)` — one matrix multiply.

No sampling. No NLI. No clustering. ~10× fewer forward passes than Kuhn.

---

### 2.3 INSIDE — Chen et al. (ICLR 2024)  *(not yet implemented)*

Two mechanisms operating on K=10 response embeddings at the **middle transformer layer** (≈ L/2):

**EigenScore:** build a covariance matrix Σ = Z^T · J · Z over the K hidden-state vectors Z. Score = (1/K) Σᵢ log(λᵢ). When responses are semantically similar, eigenvalue mass concentrates on one component (low score). When they diverge, eigenvalues spread (high score). No NLI needed — semantic divergence is captured in embedding geometry. Reported AUROC on TriviaQA/LLaMA-7B: ~83% vs ~65% for SE.

**Feature Clipping:** a forward hook that clips extreme activations in the penultimate layer using percentile thresholds from a calibration set. Unlike Kossen (read-only), this *modifies* the computation graph at inference time. Reduces overconfident generations without retraining. Full implementation spec in [src/ctgt/inside_2024/inside.py](src/ctgt/inside_2024/inside.py).

---

### 2.4 Method comparison

| Method | Model access | Per-query cost | Key idea |
|---|---|---|---|
| Kuhn / Farquhar SE | Logprobs | M=10 samples + NLI | Entropy over semantic clusters |
| Kossen SEP | Hidden states (read) | 1 forward pass | Linear probe on last-input-token activation |
| Chen INSIDE | Hidden states (read + write) | K=10 samples, no NLI | EigenScore on mid-layer covariance + activation clipping |

The progression represents increasing depth of access: output probabilities → reading internals → modifying internals.

---

## 3. Dataset: TriviaQA

We evaluate on **TriviaQA** `rc.nocontext` (validation set, 17,944 questions) — trivia answered from memory, no supporting document. Closed-book forces genuine uncertainty: the model either knows the answer or it doesn't.

**Correctness criterion:** RougeL(model answer, any gold alias) > 0.3, following Kuhn et al.

RougeL is the F-score of the Longest Common Subsequence between reference X (length m) and prediction Y (length n):

$$R_{lcs} = \frac{LCS(X,Y)}{m}, \quad P_{lcs} = \frac{LCS(X,Y)}{n}, \quad \text{RougeL} = \frac{2\,R_{lcs}\,P_{lcs}}{R_{lcs} + P_{lcs}}$$

We check the model's primary answer against every gold alias; a match on any alias counts as correct.

---

## 4. Hallucination Detection: End-to-End

For each question the system produces a **primary answer** (representative of the highest-probability semantic cluster) and an **SE score** (a scalar in [0, log M] measuring semantic diversity across M samples). The SE score drives a threshold detector:

```
SE ≥ τ  →  flag as hallucination
SE < τ  →  return answer to user
```

Rather than fixing τ and measuring accuracy, we sweep τ and compute **AUROC** — the probability that a randomly chosen hallucinated answer has higher SE than a randomly chosen correct answer. AUROC = 0.5 is random; AUROC = 1.0 is perfect separation.

*Example — "Who invented the telephone?"*

```
s1:  "Alexander Graham Bell"      log_prob = -0.12 ┐
s2:  "Bell invented it in 1876"   log_prob = -0.15 ├─ Cluster 1  p = 0.95
s3:  "The telephone is Bell's"    log_prob = -0.18 ┘
s10: "Nikola Tesla"               log_prob = -2.10  ── Cluster 2  p = 0.05

SE = -(0.95·log 0.95 + 0.05·log 0.05) = 0.20  →  low uncertainty
```

**Hyperparameters**

| Parameter | Value | Rationale |
|---|---|---|
| Samples M | 10 | Diminishing returns beyond 10 (Kuhn et al., Fig. 3b) |
| Temperature | 0.5 | Balances diversity vs accuracy |
| Max new tokens | 128 | Sufficient for factual QA |
| Default threshold τ | ln(2) ≈ 0.69 | Entropy of a 50/50 split between two meanings |

---

## 5. Results

All runs: TriviaQA `rc.nocontext`, N=300 questions, M=10 samples, temp=0.5, Modal A10G GPU.

### 5.1 Summary

| Model | Params | Acc | Kuhn SE AUROC | PE AUROC | Kossen SEP AUROC | SEP gap |
|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 7B | 61% | 0.720 | 0.330 | 0.750 | −0.030 |
| Llama-3.1-8B-Instruct | 8B | — | — | — | — | — |
| Qwen2.5-1.5B-Instruct | 1.5B | 39% | 0.755 | 0.293 | — | — |
| Kuhn et al. (OPT-30B) | 30B | ~50% | ~0.830 | — | — | — |

*Kossen SEP and Llama results pending; table will be updated after benchmarks complete.*

### 5.2 Per-model results

Each model shows two plots: **Kuhn SE** (left pair — ROC + SE distribution) and **Kossen SEP** (right pair — ROC + probe score distribution).

---

#### Mistral-7B-Instruct-v0.3

<p align="center"><img src="outputs/sep_data_mistral-7b-instruct-v0.3_q300_s10_t0.5_20260526_021404_plots.png" width="75%"></p>

*Kuhn SE · Mistral-7B · N=300 · SE AUROC=0.720 · PE AUROC=0.330 · Acc=61% · 75s. PE falls below the diagonal — the concise system prompt makes the model lexically rigid, so all M=10 samples return the same wrong tokens on incorrect questions (PE≈0) while correct answers show slight variation. SE is unaffected: "JFK", "John Kennedy", "John F. Kennedy" all land in one cluster.*

<p align="center"><img src="outputs/sep_data_mistral-7b-instruct-v0.3_q300_s10_t0.5_20260526_021404_sep_plots.png" width="75%"></p>

*Kossen SEP · Mistral-7B · N=300 · 5-fold CV · layer 21. SEP probe achieves SE-level AUROC at 1/10th the inference cost.*

---

#### Llama-3.1-8B-Instruct

*Results pending — benchmark in progress.*

---

#### Qwen2.5-1.5B-Instruct

<p align="center"><img src="outputs/qwen2.5-1.5b-instruct_q300_s10_t0.5_20260526_021523_plots.png" width="75%"></p>

*Kuhn SE · Qwen2.5-1.5B · N=300 · SE AUROC=0.755 · PE AUROC=0.293 · Acc=39% · 90s. Same PE overconfidence pattern as Mistral. SE AUROC slightly higher despite the smaller model — likely a sampling artifact at N=300.*

<p align="center"><img src="outputs/sep_data_qwen2.5-1.5b-instruct_q300_s10_t0.5_plots_sep.png" width="75%"></p>

*Kossen SEP · Qwen2.5-1.5B · N=300 · 5-fold CV. Results pending.*

---

## 6. Limitations

**Requires logprob access.** SE needs token-level log-probs from the generating model. Black-box APIs (e.g. GPT-4 via OpenAI) do not expose these. Any open-source model works.

**Does not detect confident hallucinations.** If the model is consistently wrong in the same way across all M=10 samples, SE ≈ 0 and the answer is not flagged. SE detects *uncertainty*, not *incorrectness per se*.

**NLI errors add noise.** DeBERTa achieves 92.7% semantic equivalence accuracy (Kuhn et al.). Errors in either direction corrupt the SE estimate.

**Scale matters.** OPT-2.7B shows SE AUROC ≈ 0.5. Model capability is a prerequisite for SE to be informative — Kuhn et al. achieve ~0.83 with OPT-30B.

---

## 7. Future Work

**INSIDE (Chen et al., ICLR 2024, 356 citations)** — EigenScore + feature clipping. No NLI; operates in embedding geometry. Requires forward hooks to modify activations in-flight. Full spec in [src/ctgt/inside_2024/inside.py](src/ctgt/inside_2024/inside.py).

**Diversity-steered sampling (Park & Cho, NeurIPS 2025)** — penalise semantically redundant outputs during generation. Same AUROC with ~4 samples instead of 10 — 60% LLM cost reduction with no architecture changes.

**p(True) baseline (Kadavath et al., 2022)** — ask the model "is your answer correct?" as a self-evaluation signal. A third AUROC comparison point alongside SE and PE.

**Larger models** — Mistral-7B hits 61% accuracy. Llama-3.1-8B or Qwen2.5-7B would push accuracy higher and close the gap to Kuhn et al.'s results.

---

## References

- Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature. **1561 citations.** [arXiv:2303.08896](https://arxiv.org/abs/2303.08896)
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR 2023. [arXiv:2302.09664](https://arxiv.org/abs/2302.09664)
- Kossen, J., Han, J., Razzak, M., Schut, L., Malik, S., & Gal, Y. (2024). *Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs.* **161 citations.** [arXiv:2406.15927](https://arxiv.org/abs/2406.15927)
- Chen, C., Liu, K., Chen, Z., Gu, Y., Wu, Y., Tao, M., Fu, Z., & Ye, J. (2024). *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection.* ICLR 2024. **356 citations.** [arXiv:2402.03744](https://arxiv.org/abs/2402.03744)
- Park, J. W., & Cho, K. (2025). *Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling.* NeurIPS 2025.
- He, P. et al. (2020). *DeBERTa: Decoding-Enhanced BERT with Disentangled Attention.* [arXiv:2006.03654](https://arxiv.org/abs/2006.03654)
- Kadavath, S. et al. (2022). *Language Models (Mostly) Know What They Know.* [arXiv:2207.05221](https://arxiv.org/abs/2207.05221)
- Joshi, M. et al. (2017). *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.* ACL 2017. [arXiv:1705.03551](https://arxiv.org/abs/1705.03551)
