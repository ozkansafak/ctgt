# Hallucination Detection via Semantic Entropy

---

## 1. Problem

LLMs can produce fluent, confident, factually wrong answers. Token-level confidence is not enough because natural language has many valid surface forms for the same meaning. This project implements Semantic Entropy as an expensive reference detector and Semantic Entropy Probes as a production-friendly approximation.

---

## 2. Method

Semantic Entropy samples M completions, clusters them by meaning using bidirectional NLI, then computes entropy over semantic clusters rather than strings. High SE means the model's sampled answers disagree in meaning.

Semantic Entropy Probes distill this expensive signal into a lightweight logistic regression probe over the LLM hidden state. At inference time, the probe reads the last-input-token activation from the best layer and outputs a hallucination-risk score with no extra generations and no NLI calls.

### 2.1 Semantic Entropy (Kuhn, 2023)

Standard **Predictive Entropy (PE)** measures token-based entropy, which gets inflated by paraphrasing:

```
PE = -∑_s  p(s|x) · log p(s|x)
```

**Semantic Entropy (SE)** collapses different completions into meaning clusters first, then computes entropy over those clusters:

```
SE = -∑_c  p(c|x) · log p(c|x),   p(c|x) = ∑_{s ∈ c} p(s|x)
```

> **Key insight:** SE does not predict whether a specific completion is wrong. It detects whether the model is in a hallucination-prone state — i.e. whether its answers disagree in meaning across samples. The known blind spot is that if the model is consistently wrong in the same way across all M samples, SE = 0 and the hallucination tendency is not flagged.

**Algorithm:**
1. **Sample:** draw M=10 completions at temperature 0.5. Record length-normalized log-probs: `log_prob = (1/n) · ∑ log p(token_i)`
2. **Cluster:** run DeBERTa NLI bidirectionally against each cluster's representative; assign on mutual entailment, else start a new cluster — O(M · C).
3. **Aggregate:** sum sentence probabilities within each cluster: `p(c) = ∑_{s∈c} exp(log_prob_s)`
4. **Entropy:** `SE = -∑_c p(c) · log p(c)`. SE = 0: one shared meaning. SE = log(10) ≈ 2.3: every sample disagrees.

*Example: "Who invented the telephone?"*

```
s1:  "Alexander Graham Bell"      log_prob = -0.12 ┐
s2:  "Bell invented it in 1876"   log_prob = -0.15 ├─ Cluster 1  p = 0.95
s3:  "The telephone is Bell's"    log_prob = -0.18 ┘
s10: "Nikola Tesla"               log_prob = -2.10  ── Cluster 2  p = 0.05

SE = -(0.95·log 0.95 + 0.05·log 0.05) = 0.20  ->  low uncertainty
```

### 2.2 Semantic Entropy Probes (Kossen, 2024)

Kuhn's algorithm makes M=10 inferences per prompt plus NLI calls — a production bottleneck. Kossen's insight is that **the model already encodes its uncertainty in the hidden states before generating a single output token**. A cheap linear probe trained on last-token embeddings can read it out at inference time.

**Training (one-time):**
1. Run Kuhn's pipeline on N questions → one SE score per question.
2. Binarize SE scores via Otsu threshold → 0/1 labels.
3. One forward pass per question → extract last-input-token hidden state at every layer, shape `(N, d)`.
4. Fit logistic regression per layer; pick the layer with highest validation AUROC.
5. Refit on all training data at the best layer. Save `W`, `b`, and `best_layer`.

**Inference:**
1. One forward pass through the LLM.
2. Slice `hidden_state[best_layer][-1]`, shape `(d,)`.
3. `P(uncertain) = sigmoid(W * h + b)` — one matrix multiply.

No sampling, no NLI.

### 2.3 Method comparison

| Method | Model access | Per-query cost | Key idea | Status |
|---|---|---|---|---|
| Kuhn / Farquhar SE | Logprobs | M=10 samples + NLI | Entropy over semantic clusters  |
| Kossen SEP | Hidden states (read) | 1 forward pass | Linear probe on last-input-token activation | 

The progression represents increasing depth of access: output probabilities → reading internals → modifying internals.

---

## 3. Evaluation

Dataset: TriviaQA `rc.nocontext` closed-book QA (validation set, 17,944 questions).  
Correctness: RougeL against any gold alias, threshold 0.3.  
Metric: AUROC, where a higher hallucination score should predict a wrong answer.

\[
\mathrm{AUROC} = P(s(x_{\text{hallucinated}}) > s(x_{\text{correct}}))
\]

**Experimental split.** TriviaQA is shuffled with seed 42. The primary reported results use a 10,000-question benchmark split, questions 300–10,299. For SE and PE, scores are computed directly on this split. For SEP, the probe is evaluated with 5-fold cross-validation on the same split: in each fold, the probe is trained on 80% of questions and evaluated on the held-out 20%. The first 300 questions are used only for wall-time/smoke-test measurements.

**Hyperparameters**

| Parameter | Value | Rationale |
|---|---|---|
| Samples M | 10 | Diminishing returns beyond 10 (Kuhn et al., Fig. 3b) |
| Temperature | 0.5 | Balances diversity vs accuracy |
| Max new tokens | 128 | Sufficient for factual QA |
| Default threshold τ | ln(2) = 0.69 | Entropy of a 50/50 split between two meanings |

---

## 4. Results

TriviaQA `rc.nocontext`, M=10 samples, temp=0.5, Modal A10G GPU.

| Model | Acc | SE AUROC | SEP AUROC | PE AUROC | SEP gap | SEP best layer |
|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 70% | 0.772 | 0.742 | 0.273 | −0.030 | 31 |
| Meta-Llama-3.1-8B-Instruct | **75%** | **0.790** | **0.746** | 0.258 | −0.044 | 31 |
| Qwen2.5-1.5B-Instruct | 41% | 0.728 | 0.705 | 0.303 | −0.024 | 26 |
| Kuhn et al. (OPT-30B) | ~50% | ~0.830 | n/a | n/a | n/a | n/a |

SE and PE scored directly on N=10,000. SEP evaluated via 5-fold CV on the same split. SEP probes recover 94–97% of SE's AUROC with a single forward pass: no sampling, no NLI. Wall times measured on N=300 smoke-test runs on A10G; speedup is lower than the theoretical 10× because Modal autoscaling already parallelizes Kuhn's sampling.

Full per-model plots in [APPENDIX.md](APPENDIX.md).

<p align="center"><img src="outputs/sep_probe_mistral-7b-instruct-v0.3_sep_plots.png" width="75%"></p>

*Mistral-7B · 5-fold CV · layer 31 · SE AUROC=0.772 · SEP AUROC=0.742 · PE AUROC=0.273. All three methods on one ROC: SE (blue) and SEP (green) both detect hallucination risk; PE (red) inverts below the diagonal.*

---

## 5. Why PE Fails

PE AUROC < 0.5 means PE systematically ranks correct answers as *more* uncertain than hallucinated ones — the opposite of useful.

PE fails because it measures lexical uncertainty, not semantic uncertainty. Correct answers often have high PE: the same fact expressed through aliases, qualifiers, or varied syntax generates diverse token sequences. Hallucinated answers from an instruction-tuned model tend to have low PE: the model picks one confident, plausible-sounding entity and states it flatly. This ranking inversion is not noise; it is a structural artifact of instruction tuning. SE fixes it by collapsing lexical variants into meaning clusters before computing entropy.

Production design details (request flow, latency budget, offline training) are in [APPENDIX.md](APPENDIX.md).

---

## 6. Limitations and Future Work

- SE and SEP detect uncertainty, not guaranteed incorrectness.
- Confidently wrong answers remain a blind spot: if the model outputs the same wrong answer across all M samples, SE = 0 and it is not flagged.
- Requires logprob and hidden-state access; black-box APIs are not supported.

**Future work:**

- **Retrieval-grounded verifier:** for high-risk answers, retrieve evidence and check answer support. Directly addresses the confident-wrong blind spot.
- **p(True) baseline (Kadavath et al., 2022):** ask the model to self-assess correctness and compare AUROC against SE and SEP.
- **Diversity-steered sampling (Park & Cho, NeurIPS 2025):** penalize semantically redundant outputs during generation. Same AUROC with ~4 samples instead of 10, 60% LLM cost reduction.

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
