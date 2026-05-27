# Hallucination Detection via Semantic Entropy


## 1. Problem

LLMs can produce fluent, confident but factually wrong answers. Token level confidence is not sufficient because natural language has many valid surface forms that conveys the same information. This project implements Semantic Entropy, SE, (Kuhn et al, 2023) as an expensive reference detector and Semantic Entropy Probes, SEP, (Kossen et al, 2024) as a distilled production-friendly approximation.


## 2. Method

Semantic Entropy samples M sentence completions, clusters them by meaning using a bidirectional NLI model, then computes entropy over semantic clusters rather than individual tokens. High SE means the model's sampled answers disagree in meaning.

Semantic Entropy Probes distill this expensive signal into a lightweight logistic regression probe over the last token's hidden state. At inference time, the probe reads the last input token's embedding from the best layer and outputs a hallucination-risk score without the need to draw multiple sentences and no NLI calls.

### 2.1 Semantic Entropy (Kuhn et al, 2023)

Standard **Predictive Entropy (PE)** measures token-based entropy, which gets inflated by paraphrasing:

```
PE = -∑_s  p(s|x) · log p(s|x)
```

**Semantic Entropy (SE)** firstly collapses different completions into meaning clusters, then computes entropy over those clusters (see [worked example](APPENDIX.md#a1-se-worked-example)):

```
SE = -∑_c  p(c|x) · log p(c|x),   p(c|x) = ∑_{s ∈ c} p(s|x)
```

> **Key insight:** SE does not predict whether a specific completion is wrong. It detects whether the model is in a hallucination-prone state — i.e. whether its answers disagree in meaning across samples. If the model is consistently wrong in the same way across all M samples, SE = 0 and the hallucination tendency doesn't get flagged.

**Algorithm:**
1. **Sample:** draw M=10 completions at temperature 0.5. Normalize the log probabilities wrt sentence length: `log_prob = (1/n) · ∑ log p(token_i)`
2. **Cluster:** run DeBERTa NLI bidirectionally against each cluster's representative. Assign on mutual entailment, else start a new cluster — O(M · C).
3. **Aggregate:** sum sentence probabilities within each cluster: `p(c) ∝ ∑_{s∈c} exp(log_prob_s)`, then normalize so summation of cluster probabilities add up to 1.
4. **Entropy:** `SE = -∑_c p(c) · log p(c)`. SE = 0: one shared meaning. SE = log(10) ≈ 2.3: every sample disagrees.

For example, paraphrases like "Thomas Edison," "Edison," and "the light bulb was invented by Edison" collapse into one semantic cluster, while "Nikola Tesla" forms a separate cluster. SE measures entropy over those meanings rather than over the surface strings. (See [APPENDIX.md](APPENDIX.md) for a worked numerical example.)

### 2.2 Semantic Entropy Probes (Kossen et al, 2024)

Kuhn's algorithm makes M=10 inferences per prompt plus NLI calls. Deploying this algorithm would not be feasible. Kossen's result suggests that semantic uncertainty is already linearly predictable from the last-input-token hidden state.

**Training:**
1. Run Kuhn's pipeline on N=10,000 questions -> one SE score per question.
2. Binarize SE scores via a threshold -> 0/1 labels.
3. One forward pass per question -> extract last input token's hidden state at every layer, shape `(N, d)`.
4. Fit logistic regression per layer, pick the layer with highest validation AUROC. Following Kossen et al. §5, we use sklearn's `LogisticRegression` with its default L2 penalty ($C=1.0$). Regularization is essential in the underdetermined regime $N < d$. Without it, the probe overfits.
5. Refit on all training data at the best layer. Save `W`, `b`, and `best_layer`.

**Inference:**
1. One forward pass through the LLM.
2. Extract `h = hidden_state[best_layer][-1]`, shape `(d,)`.
3. `P(hallucination) = sigmoid(W * h + b)` — one matrix multiply.

No sampling, no NLI.

### 2.3 Method comparison

| Method | Model access | Per-query cost | Key idea |
|---|---|---|---|
| Semantic Entropy | Logprobs | M=10 samples + NLI | Entropy over semantic clusters |
| Semantic Entropy Probe | Hidden states | 1 forward pass | Linear probe on last-input-token activation |


## 3. Evaluation

Dataset: TriviaQA `rc.nocontext` closed-book QA (validation set, 17,944 questions).  
Correctness Label: RougeL against any gold alias, threshold 0.3.  
Metric: AUROC for a binary classifier 
Accuracy: fraction of questions where the evaluated answer is marked correctly, a TP or TN.

$$\text{AUROC} = P\bigl(s(x_{\text{wrong}}) > s(x_{\text{correct}})\bigr)$$

**Experimental split.** TriviaQA is shuffled. The primary reported results use a 10,000-question benchmark split, questions 300–10,299. For SE and PE, scores are computed directly on this split. For SEP, the probe is evaluated with 5-fold cross-validation on the same split: in each fold, the probe is trained on 80% of questions and evaluated on the held-out 20%. The first 300 questions are used only for wall-time/smoke-test measurements.

**Hyperparameters**

| Parameter | Value | Rationale |
|---|---|---|
| Samples M | 10 | Diminishing returns beyond 10 (Kuhn et al, Fig. 3b) |
| Temperature | 0.5 | Balances diversity vs accuracy |
| Max new tokens | 128 | Sufficient for TriviaQA dataset |


## 4. Results

TriviaQA `rc.nocontext`, M=10 samples, temp=0.5, Modal A10G GPU.

| Model | Acc | SE AUROC | SEP AUROC | PE AUROC | SEP gap | SEP best layer |
|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 70% | 0.772 | 0.742 | 0.273 | −0.030 | 31 |
| Meta-Llama-3.1-8B-Instruct | **75%** | **0.790** | **0.746** | 0.258 | −0.044 | 31 |
| Qwen2.5-1.5B-Instruct | 41% | 0.728 | 0.705 | 0.303 | −0.024 | 26 |

For context, Kuhn et al. report SE AUROC around 0.83 with OPT-30B, so the reproduced 7–8B results are directionally consistent despite using smaller instruction-tuned models.

SE and PE scored directly on N=10,000. SEP evaluated via 5-fold CV on the same split. SEP probes recover 94–97% of SE's AUROC with a single forward pass: no sampling, no NLI. Wall times measured on N=300 smoke-test runs on A10G; speedup is lower than the theoretical 10× because Modal autoscaling already parallelizes Kuhn's sampling.

More plots in [APPENDIX §A2](APPENDIX.md#a2-per-model-plots).

<p align="center"><img src="outputs/sep_probe_mistral-7b-instruct-v0.3_sep_plots.png" width="75%"></p>

*Mistral-7B · 5-fold CV · layer 31 · SE AUROC=0.772 · SEP AUROC=0.742 · PE AUROC=0.273. All three methods on one ROC: SE (blue) and SEP (green) both detect hallucination risk; PE (red) inverts below the diagonal.*


## 5. Interpretation of Why PE Fails

PE AUROC < 0.5 means PE systematically ranks correct answers as *more* uncertain than hallucinated ones — the opposite of useful.

PE fails because it measures lexical uncertainty, not semantic uncertainty. The inversion traces to two opposing forces. For a correct answer, the training data contains the same fact in many forms — "Edison," "Thomas Edison," "Edison (1879)," "the light bulb was invented by Edison" — so probability mass is spread across many valid phrasings. Temperature sampling draws from that spread distribution and produces lexically diverse samples, inflating PE. 

For a hallucinated answer, instruction tuning and RLHF train the model to sound confident and direct. Hedging is penalized. The model commits to the most plausible sounding entity and expresses it with a sharp probability peak. Sampling from a peaked distribution returns near identical outputs regardless of temperature, suppressing PE. The result: correct answers carry high PE, hallucinated answers carry low PE — the opposite of what a useful uncertainty signal requires. This ranking inversion is not noise; it is a structural artifact of instruction tuning. SE fixes it by collapsing lexical variants into meaning clusters before computing entropy.

In AUROC terms, this means $P(\mathrm{PE}(s_{\text{wrong}}) > \mathrm{PE}(s_{\text{correct}})) < 0.5$.

Kuhn et al. report PE AUROC slightly below SE AUROC (~0.79 vs ~0.83) rather than inverted, because they evaluate on OPT-30B, a base model without instruction tuning. On a base model the confident-hallucination artifact does not apply, so PE behaves as expected. Our results use instruction-tuned models (Mistral, Llama, Qwen), where the inversion is consistent and structural.

A corollary: flipping the PE label (treating low PE as the hallucination signal) recovers 0.727–0.742 AUROC across the three models — comparable to SEP and requiring no NLI calls. The catch is that knowing which direction to flip requires validation data, making it no cheaper to deploy than SE or SEP.

For production design details (request flow, latency budget, offline training) see [APPENDIX.md](APPENDIX.md).


## 6. Limitations 

- SE and SEP detect uncertainty, not guaranteed incorrectness.
- Confidently wrong answers remain a blind spot: if the model outputs the same wrong answer across all M samples, SE = 0 and it is not flagged.
- Requires logprob and hidden-state access; black-box APIs are not supported.

The most important extension is a retrieval-grounded verifier. For highrisk answers or high stakes domains, retrieve external evidence and check whether the generated answer is supported. This addresses the main SE/SEP blind spot by adding an external truth signal rather than relying only on model uncertainty.


## References

1. Kuhn, L., Gal, Y., & Farquhar, S. (2023). *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR 2023. [arXiv:2302.09664](https://arxiv.org/abs/2302.09664)
2. Kossen, J., Han, J., Razzak, M., Schut, L., Malik, S., & Gal, Y. (2024). *Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs.* **161 citations.** [arXiv:2406.15927](https://arxiv.org/abs/2406.15927)
3. He, P. et al. (2020). *DeBERTa: Decoding-Enhanced BERT with Disentangled Attention.* [arXiv:2006.03654](https://arxiv.org/abs/2006.03654)
4. Joshi, M. et al. (2017). *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.* ACL 2017. [arXiv:1705.03551](https://arxiv.org/abs/1705.03551)
5. Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature. [Nature](https://www.nature.com/articles/s41586-024-07421-0)
6. Chen, C., Liu, K., Chen, Z., Gu, Y., Wu, Y., Tao, M., Fu, Z., & Ye, J. (2024). *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection.* ICLR 2024. **356 citations.** [arXiv:2402.03744](https://arxiv.org/abs/2402.03744)
7. Park, J. W., & Cho, K. (2025). *Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling.* NeurIPS 2025. [Poster](https://neurips.cc/virtual/2025/loc/san-diego/poster/118777)
