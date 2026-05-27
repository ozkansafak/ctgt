# Appendix

## A1. SE worked example

*"Who invented the light bulb?"*

```
s1:  "Thomas Edison"              log_prob = -0.12 ┐
s2:  "Edison invented it in 1879" log_prob = -0.15 ├─ Cluster 1  p = 0.95
s3:  "The light bulb is Edison's" log_prob = -0.18 ┘
s10: "Nikola Tesla"               log_prob = -2.10  ── Cluster 2  p = 0.05

SE = -(0.95·log 0.95 + 0.05·log 0.05) = 0.20  ->  low uncertainty
```

Cluster 1 has high aggregate probability (three paraphrases of the same correct answer). Cluster 2 has low probability (one wrong answer). SE is low because the model is nearly certain about one meaning. If the model were split evenly across two meanings, SE would approach log(2) ≈ 0.69.

---

## A2. Per-model plots


Each model shows two figure pairs: **Kuhn SE** (ROC curve + SE distribution) and **Kossen SEP** (ROC curve + probe score distribution). All runs: TriviaQA `rc.nocontext`, N=10,000, M=10 samples, temp=0.5, Modal A10G.

---

### A2.1 Mistral-7B-Instruct-v0.3

<p align="center"><img src="outputs/sep_data_mistral-7b-instruct-v0.3_q10000_s10_t0.5_20260526_204939_plots.png" width="75%"></p>

*Kuhn SE · Mistral-7B · N=10,000 · SE AUROC=0.772 · PE AUROC=0.273 · Acc=70% · 75s per 300q. PE falls below the diagonal: instruction tuning makes the model generate confident-sounding answers regardless of correctness, so hallucinated answers carry low entropy and rank above correct ones. SE is unaffected: "JFK", "John Kennedy", "John F. Kennedy" all land in one cluster.*

<p align="center"><img src="outputs/sep_probe_mistral-7b-instruct-v0.3_sep_plots.png" width="75%"></p>

*Kossen SEP · Mistral-7B · N=10,000 · 5-fold CV · layer 31 · SEP AUROC=0.742 · gap=−0.030.*

---

### A2.2 Meta-Llama-3.1-8B-Instruct

<p align="center"><img src="outputs/sep_data_meta-llama-3.1-8b-instruct_q10000_s10_t0.5_20260526_211214_plots.png" width="75%"></p>

*Kuhn SE · Llama-3.1-8B · N=10,000 · SE AUROC=0.790 · PE AUROC=0.258 · Acc=75% · 108s per 300q. Highest accuracy and SE AUROC of all tested models. Same PE inversion: confident wrong answers rank below confident correct ones.*

<p align="center"><img src="outputs/sep_probe_meta-llama-3.1-8b-instruct_sep_plots.png" width="75%"></p>

*Kossen SEP · Llama-3.1-8B · N=10,000 · 5-fold CV · layer 31 · SEP AUROC=0.746 · gap=−0.044.*

---

### A2.3 Qwen2.5-1.5B-Instruct

<p align="center"><img src="outputs/sep_data_qwen2.5-1.5b-instruct_q10000_s10_t0.5_20260526_213050_plots.png" width="75%"></p>

*Kuhn SE · Qwen2.5-1.5B · N=10,000 · SE AUROC=0.728 · PE AUROC=0.303 · Acc=41% · 90s per 300q. Smallest model, lowest accuracy. SE AUROC is competitive despite 1.5B parameters.*

<p align="center"><img src="outputs/sep_probe_qwen2.5-1.5b-instruct_sep_plots.png" width="75%"></p>

*Kossen SEP · Qwen2.5-1.5B · N=10,000 · 5-fold CV · layer 26 · SEP AUROC=0.705 · gap=−0.024. Smallest probe gap of the three models.*

---

## A3. Experimental split

TriviaQA `rc.nocontext` (17,944 validation questions), shuffled with seed 42, M=10 samples per question, temp=0.5, Modal A10G GPU.

| | Benchmark split | Smoke-test / wall-time |
|---|---|---|
| Questions | 300–10,299 (N=10,000) | 0–299 (N=300) |
| SE / PE | Scored directly on all 10,000 | — |
| SEP | 5-fold CV on same 10,000 | — |
| Wall-time | — | Kuhn and SEP inference timed here |

The first 300 questions are disjoint from the benchmark split (same shuffle seed, no overlap).

---

## A4. Production Design

In production, SE is too expensive because it requires M generations plus NLI clustering. SEP is cheap: one prefill pass, one hidden-state slice, one matrix multiply. The probe can be served alongside the LLM and adds negligible compute relative to decoding.

### Request flow

```
                    ┌──────────────────────────┐
                    │    Incoming question, Q  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     LLM forward pass     │
                    │     (1 pass)             │
                    └────────────┬─────────────┘
                                 │ hidden states available
                    ┌────────────┴────────────┐
                    ▼                         ▼
         ┌───────────────────┐    ┌─────────────────────────┐
         │  Token generation │    │   Probe evaluation      │
         │  via LLM          │    │   h[best_layer][-1]     │
         │  inference        │    │   -> sigmoid(W * h + b) │
         └──────────┬────────┘    └───────────┬─────────────┘
                    │                         │
                    │      answer string A    │  P(uncertain) ∈ [0,1]
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │        Response            │
                    │  { answer: A,              │
                    │    hallucination_risk: p } │
                    └────────────────────────────┘
```

The probe runs at the **last input token position** — the model has attended to the full question but has not yet emitted a single output token. Slicing `hidden_states[best_layer][-1]` and running `sigmoid(W · h + b)` adds < 1 ms on top of the prefill. Token generation then proceeds from the same KV cache.

### Offline training (one-time)

1. **Collect training data:** run the full Kuhn pipeline on N ≥ 500 questions -> SE score per question. Takes ~75s per 300 questions on A10G.
2. **Binarize:** Otsu threshold γ* on SE scores -> 0/1 uncertain labels.
3. **Extract hidden states:** one forward pass per question, slice `hidden_states[layer][-1]` at every layer.
4. **Grid-search layers:** fit logistic regression per layer, pick best AUROC. For 7–8B models, layer 31 (of 32) wins; for Qwen-1.5B (28 layers), layer 26.
5. **Save probe:** `W` (d_model × 1) + `b` (scalar) + `best_layer` -> `sep_probe_<model>.pkl`.

Re-training is only needed when the base LLM changes.
