# Hallucination Detection via Semantic Entropy

Full technical report: [DESIGN.md](DESIGN.md)

Implements and benchmarks two papers on LLM hallucination detection: Kuhn/Farquhar Semantic Entropy (logprobs) and Kossen Semantic Entropy Probes (hidden states). A third method, Chen INSIDE (hidden states read+write), is documented but not yet implemented.


---

## Glossary

| Term | Definition |
|---|---|
| SE | Semantic Entropy: entropy over meaning clusters, not token sequences |
| PE | Predictive Entropy: entropy over raw token sequences (baseline) |
| SEP | Semantic Entropy Probe: logistic regression on last token embeddings |
| NLI | Natural Language Inference: entailment / neutral / contradiction |
| DeBERTa | NLI model used for clustering (`cross-encoder/nli-deberta-v3-large`) |
| AUROC | Separation quality: 0.5 = random, 1.0 = perfect |
| RougeL | String overlap metric, used to judge answer correctness (threshold 0.3) |
| TriviaQA | Evaluation dataset, closed-book (model answers from memory) |

---

## Results

TriviaQA `rc.nocontext`, N=10,000, M=10 samples, temp=0.5, Modal A10G. Train and eval sets are disjoint.

| Model | Acc | Kuhn SE | Kossen SEP | Wall time (Kuhn / SEP) |
|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 70% | 0.772 | 0.742 | 75s / 50s |
| Meta-Llama-3.1-8B-Instruct | 75% | 0.790 | 0.746 | 108s / 70s |
| Qwen2.5-1.5B-Instruct | 41% | 0.728 | 0.705 | 90s / 31s |

SEP probes recover 94-97% of SE AUROC using a single forward pass. For explanation of PE AUROC < 0.5, see DESIGN.md.

---

## Project structure

```
modal_app.py      # Modal GPU service and all benchmark entrypoints
train_probe.py    # Train SEP logistic regression probe
results.py        # Plot ROC curves and SE distributions
outputs/          # Benchmark JSONs and plots

src/ctgt/
  sampling.py     # LLM generation and hidden state extraction
  kuhn_2024/      # SE: entailment clustering and entropy scoring
  kossen_2024/    # SEP: probe training and inference
  inside_2024/    # INSIDE stub (not yet implemented)
```

---

## Setup

```bash
git clone <repo> && cd ctgt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
modal setup
```

---

## Usage

```bash
# Single question
modal run modal_app.py::app.analyze --question "Who invented the telephone?"

# Benchmark (Kuhn SE)
modal run modal_app.py::app.benchmark --n-questions 300 --llm-model mistralai/Mistral-7B-Instruct-v0.3

# Collect SEP training data then train probe
modal run modal_app.py::app.collect_sep_training_data --n-questions 300 --llm-model mistralai/Mistral-7B-Instruct-v0.3
python train_probe.py outputs/sep_data_<run>.json

# Plot results
python results.py outputs/<run>.json
```

---

## References

- Farquhar, Kossen, Kuhn, Gal (2024). *Detecting Hallucinations in LLMs Using Semantic Entropy.* Nature. [arXiv:2303.08896](https://arxiv.org/abs/2303.08896)
- Kossen, Han, Razzak, Schut, Malik, Gal (2024). *Semantic Entropy Probes.* [arXiv:2406.15927](https://arxiv.org/abs/2406.15927)
- Chen et al. (2024). *INSIDE: LLMs Internal States Retain the Power of Hallucination Detection.* ICLR 2024. [arXiv:2402.03744](https://arxiv.org/abs/2402.03744)
- Park, Cho (2025). *Efficient Semantic Uncertainty Quantification via Diversity-Steered Sampling.* NeurIPS 2025.
