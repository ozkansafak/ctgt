"""
INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection
Chen et al., ICLR 2024  —  https://arxiv.org/abs/2402.03744

NOT YET IMPLEMENTED — documented here as a reference for future work.
See DESIGN.md "Future Work" for context.

---

Method summary
--------------
INSIDE proposes two complementary techniques that operate on LLM hidden states
rather than output token probabilities:

1. EigenScore
   Captures semantic *consistency* of K generated responses in dense embedding
   space by examining the covariance matrix of their hidden-state representations.

   Steps:
     a. Generate K=10 responses for a question x.
     b. For each response, extract the last-token embedding at the middle
        transformer layer (layer ≈ L/2; e.g. layer 17 for LLaMA-7B).
     c. Build a (K × d) embedding matrix Z.
     d. Compute the regularised covariance matrix:  Σ = Z^T · J_d · Z + α·I_K
        where J_d is a centering matrix and α=0.001 is a small regulariser.
     e. EigenScore = (1/K) · Σ_i log(λ_i)  for eigenvalues λ_i of Σ.

   Interpretation: When responses are similar (low hallucination risk), the
   covariance matrix is nearly rank-1 — eigenvalues concentrate on the first
   component. When responses diverge (high hallucination risk), eigenvalues
   spread, increasing the EigenScore. This is semantically richer than token-
   level entropy because it operates in the model's compressed representation
   space rather than over the discrete vocabulary.

   Reported AUROC on TriviaQA/LLaMA-7B: 82.7–82.9 % vs ~65 % for Semantic
   Entropy (CoQA comparison), a substantial improvement.

2. Feature Clipping
   A *test-time activation intervention* that truncates extreme values in the
   penultimate layer's hidden states before the final projection, reducing
   overconfident generations.

   Steps:
     a. Offline: maintain a memory bank of N=3000 token embeddings from a
        calibration set.
     b. Set thresholds [h_min, h_max] at the bottom / top p=0.2 percentile
        of the memory bank distribution.
     c. At inference: for each token's hidden state h in the penultimate layer,
        clip dimension-wise:
            FC(h) = clip(h, h_min, h_max)

   Crucially, this requires a PyTorch **forward hook** — not just reading
   hidden states — because the activations must be *modified* before the next
   layer computes. This is distinct from our current approach of reading via
   output_hidden_states=True (read-only).

   Effect: reduces overconfident generations; improves calibration without
   retraining or fine-tuning the LLM.

---

Key architectural difference from Kuhn + Kossen
-------------------------------------------------

Method               Weights access      What it does with internals
---                  ---                 ---
Farquhar 2024        Logprobs only       Samples M outputs → NLI cluster → entropy
Kossen SEPs 2024     Hidden states (r)   Linear probe on frozen last-token activation
Chen INSIDE 2024     Hidden states (r+w) EigenScore on mid-layer + forward-hook clipping

Kossen requires read-only access; INSIDE requires a forward hook to *write*
modified activations back into the computation graph at inference time.

---

Layer and token position choices
---------------------------------
- EigenScore: middle layer (≈ L/2). Ablation shows middle layers outperform
  both shallow and final layers for long-form generation. This differs from
  Kossen's finding that final layers are best for short-form QA.
- Feature Clipping: penultimate layer.
- Token position: last generated token of each response (not the pre-generation
  token used by Kossen's TBG approach).

---

Implementation notes for when this is built
--------------------------------------------
1. Forward hook registration:

    hidden_cache = {}

    def hook_fn(module, input, output):
        hidden_cache['mid'] = output[0]   # shape: (batch, seq_len, d)

    handle = model.model.layers[mid_layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(**inputs)
    handle.remove()
    embedding = hidden_cache['mid'][0, -1, :]   # last token, middle layer

2. Feature clipping hook (modifies activations in-place):

    def clip_hook_fn(module, input, output):
        hidden = output[0]
        clipped = hidden.clamp(min=h_min, max=h_max)
        return (clipped,) + output[1:]

    handle = model.model.layers[penultimate_layer].register_forward_hook(clip_hook_fn)

3. EigenScore computation is cheap (CPU, K×K matrix where K=10), so it can
   run locally after extracting embeddings from the GPU.

---

References
----------
Chen, C., Liu, K., Chen, Z., Gu, Y., Wu, Y., Tao, M., Fu, Z., & Ye, J. (2024).
INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection.
ICLR 2024. https://arxiv.org/abs/2402.03744
"""
from __future__ import annotations

# This module is intentionally left unimplemented.
# It documents the INSIDE method for reference and future implementation.
# See the module docstring above for full technical details.


class INSIDEDetector:
    """Placeholder for the INSIDE hallucination detector (Chen et al., ICLR 2024).

    Not implemented yet. See module docstring for the full technical specification.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "INSIDEDetector is not yet implemented. "
            "See src/ctgt/inside.py for the full design spec."
        )
