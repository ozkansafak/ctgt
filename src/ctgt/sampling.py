"""LLM sampling with per-sequence log-probability computation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Sample:
    text: str
    log_prob: float             # sum of per-token log probs (raw)
    log_prob_normalized: float  # log_prob / num_tokens


class LLMSampler:
    """
    Generates N stochastic completions from an instruction-tuned LLM and
    returns each completion paired with its length-normalized log-probability.

    Uses a single batched model.generate() call for efficiency.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # small default; swap for larger when ready

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _build_prompt(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def sample(
        self,
        question: str,
        n: int = 10,
        temperature: float = 0.5,
        max_new_tokens: int = 128,
    ) -> list[Sample]:
        """
        Draw n completions and return each with its length-normalized log-prob.

        Temperature 0.5 balances diversity and accuracy per Kuhn et al. (2023).
        """
        prompt = self._build_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        # Replicate the single prompt n times for batched generation
        expanded_ids = inputs["input_ids"].expand(n, -1)
        expanded_mask = inputs["attention_mask"].expand(n, -1)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=expanded_ids,
                attention_mask=expanded_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # output.scores: tuple of (n, vocab_size) tensors, one per generated step
        # output.sequences: (n, input_len + generated_len)
        samples = []
        for i in range(n):
            generated_ids = output.sequences[i, input_len:]

            # Trim padding / tokens after first EOS
            eos_pos = (generated_ids == self.tokenizer.eos_token_id).nonzero()
            if len(eos_pos) > 0:
                generated_ids = generated_ids[: eos_pos[0].item()]

            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            n_tokens = len(generated_ids)

            log_prob = 0.0
            for t, score in enumerate(output.scores):
                if t >= n_tokens:
                    break
                token_id = output.sequences[i, input_len + t].item()
                log_prob += F.log_softmax(score[i], dim=-1)[token_id].item()

            samples.append(
                Sample(
                    text=text,
                    log_prob=log_prob,
                    log_prob_normalized=log_prob / max(n_tokens, 1),
                )
            )

        return samples
