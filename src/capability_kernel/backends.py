"""Where generation happens, so the chat loop does not care.

Two runtimes, and which one you can use is decided by the model rather than by
preference. `gemma-4-E4B` is the variant a clinic workstation can run and
llama.cpp refuses to load it — it reports the right architecture and then 720 of
an expected 2131 tensors, because the MatFormer nesting is not something that
build understands. `gemma4:12b` loads there and is unusable as a chat model.

That split was allowed to leak into the experiments for too long: the masked arm
ran on one model through one runtime and the unmasked arm on another, which
makes any difference between them a difference of *something*, not of the mask.
The comparison that answers whether enforcement helps has to hold the model
fixed and vary only enforcement, and holding gemma-4-E4B fixed means both arms
run here.
"""

from __future__ import annotations

from typing import Protocol


class Backend(Protocol):
    """What the chat loop needs from a runtime."""

    def tokenize(self, text: str) -> list[int]: ...
    def detokenize(self, tokens: list[int]) -> str: ...

    def generate(self, prompt: str, *, processor, max_tokens: int,
                 temperature: float, stop: list[str]) -> str:
        """Continue `prompt`. `processor` is None for the unmasked arm."""
        ...


class LlamaBackend:
    """llama.cpp on a quantised gguf — the deployment path."""

    def __init__(self, llama) -> None:
        self.llama = llama

    def tokenize(self, text: str) -> list[int]:
        return self.llama.tokenize(text.encode(), add_bos=False, special=False)

    def detokenize(self, tokens: list[int]) -> str:
        return self.llama.detokenize(list(tokens)).decode("utf-8", "replace")

    def generate(self, prompt, *, processor, max_tokens, temperature, stop) -> str:
        from llama_cpp import LogitsProcessorList

        out = self.llama(prompt, max_tokens=max_tokens, temperature=temperature,
                         stop=stop,
                         logits_processor=(LogitsProcessorList([processor])
                                           if processor is not None else None))
        return out["choices"][0]["text"]


class HFBackend:
    """transformers on bf16 weights — the only path for gemma-4-E4B.

    Stop sequences are applied after generation rather than during it. Doing it
    properly needs a StoppingCriteria that decodes on every step, and the cost of
    that is real while the benefit here is not: the loop parses the first action
    out of the text and ignores the rest either way.
    """

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(list(tokens))

    def generate(self, prompt, *, processor, max_tokens, temperature, stop) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        n_prompt = ids.input_ids.shape[1]

        if processor is not None:
            # The prompt may end on the arming word to force enforcement from
            # the first generated token. The processor has to be told where
            # generation begins or it will not see it.
            processor.inner.prompt_len = n_prompt if not prompt.rstrip().endswith("ACTION") \
                else len(self.tokenize(prompt[:prompt.rstrip().rfind("ACTION")]))

        out = self.model.generate(
            **ids, max_new_tokens=max_tokens,
            do_sample=temperature > 0, temperature=temperature or None,
            pad_token_id=self.tokenizer.eos_token_id,
            logits_processor=[processor] if processor is not None else None)

        text = self.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True)
        for marker in stop:
            if marker in text:
                text = text.split(marker)[0]
        return text
