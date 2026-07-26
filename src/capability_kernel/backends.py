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

import json
import re
from typing import Protocol


#: Gemma 4 wraps a call in `<|tool_call>` … `<tool_call|>` and quotes each
#: argument with `<|"|>`, which is neither the OpenAI shape nor JSON. Kept as a
#: pattern rather than hardcoded into one backend because both runtimes hit it:
#: llama.cpp leaves it in content, and transformers hands back whatever the chat
#: template produced.
_GEMMA_CALL = re.compile(r"call:\s*(\w+)\s*\{(.*?)\}", re.S)
_GEMMA_ARG = re.compile(r"(\w+)\s*:\s*<\|\"\|>(.*?)<\|\"\|>", re.S)


def parse_tool_calls(text: str) -> list[dict]:
    """Recover tool calls a runtime did not normalise.

    Tries the model-specific form first and falls back to balanced JSON, which
    is what most templates emit. Returns the OpenAI shape either way, so the
    agent loop never learns which model it is talking to.
    """
    calls = []
    for name, body in _GEMMA_CALL.findall(text):
        args = {k: v.strip() for k, v in _GEMMA_ARG.findall(body)}
        if not args:
            # Some turns quote nothing. `key: value` per comma is the only
            # other shape observed, and guessing further would invent calls.
            args = {k.strip(): v.strip()
                    for k, _, v in (p.partition(":") for p in body.split(","))
                    if k.strip() and v.strip()}
        calls.append({"function": {"name": name, "arguments": json.dumps(args)}})

    if calls:
        return calls

    for chunk in _json_objects(text):
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or obj.get("function")
        if isinstance(name, str) and name:
            calls.append({"function": {
                "name": name,
                "arguments": json.dumps(obj.get("arguments")
                                        or obj.get("parameters") or {}),
            }})
    return calls


class Backend(Protocol):
    """What the chat loop needs from a runtime."""

    def tokenize(self, text: str) -> list[int]: ...
    def detokenize(self, tokens: list[int]) -> str: ...

    def generate(self, prompt: str, *, processor, max_tokens: int,
                 temperature: float, stop: list[str]) -> str:
        """Continue `prompt`. `processor` is None for the unmasked arm.

        Used by the mask experiment, which needs a text protocol because a
        logits processor has nowhere to attach in a chat-completion API.
        """
        ...

    def chat(self, messages: list[dict], tools: list[dict], *,
             temperature: float, max_tokens: int) -> dict:
        """One turn of native tool calling.

        The production path. Returns ``{"content": str, "tool_calls": [...]}``
        in the OpenAI shape, because that is what both runtimes emit and what
        the agent loop reads.
        """
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


    def chat(self, messages, tools, *, temperature, max_tokens) -> dict:
        out = self.llama.create_chat_completion(
            messages=messages, tools=tools or None,
            tool_choice="auto" if tools else None,
            temperature=temperature, max_tokens=max_tokens)
        message = (out.get("choices") or [{}])[0].get("message") or {}
        content = message.get("content") or ""
        calls = message.get("tool_calls") or []

        # llama.cpp normalises the formats it knows and returns the rest as
        # content. Gemma 4 emits its own, so a correct call arrives looking like
        # a refusal — measured: the model chose a valid combination on the
        # dictated tooth and the turn was scored as producing nothing.
        if not calls and content:
            calls = parse_tool_calls(content)
            if calls:
                content = ""

        return {"content": content, "tool_calls": calls}


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

    def chat(self, messages, tools, *, temperature, max_tokens) -> dict:
        """Native tool calling through the chat template.

        transformers renders tools into the prompt and the model emits a call in
        whatever form its template defines, so the parsing is model-specific in
        a way llama.cpp's server hides. Gemma emits a JSON object; anything else
        is returned as content and the agent loop treats it as prose, which is
        the correct outcome for a model that declined to call a tool.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=False,
            add_generation_prompt=True)
        text = self.generate(prompt, processor=None, max_tokens=max_tokens,
                             temperature=temperature, stop=[])

        calls = parse_tool_calls(text)
        return {"content": "" if calls else text.strip(), "tool_calls": calls}


def _json_objects(text: str):
    """Balanced top-level ``{...}`` spans, in order.

    A regex cannot do this — arguments nest — and a model that wrote prose
    around its call would defeat a strict json.loads of the whole reply.
    """
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None
