"""The same mask, on the transformers stack.

`processor.py` was written against llama.cpp because that is what enforces on a
quantised gguf, which is what a clinic can run. But the interpretability tooling
— Jacobian lenses, activation probes — needs bf16 safetensors and cannot read a
gguf, so the natural conclusion is that structural enforcement and semantic
monitoring must live in two processes on two copies of the model, with an
unverified assumption that the bf16 copy's internals describe the quantised
one's behaviour.

That conclusion is an artefact of the runtime choice, not of the mechanism. A
mask needs a hook on the logits, and `transformers` has the same one. Running
the mask here puts both layers on one model in one process — and specifically on
gemma-4-E4B, which is the variant that has a published lens *and* the variant a
clinic can deploy, and which llama.cpp cannot load at all.

The split still exists in production: quantise, keep the mask, drop the lens.
But it no longer has to exist during the experiment that compares them, which is
where the assumption would have done its damage.

Requires `pip install capability-kernel[hf]`.
"""

from __future__ import annotations

import numpy as np

from .compiler import ARM, CompiledSurface
from .processor import CapabilityProcessor, Telemetry


class HFCapabilityProcessor:
    """`transformers.LogitsProcessor` wrapping the same enforcement logic.

    Not a reimplementation. The trie walk, the phase control, the slot handling
    and the rejected-mass telemetry are all :class:`CapabilityProcessor`; this
    converts tensors and hands over. Two copies of a security-critical decision
    procedure would drift, and the one that drifted would be the one nobody ran.
    """

    def __init__(self, surface: CompiledSurface, tokenizer, *,
                 arm: str = ARM, enabled: set[int] | None = None,
                 telemetry: Telemetry | None = None) -> None:
        self.tokenizer = tokenizer
        self.inner = CapabilityProcessor(
            surface, arm,
            detokenize=lambda ids: tokenizer.decode(list(ids)),
            close_tokens=tokenizer.encode("\n", add_special_tokens=False),
            enabled=enabled, telemetry=telemetry,
        )

    @property
    def telemetry(self) -> Telemetry:
        return self.inner.telemetry

    @property
    def desynchronised(self) -> str | None:
        return self.inner.desynchronised

    def reset(self) -> None:
        self.inner.reset()

    def __call__(self, input_ids, scores):
        """transformers passes batched tensors; the walk is per-sequence.

        Only batch size 1 is supported, and that is checked rather than assumed.
        Masking a batch against one sequence's trie position would silently
        constrain every other row to the wrong step — output that looks fine and
        is enforced against the wrong state.
        """
        import torch

        if input_ids.shape[0] != 1:
            raise ValueError(
                f"batch size {input_ids.shape[0]}: the mask tracks one walk, so "
                f"a batch would be enforced against another sequence's position"
            )

        row = scores[0].detach().to(torch.float32).cpu().numpy()
        masked = self.inner(input_ids[0].tolist(), row)

        out = torch.from_numpy(np.asarray(masked)).to(scores.device, scores.dtype)
        return out.unsqueeze(0)
