"""Subclass one or more typed lifecycle hooks; return the transformed value."""

from torch import Tensor, nn

from .types import QKV, AttentionContext


class AttentionHook(nn.Module):
    """Modules are registered: parameters, device transfers and eval all work.

    Hooks run in supplied order at each stage. after_projection transforms
    current Q/K/V before caching (e.g. positional encoding). before_attention
    transforms the current context without rewriting retained state.
    """

    def after_projection(self, qkv: QKV, positions: Tensor) -> QKV:
        return qkv

    def before_attention(self, context: AttentionContext, positions: Tensor) -> AttentionContext:
        return context

    def after_attention(self, attended: Tensor) -> Tensor:
        return attended

    def after_output(self, output: Tensor) -> Tensor:
        return output
