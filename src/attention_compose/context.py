"""Structural attention masks. Causality is composed by the kernel."""

from abc import ABC, abstractmethod
from dataclasses import replace

from torch import Tensor

from .types import AttentionContext


class ContextPolicy(ABC):
    @abstractmethod
    def apply(self, context: AttentionContext, query_positions: Tensor) -> AttentionContext: ...


def _restrict(context: AttentionContext, allowed: Tensor) -> AttentionContext:
    if context.allowed is not None:
        allowed = allowed & context.allowed
    return replace(context, allowed=allowed)


class FullContext(ContextPolicy):
    def apply(self, context: AttentionContext, query_positions: Tensor) -> AttentionContext:
        return context


class LocalContext(ContextPolicy):
    """Allow distance < window_size; causal mode keeps current + previous W-1."""

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size

    def apply(self, context: AttentionContext, query_positions: Tensor) -> AttentionContext:
        distance = (query_positions[:, None] - context.positions[None, :]).abs()
        return _restrict(context, distance < self.window_size)


class LocalGlobalContext(LocalContext):
    """Local attention plus global keys; optionally global queries see all keys.

    Initial globals are absolute positions [0, num_global_tokens). Explicit
    global_positions can identify other keys. This is a dense masked kernel.
    """

    def __init__(
        self,
        window_size: int,
        num_global_tokens: int = 0,
        *,
        global_positions: tuple[int, ...] = (),
        global_queries: bool = False,
    ) -> None:
        super().__init__(window_size)
        if num_global_tokens < 0 or any(p < 0 for p in global_positions):
            raise ValueError("global positions/count must be nonnegative")
        self.num_global_tokens = num_global_tokens
        self.global_positions = global_positions
        self.global_queries = global_queries

    def _global(self, positions: Tensor) -> Tensor:
        result = positions < self.num_global_tokens
        for position in self.global_positions:
            result = result | (positions == position)
        return result

    def apply(self, context: AttentionContext, query_positions: Tensor) -> AttentionContext:
        distance = (query_positions[:, None] - context.positions[None, :]).abs()
        allowed = (distance < self.window_size) | self._global(context.positions)[None, :]
        if self.global_queries:
            allowed = allowed | self._global(query_positions)[:, None]
        return _restrict(context, allowed)
