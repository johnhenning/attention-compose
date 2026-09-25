"""Tensor records. Q/K/V have shape [batch, heads, tokens, head_dim]."""

from dataclasses import dataclass
from typing import Generic

from torch import Tensor
from typing_extensions import TypeVar

StateT = TypeVar("StateT", default=None)


@dataclass(frozen=True)
class QKV:
    q: Tensor
    k: Tensor
    v: Tensor


@dataclass(frozen=True)
class AttentionContext:
    k: Tensor
    v: Tensor
    positions: Tensor
    allowed: Tensor | None = None


@dataclass(frozen=True)
class AttentionOutput(Generic[StateT]):
    output: Tensor
    state: StateT | None


@dataclass(frozen=True)
class PreparedAttention(Generic[StateT]):
    """Current context plus next retained state; their lengths may differ."""

    context: AttentionContext
    state: StateT | None
