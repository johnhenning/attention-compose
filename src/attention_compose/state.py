from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic

import torch
from torch import Tensor

from .retention import FullRetention, RetentionPolicy
from .types import QKV, AttentionContext, PreparedAttention, StateT


@dataclass(frozen=True)
class DenseKVState:
    k: Tensor
    v: Tensor
    positions: Tensor
    next_position: int


class AttentionStateManager(ABC, Generic[StateT]):
    def next_position(self, state: StateT | None) -> int:
        return 0

    @abstractmethod
    def prepare(
        self, qkv: QKV, positions: Tensor, state: StateT | None
    ) -> PreparedAttention[StateT]:
        """Return the current context and next state without mutating previous state."""
        ...


class StatelessManager(AttentionStateManager[None]):
    def next_position(self, state: None) -> int:
        if state is not None:
            raise ValueError("stateless attention requires state=None")
        return 0

    def prepare(self, qkv: QKV, positions: Tensor, state: None) -> PreparedAttention[None]:
        self.next_position(state)
        return PreparedAttention(AttentionContext(qkv.k, qkv.v, positions), None)


def _validate_append(k: Tensor, v: Tensor, qkv: QKV, positions: Tensor, next_position: int) -> None:
    if k.ndim != 4 or v.shape != k.shape:
        raise ValueError("cached KV shape is invalid")
    if k.shape[:2] != qkv.k.shape[:2] or k.shape[-1] != qkv.k.shape[-1]:
        raise ValueError("cache batch/head shape does not match input; start with state=None")
    if k.device != qkv.k.device or v.device != qkv.v.device:
        raise ValueError("cache device does not match input; start with state=None")
    if int(positions[0].item()) != next_position:
        raise ValueError("positions must continue at state.next_position")


def _validate_positions(positions: Tensor) -> None:
    if positions.numel() > 1 and not bool(((positions[1:] - positions[:-1]) == 1).all()):
        raise ValueError("cached positions must be contiguous")


class DenseKVCacheManager(AttentionStateManager[DenseKVState]):
    def __init__(
        self,
        retention: RetentionPolicy | None = None,
        *,
        detach: bool = False,
        require_zero_start: bool = False,
    ) -> None:
        self.retention = retention if retention is not None else FullRetention()
        self.detach = detach
        self.require_zero_start = require_zero_start

    def next_position(self, state: DenseKVState | None) -> int:
        if state is not None and not isinstance(state, DenseKVState):
            raise ValueError("dense cache requires DenseKVState or None")
        return 0 if state is None else state.next_position

    def prepare(
        self, qkv: QKV, positions: Tensor, state: DenseKVState | None
    ) -> PreparedAttention[DenseKVState]:
        self.next_position(state)
        if state is None and self.require_zero_start and int(positions[0].item()) != 0:
            raise ValueError("this cache composition must start at position zero")
        _validate_positions(positions)
        k, v = qkv.k, qkv.v
        all_positions = positions
        if state is not None:
            _validate_append(state.k, state.v, qkv, positions, state.next_position)
            if state.k.dtype != k.dtype or state.v.dtype != v.dtype:
                raise ValueError("cache dtype does not match input; start with state=None")
            k, v = torch.cat([state.k, k], dim=2), torch.cat([state.v, v], dim=2)
            all_positions = torch.cat([state.positions, positions])
        # Select next storage only AFTER making the complete current context.
        context = AttentionContext(k, v, all_positions)
        indices = self.retention.indices(all_positions)
        kept_k, kept_v = k.index_select(2, indices), v.index_select(2, indices)
        if self.detach:
            kept_k, kept_v = kept_k.detach(), kept_v.detach()
        new_state = DenseKVState(
            kept_k, kept_v, all_positions.index_select(0, indices), int(positions[-1].item()) + 1
        )
        return PreparedAttention(context, new_state)
