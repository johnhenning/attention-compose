"""Opaque, caller-owned state with cache storage independent of query context."""

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


@dataclass(frozen=True)
class QuantizedTensor:
    values: Tensor
    scales: Tensor

    @staticmethod
    def quantize(x: Tensor) -> "QuantizedTensor":
        if not x.is_floating_point() or x.ndim != 4 or x.shape[-1] == 0:
            raise ValueError("quantization requires floating [batch, heads, tokens, dim] tensors")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("quantization requires finite values")
        work = x.detach().float()
        if not bool(torch.isfinite(work).all()):
            raise ValueError("quantization requires values representable in FP32")
        scales = work.abs().amax(dim=-1, keepdim=True) / 127.0
        scales = scales.clamp_min(torch.finfo(torch.float32).tiny)
        values = (work / scales).round().clamp(-127, 127).to(torch.int8)
        return QuantizedTensor(values, scales)

    def dequantize(self, dtype: torch.dtype) -> Tensor:
        return (self.values.float() * self.scales).to(dtype)

    def select(self, indices: Tensor) -> "QuantizedTensor":
        return QuantizedTensor(
            self.values.index_select(2, indices), self.scales.index_select(2, indices)
        )

    def append(self, other: "QuantizedTensor") -> "QuantizedTensor":
        return QuantizedTensor(
            torch.cat([self.values, other.values], dim=2),
            torch.cat([self.scales, other.scales], dim=2),
        )


@dataclass(frozen=True)
class Int8KVState:
    k: QuantizedTensor
    v: QuantizedTensor
    positions: Tensor
    next_position: int
    dtype: torch.dtype


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


class Int8KVCacheManager(AttentionStateManager[Int8KVState]):
    def __init__(
        self, retention: RetentionPolicy | None = None, *, require_zero_start: bool = False
    ) -> None:
        self.retention = retention if retention is not None else FullRetention()
        self.require_zero_start = require_zero_start

    def next_position(self, state: Int8KVState | None) -> int:
        if state is not None and not isinstance(state, Int8KVState):
            raise ValueError("INT8 cache requires Int8KVState or None")
        return 0 if state is None else state.next_position

    def prepare(
        self, qkv: QKV, positions: Tensor, state: Int8KVState | None
    ) -> PreparedAttention[Int8KVState]:
        self.next_position(state)
        if state is None and self.require_zero_start and int(positions[0].item()) != 0:
            raise ValueError("this cache composition must start at position zero")
        if torch.is_grad_enabled():
            raise ValueError(
                "INT8 cache is inference-only; use torch.no_grad() or torch.inference_mode()"
            )
        _validate_positions(positions)
        k, v = QuantizedTensor.quantize(qkv.k), QuantizedTensor.quantize(qkv.v)
        all_positions = positions
        if state is not None:
            _validate_append(state.k.values, state.v.values, qkv, positions, state.next_position)
            if state.dtype != qkv.k.dtype:
                raise ValueError("cache dtype does not match input; start with state=None")
            k, v = state.k.append(k), state.v.append(v)
            all_positions = torch.cat([state.positions, positions])
        context = AttentionContext(
            k.dequantize(qkv.k.dtype), v.dequantize(qkv.v.dtype), all_positions
        )
        indices = self.retention.indices(all_positions)
        new_state = Int8KVState(
            k.select(indices),
            v.select(indices),
            all_positions.index_select(0, indices),
            int(positions[-1].item()) + 1,
            qkv.k.dtype,
        )
        return PreparedAttention(context, new_state)
