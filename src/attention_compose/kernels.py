"""Reference and SDPA implementations with identical explicit-mask semantics."""

from abc import ABC, abstractmethod
from math import sqrt

import torch
from torch import Tensor
from torch.nn import functional as F

from .types import AttentionContext


def expand_kv_heads(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    if k.shape[1] <= 0 or q.shape[1] % k.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    groups = q.shape[1] // k.shape[1]
    return k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)


def attention_mask(q: Tensor, context: AttentionContext, positions: Tensor, causal: bool) -> Tensor:
    shape = (q.shape[-2], context.k.shape[-2])
    allowed = torch.ones(shape, dtype=torch.bool, device=q.device)
    if causal:
        allowed = allowed & (context.positions[None, :] <= positions[:, None])
    if context.allowed is not None:
        if context.allowed.dtype != torch.bool or context.allowed.shape != shape:
            raise ValueError("allowed must be a Boolean [query_length, key_length] mask")
        allowed = allowed & context.allowed
    return allowed[None, None, :, :]


class AttentionKernel(ABC):
    @abstractmethod
    def __call__(
        self,
        q: Tensor,
        context: AttentionContext,
        query_positions: Tensor,
        *,
        causal: bool = True,
        dropout_p: float = 0.0,
    ) -> Tensor: ...


class ReferenceAttentionKernel(AttentionKernel):
    def __call__(
        self,
        q: Tensor,
        context: AttentionContext,
        query_positions: Tensor,
        *,
        causal: bool = True,
        dropout_p: float = 0.0,
    ) -> Tensor:
        k, v = expand_kv_heads(q, context.k, context.v)
        dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        scores = (q.to(dtype) @ k.to(dtype).transpose(-2, -1)) / sqrt(q.shape[-1])
        allowed = attention_mask(q, context, query_positions, causal)
        scores = scores.masked_fill(~allowed, float("-inf"))
        # Avoid NaNs and NaN gradients from softmax([-inf, ...]).
        scores = torch.where(allowed.any(dim=-1, keepdim=True), scores, 0.0)
        probabilities = scores.softmax(dim=-1).masked_fill(~allowed, 0.0)
        probabilities = F.dropout(probabilities, p=dropout_p, training=dropout_p > 0)
        return (probabilities @ v.to(dtype)).to(q.dtype)


class SDPAAttentionKernel(AttentionKernel):
    def __call__(
        self,
        q: Tensor,
        context: AttentionContext,
        query_positions: Tensor,
        *,
        causal: bool = True,
        dropout_p: float = 0.0,
    ) -> Tensor:
        k, v = expand_kv_heads(q, context.k, context.v)
        allowed = attention_mask(q, context, query_positions, causal)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=dropout_p, is_causal=False
        )
