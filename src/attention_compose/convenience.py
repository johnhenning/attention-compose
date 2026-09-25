from collections.abc import Sequence
from typing import TypedDict, Unpack

from .attention import Attention
from .hooks import AttentionHook
from .kernels import AttentionKernel
from .projections import GroupedQueryProjection, MultiHeadProjection, MultiQueryProjection


class AttentionOptions(TypedDict, total=False):
    kernel: AttentionKernel
    hooks: Sequence[AttentionHook]
    causal: bool
    dropout: float


class MultiHeadAttention(Attention):
    def __init__(self, d_model: int, num_heads: int, **options: Unpack[AttentionOptions]) -> None:
        super().__init__(MultiHeadProjection(d_model, num_heads), **options)


class GroupedQueryAttention(Attention):
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        **options: Unpack[AttentionOptions],
    ) -> None:
        super().__init__(GroupedQueryProjection(d_model, num_query_heads, num_kv_heads), **options)


class MultiQueryAttention(Attention):
    def __init__(
        self, d_model: int, num_query_heads: int, **options: Unpack[AttentionOptions]
    ) -> None:
        super().__init__(MultiQueryProjection(d_model, num_query_heads), **options)
