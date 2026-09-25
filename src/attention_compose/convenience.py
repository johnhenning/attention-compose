from collections.abc import Sequence
from typing import TypedDict, Unpack

from .attention import Attention
from .context import FullContext, LocalContext, LocalGlobalContext
from .hooks import AttentionHook
from .kernels import AttentionKernel
from .projections import GroupedQueryProjection, MultiHeadProjection, MultiQueryProjection
from .retention import AttentionSinkRetention, RetentionPolicy, SlidingWindowRetention
from .state import DenseKVCacheManager, DenseKVState


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


class LocalAttention(Attention):
    def __init__(
        self, d_model: int, num_heads: int, window_size: int, **options: Unpack[AttentionOptions]
    ) -> None:
        super().__init__(
            MultiHeadProjection(d_model, num_heads),
            context_policy=LocalContext(window_size),
            **options,
        )


class SparseAttention(Attention):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        *,
        num_global_tokens: int = 1,
        global_positions: tuple[int, ...] = (),
        global_queries: bool = False,
        **options: Unpack[AttentionOptions],
    ) -> None:
        super().__init__(
            MultiHeadProjection(d_model, num_heads),
            context_policy=LocalGlobalContext(
                window_size,
                num_global_tokens,
                global_positions=global_positions,
                global_queries=global_queries,
            ),
            **options,
        )


class CachedAttention(Attention[DenseKVState]):
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        *,
        num_kv_heads: int | None = None,
        retention: RetentionPolicy | None = None,
        detach: bool = False,
        **options: Unpack[AttentionOptions],
    ) -> None:
        super().__init__(
            GroupedQueryProjection(
                d_model, num_query_heads, num_query_heads if num_kv_heads is None else num_kv_heads
            ),
            DenseKVCacheManager(retention, detach=detach),
            FullContext(),
            **options,
        )


class SlidingWindowAttention(Attention[DenseKVState]):
    def __init__(
        self, d_model: int, num_heads: int, window_size: int, **options: Unpack[AttentionOptions]
    ) -> None:
        if options.get("causal", True) is False:
            raise ValueError("sliding cached attention requires causal=True")
        super().__init__(
            MultiHeadProjection(d_model, num_heads),
            DenseKVCacheManager(SlidingWindowRetention(window_size)),
            LocalContext(window_size),
            **options,
        )


class AttentionSinkAttention(Attention[DenseKVState]):
    """A total budget of window_size, including num_sink_tokens initial tokens.

    Start at position zero so retained initial tokens and global keys coincide.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        num_sink_tokens: int = 4,
        **options: Unpack[AttentionOptions],
    ) -> None:
        if options.get("causal", True) is False:
            raise ValueError("sink cached attention requires causal=True")
        super().__init__(
            MultiHeadProjection(d_model, num_heads),
            DenseKVCacheManager(
                AttentionSinkRetention(window_size, num_sink_tokens), require_zero_start=True
            ),
            LocalGlobalContext(window_size - num_sink_tokens, num_sink_tokens),
            **options,
        )
