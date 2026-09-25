"""Compose projections, state managers, policies, kernels, and hooks."""

from .attention import Attention
from .config import AttentionConfig, BuiltAttention, CacheKind, build_attention
from .context import ContextPolicy, FullContext, LocalContext, LocalGlobalContext
from .convenience import (
    AttentionSinkAttention,
    CachedAttention,
    GroupedQueryAttention,
    LocalAttention,
    MultiHeadAttention,
    MultiQueryAttention,
    QuantizedCachedAttention,
    SlidingWindowAttention,
    SparseAttention,
)
from .hooks import AttentionHook
from .kernels import AttentionKernel, ReferenceAttentionKernel, SDPAAttentionKernel, expand_kv_heads
from .projections import (
    AttentionProjection,
    GroupedQueryProjection,
    MultiHeadProjection,
    MultiQueryProjection,
)
from .retention import (
    AttentionSinkRetention,
    FullRetention,
    RetentionPolicy,
    SlidingWindowRetention,
)
from .state import (
    AttentionStateManager,
    DenseKVCacheManager,
    DenseKVState,
    Int8KVCacheManager,
    Int8KVState,
    QuantizedTensor,
    StatelessManager,
)
from .types import QKV, AttentionContext, AttentionOutput, PreparedAttention

__all__ = [
    "QKV",
    "Attention",
    "AttentionConfig",
    "AttentionContext",
    "AttentionHook",
    "AttentionKernel",
    "AttentionOutput",
    "AttentionProjection",
    "AttentionSinkAttention",
    "AttentionSinkRetention",
    "AttentionStateManager",
    "BuiltAttention",
    "CacheKind",
    "CachedAttention",
    "ContextPolicy",
    "DenseKVCacheManager",
    "DenseKVState",
    "FullContext",
    "FullRetention",
    "GroupedQueryAttention",
    "GroupedQueryProjection",
    "Int8KVCacheManager",
    "Int8KVState",
    "LocalAttention",
    "LocalContext",
    "LocalGlobalContext",
    "MultiHeadAttention",
    "MultiHeadProjection",
    "MultiQueryAttention",
    "MultiQueryProjection",
    "PreparedAttention",
    "QuantizedCachedAttention",
    "QuantizedTensor",
    "ReferenceAttentionKernel",
    "RetentionPolicy",
    "SDPAAttentionKernel",
    "SlidingWindowAttention",
    "SlidingWindowRetention",
    "SparseAttention",
    "StatelessManager",
    "build_attention",
    "expand_kv_heads",
]
