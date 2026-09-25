"""A validated, serializable configuration for dynamic construction."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from .attention import Attention
from .context import ContextPolicy, FullContext, LocalContext, LocalGlobalContext
from .hooks import AttentionHook
from .kernels import AttentionKernel, ReferenceAttentionKernel, SDPAAttentionKernel
from .projections import GroupedQueryProjection
from .retention import (
    AttentionSinkRetention,
    FullRetention,
    RetentionPolicy,
    SlidingWindowRetention,
)
from .state import (
    DenseKVCacheManager,
    DenseKVState,
    Int8KVCacheManager,
    Int8KVState,
    StatelessManager,
)

CacheKind = Literal["none", "dense", "int8"]
BuiltAttention = Attention[DenseKVState | Int8KVState | None]


@dataclass(frozen=True)
class AttentionConfig:
    d_model: int
    num_query_heads: int
    num_kv_heads: int | None = None
    cache: CacheKind = "none"
    retention: Literal["full", "sliding", "sinks"] = "full"
    context: Literal["full", "local", "local_global"] = "full"
    kernel: Literal["reference", "sdpa"] = "sdpa"
    window_size: int | None = None
    num_sink_tokens: int = 0
    num_global_tokens: int = 0
    global_positions: tuple[int, ...] = ()
    global_queries: bool = False
    causal: bool = True
    dropout: float = 0.0
    bias: bool = False
    detach_cache: bool = False

    def __post_init__(self) -> None:
        if self.cache not in ("none", "dense", "int8"):
            raise ValueError("unknown cache")
        if self.retention not in ("full", "sliding", "sinks"):
            raise ValueError("unknown retention")
        if self.context not in ("full", "local", "local_global"):
            raise ValueError("unknown context")
        if self.kernel not in ("reference", "sdpa"):
            raise ValueError("unknown kernel")
        kv = self.num_query_heads if self.num_kv_heads is None else self.num_kv_heads
        if min(self.d_model, self.num_query_heads, kv) <= 0:
            raise ValueError("dimensions and heads must be positive")
        if self.d_model % self.num_query_heads or self.num_query_heads % kv:
            raise ValueError("incompatible head dimensions")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.cache == "none" and self.retention != "full":
            raise ValueError("retention requires a cache")
        if self.cache != "none" and not self.causal:
            raise ValueError("incremental cache requires causal=True")
        if self.detach_cache and self.cache != "dense":
            raise ValueError("detach_cache only applies to dense cache")
        needs_window = self.retention != "full" or self.context != "full"
        if needs_window and (self.window_size is None or self.window_size <= 0):
            raise ValueError("selected policy requires positive window_size")
        if self.window_size is not None and self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.num_sink_tokens < 0 or self.num_global_tokens < 0:
            raise ValueError("sink and global counts must be nonnegative")
        if (
            self.retention == "sinks"
            and self.window_size is not None
            and self.num_sink_tokens >= self.window_size
        ):
            raise ValueError("num_sink_tokens must be smaller than window_size")
        if self.num_sink_tokens and self.retention != "sinks":
            raise ValueError("num_sink_tokens requires sinks retention")
        if (
            self.num_global_tokens or self.global_positions or self.global_queries
        ) and self.context != "local_global":
            raise ValueError("global options require local_global context")
        if any(p < 0 for p in self.global_positions):
            raise ValueError("global_positions must be nonnegative")


def build_attention(
    config: AttentionConfig, *, hooks: Sequence[AttentionHook] = ()
) -> BuiltAttention:
    """Return a dynamically typed model supporting a typed state round trip.

    The chosen manager validates state types at runtime. For a precise state
    specialization at compile time, construct Attention with a manager directly.
    """
    return cast(BuiltAttention, _build_attention(config, hooks=hooks))


def _build_attention(
    config: AttentionConfig, *, hooks: Sequence[AttentionHook] = ()
) -> Attention[None] | Attention[DenseKVState] | Attention[Int8KVState]:
    """Build components. For an exact state type, compose Attention directly.

    window_size is the retention budget and local width. With sinks and
    local_global together, the local width is budget minus sink count and
    initial global count defaults to num_sink_tokens.
    """
    retention: RetentionPolicy = FullRetention()
    context: ContextPolicy = FullContext()
    zero_start = config.retention == "sinks" and config.context == "local_global"
    if config.retention == "sliding":
        assert config.window_size is not None
        retention = SlidingWindowRetention(config.window_size)
    elif config.retention == "sinks":
        assert config.window_size is not None
        retention = AttentionSinkRetention(config.window_size, config.num_sink_tokens)
    if config.context == "local":
        assert config.window_size is not None
        context = LocalContext(config.window_size)
    elif config.context == "local_global":
        assert config.window_size is not None
        local_width = config.window_size
        globals_count = config.num_global_tokens
        if config.retention == "sinks":
            local_width -= config.num_sink_tokens
            globals_count = max(globals_count, config.num_sink_tokens)
        context = LocalGlobalContext(
            local_width,
            globals_count,
            global_positions=config.global_positions,
            global_queries=config.global_queries,
        )
    projection = GroupedQueryProjection(
        config.d_model,
        config.num_query_heads,
        config.num_query_heads if config.num_kv_heads is None else config.num_kv_heads,
        bias=config.bias,
    )
    kernel: AttentionKernel = (
        ReferenceAttentionKernel() if config.kernel == "reference" else SDPAAttentionKernel()
    )
    if config.cache == "dense":
        return Attention(
            projection,
            DenseKVCacheManager(
                retention, detach=config.detach_cache, require_zero_start=zero_start
            ),
            context,
            kernel,
            hooks=hooks,
            causal=config.causal,
            dropout=config.dropout,
        )
    if config.cache == "int8":
        return Attention(
            projection,
            Int8KVCacheManager(retention, require_zero_start=zero_start),
            context,
            kernel,
            hooks=hooks,
            causal=config.causal,
            dropout=config.dropout,
        )
    return Attention(
        projection,
        StatelessManager(),
        context,
        kernel,
        hooks=hooks,
        causal=config.causal,
        dropout=config.dropout,
    )
