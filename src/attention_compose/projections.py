"""Projection choices are independent of attention and cache behavior."""

from torch import Tensor, nn

from .types import QKV


class AttentionProjection(nn.Module):
    def __init__(
        self, d_model: int, num_query_heads: int, num_kv_heads: int, *, bias: bool = False
    ) -> None:
        super().__init__()
        if min(d_model, num_query_heads, num_kv_heads) <= 0:
            raise ValueError("dimensions and head counts must be positive")
        if d_model % num_query_heads or num_query_heads % num_kv_heads:
            raise ValueError(
                "d_model must divide into query heads, which must divide into KV groups"
            )
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _split(self, x: Tensor, heads: int) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], heads, self.head_dim).transpose(1, 2)

    def project(self, x: Tensor) -> QKV:
        return QKV(
            self._split(self.q_proj(x), self.num_query_heads),
            self._split(self.k_proj(x), self.num_kv_heads),
            self._split(self.v_proj(x), self.num_kv_heads),
        )

    def project_output(self, attended: Tensor) -> Tensor:
        merged = attended.transpose(1, 2).reshape(
            attended.shape[0], attended.shape[2], self.d_model
        )
        return self.out_proj(merged)


class MultiHeadProjection(AttentionProjection):
    def __init__(self, d_model: int, num_heads: int, *, bias: bool = False) -> None:
        super().__init__(d_model, num_heads, num_heads, bias=bias)


class GroupedQueryProjection(AttentionProjection):
    """Query heads share keys and values within contiguous groups."""


class MultiQueryProjection(AttentionProjection):
    def __init__(self, d_model: int, num_query_heads: int, *, bias: bool = False) -> None:
        super().__init__(d_model, num_query_heads, 1, bias=bias)
