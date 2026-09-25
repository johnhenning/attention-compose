from typing import assert_type

import torch

from attention_compose import (
    QKV,
    Attention,
    AttentionContext,
    AttentionOutput,
    AttentionStateManager,
    MultiHeadAttention,
    MultiHeadProjection,
    PreparedAttention,
)


def test_opaque_custom_state_inference() -> None:
    class CounterManager(AttentionStateManager[int]):
        def next_position(self, state: int | None) -> int:
            return 0 if state is None else state

        def prepare(
            self, qkv: QKV, positions: torch.Tensor, state: int | None
        ) -> PreparedAttention[int]:
            return PreparedAttention(
                AttentionContext(qkv.k, qkv.v, positions),
                self.next_position(state) + positions.numel(),
            )

    model = Attention(MultiHeadProjection(8, 2), CounterManager())
    assert_type(model, Attention[int])
    first = model(torch.randn(1, 3, 8))
    assert_type(first, AttentionOutput[int])
    second = model(torch.randn(1, 2, 8), first.state)
    assert second.state == 5


def test_plain_return_type() -> None:
    x = torch.randn(1, 2, 8)
    plain: Attention = MultiHeadAttention(8, 2)
    assert_type(plain(x), AttentionOutput[None])
