from typing import assert_type

import torch

from attention_compose import (
    AttentionConfig,
    AttentionOutput,
    DenseKVState,
    Int8KVState,
    QuantizedCachedAttention,
    build_attention,
)


def test_factory_state_round_trip() -> None:
    model = build_attention(AttentionConfig(8, 2, cache="dense"))
    x = torch.randn(1, 3, 8)
    first = model(x)
    assert_type(first.state, DenseKVState | Int8KVState | None)
    second = model(x, first.state)
    assert isinstance(second.state, DenseKVState)
    assert second.state.next_position == 6


def test_int8_return_type() -> None:
    with torch.no_grad():
        assert_type(
            QuantizedCachedAttention(8, 2)(torch.randn(1, 2, 8)), AttentionOutput[Int8KVState]
        )
