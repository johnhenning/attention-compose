from typing import assert_type

import torch

from attention_compose import AttentionOutput, CachedAttention, DenseKVState


def test_dense_return_type() -> None:
    assert_type(CachedAttention(8, 2)(torch.randn(1, 2, 8)), AttentionOutput[DenseKVState])
