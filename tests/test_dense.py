from copy import deepcopy

import pytest
import torch
from torch import Tensor

import attention_compose as ca


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("chunks", [[1] * 13, [7, 4, 2], [13]])
def test_dense_decode_equivalence(kv_heads: int, chunks: list[int]) -> None:
    model = ca.Attention(
        ca.GroupedQueryProjection(16, 4, kv_heads), ca.DenseKVCacheManager()
    ).eval()
    x = torch.randn(2, 13, 16)
    expected = model(x).output
    state: ca.DenseKVState | None = None
    outputs: list[Tensor] = []
    offset = 0
    for size in chunks:
        result = model(x[:, offset : offset + size], state)
        state = result.state
        outputs.append(result.output)
        offset += size
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-6, rtol=1e-5)
    assert state is not None
    assert state.next_position == 13
    assert state.k.shape == (2, kv_heads, 13, 4)


def test_dense_cache_gradients_and_detach() -> None:
    for detach in [False, True]:
        model = ca.Attention(ca.MultiHeadProjection(8, 2), ca.DenseKVCacheManager(detach=detach))
        first = torch.randn(1, 3, 8, requires_grad=True)
        state = model(first).state
        model(torch.randn(1, 1, 8), state).output.sum().backward()
        assert (first.grad is None) == detach


def test_positions_and_state_validation() -> None:
    model = ca.CachedAttention(8, 2)
    x = torch.randn(1, 3, 8)
    state = model(x, positions=torch.arange(10, 13)).state
    assert state is not None
    assert model(x, state).state is not None
    with pytest.raises(ValueError, match="positions"):
        model(x, state, positions=torch.arange(3))
    with pytest.raises(ValueError, match="batch|shape"):
        model(torch.randn(2, 3, 8), state)
    with pytest.raises(ValueError, match="positions"):
        model(x, positions=torch.tensor([0, 0, 1]))
    with pytest.raises(ValueError, match="empty|positive"):
        model(x[:, :0])


def test_dense_chunked_parameter_and_input_gradients() -> None:
    full = ca.CachedAttention(8, 2).double()
    chunked = deepcopy(full)
    x = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
    chunk_x = x.detach().clone().requires_grad_()
    full(x).output.square().sum().backward()
    state: ca.DenseKVState | None = None
    outputs: list[Tensor] = []
    for part in chunk_x.split(3, dim=1):
        result = chunked(part, state)
        state = result.state
        outputs.append(result.output)
    torch.cat(outputs, 1).square().sum().backward()
    torch.testing.assert_close(x.grad, chunk_x.grad)
    for p, q in zip(full.parameters(), chunked.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad)


def test_cache_state_is_not_mutated_and_dtype_change_rejected() -> None:
    model = ca.CachedAttention(8, 2)
    x = torch.randn(1, 3, 8)
    state = model(x).state
    assert state is not None
    previous = state.k.clone()
    model(x, state)
    torch.testing.assert_close(state.k, previous)
    assert state.next_position == 3
    assert state.positions.tolist() == [0, 1, 2]
    model.double()
    with pytest.raises(ValueError, match="dtype"):
        model(x.double(), state)
