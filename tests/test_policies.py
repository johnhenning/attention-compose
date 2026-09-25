from copy import deepcopy

import pytest
import torch
from torch import Tensor

import attention_compose as ca


@pytest.mark.parametrize("sinks", [0, 2])
@pytest.mark.parametrize("chunks", [[1] * 17, [9, 6, 2], [17]])
def test_bounded_cache_chunk_invariance(sinks: int, chunks: list[int]) -> None:
    policy = ca.LocalGlobalContext(5 - sinks, num_global_tokens=sinks)
    retention = ca.AttentionSinkRetention(5, sinks)
    model = ca.Attention(ca.MultiHeadProjection(16, 4), ca.DenseKVCacheManager(retention), policy)
    baseline = ca.Attention(deepcopy(model.projection), context_policy=policy)
    x = torch.randn(2, 17, 16)
    expected = baseline(x).output
    outputs: list[Tensor] = []
    state: ca.DenseKVState | None = None
    offset = 0
    for size in chunks:
        result = model(x[:, offset : offset + size], state)
        state = result.state
        assert state is not None
        assert state.positions.numel() <= 5
        outputs.append(result.output)
        offset += size
    torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=1e-6, rtol=1e-5)
    assert state is not None
    expected_positions = list(range(sinks)) + list(range(17 - (5 - sinks), 17))
    assert state.positions.tolist() == expected_positions
    assert state.next_position == 17


def test_retention_does_not_truncate_current_prefill() -> None:
    model = ca.Attention(
        ca.MultiHeadProjection(8, 2), ca.DenseKVCacheManager(ca.SlidingWindowRetention(3))
    )
    baseline = ca.Attention(deepcopy(model.projection))
    x = torch.randn(1, 11, 8)
    result = model(x)
    torch.testing.assert_close(result.output, baseline(x).output)
    assert result.state is not None
    assert result.state.positions.tolist() == [8, 9, 10]


def test_local_global_mask() -> None:
    positions = torch.arange(8)
    context = ca.AttentionContext(torch.zeros(1, 1, 8, 2), torch.zeros(1, 1, 8, 2), positions)
    result = ca.LocalGlobalContext(2, num_global_tokens=1, global_positions=(3,)).apply(
        context, torch.tensor([6])
    )
    assert result.allowed is not None
    assert result.allowed.tolist() == [[True, False, False, True, False, True, True, True]]


def test_sink_convenience_requires_zero_start() -> None:
    model = ca.AttentionSinkAttention(8, 2, 5, 2)
    with pytest.raises(ValueError, match="zero"):
        model(torch.randn(1, 5, 8), positions=torch.arange(10, 15))


@pytest.mark.parametrize("sinks", [0, 2])
def test_policy_convenience_matches_chunked(sinks: int) -> None:
    model = ca.AttentionSinkAttention(8, 2, 5, 2) if sinks else ca.SlidingWindowAttention(8, 2, 5)
    x = torch.randn(1, 15, 8)
    expected = model(x).output
    state: ca.DenseKVState | None = None
    pieces: list[Tensor] = []
    for part in x.split(6, dim=1):
        result = model(part, state)
        state = result.state
        pieces.append(result.output)
    torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=1e-6, rtol=1e-5)
