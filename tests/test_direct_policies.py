import pytest
import torch
from reference_implementations import direct_attention
from torch import Tensor

import attention_compose as ca


@pytest.mark.parametrize("global_queries", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_sparse_composite_matches_direct(global_queries: bool, causal: bool) -> None:
    model = ca.SparseAttention(
        16,
        4,
        3,
        num_global_tokens=1,
        global_positions=(5,),
        global_queries=global_queries,
        causal=causal,
    ).double()
    x = torch.randn(2, 10, 16, dtype=torch.float64)
    expected = direct_attention(
        x,
        dict(model.projection.named_parameters()),
        4,
        4,
        causal=causal,
        window=3,
        global_positions=(0, 5),
        global_queries=global_queries,
    )
    torch.testing.assert_close(model(x).output, expected, atol=1e-10, rtol=1e-8)


def test_local_composite_matches_direct() -> None:
    model = ca.LocalAttention(16, 4, 3).double()
    x = torch.randn(2, 10, 16, dtype=torch.float64)
    expected = direct_attention(x, dict(model.projection.named_parameters()), 4, 4, window=3)
    torch.testing.assert_close(model(x).output, expected, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("sinks", [0, 2])
@pytest.mark.parametrize("chunk_size", [1, 7, 19])
def test_window_and_sink_composites_match_direct(sinks: int, chunk_size: int) -> None:
    model = (
        ca.AttentionSinkAttention(16, 4, 5, sinks) if sinks else ca.SlidingWindowAttention(16, 4, 5)
    )
    model.double()
    x = torch.randn(2, 19, 16, dtype=torch.float64)
    expected = direct_attention(
        x,
        dict(model.projection.named_parameters()),
        4,
        4,
        window=5 - sinks,
        global_positions=tuple(range(sinks)),
    )
    state: ca.DenseKVState | None = None
    pieces: list[Tensor] = []
    for part in x.split(chunk_size, dim=1):
        result = model(part, state)
        state = result.state
        pieces.append(result.output)
    torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("sinks", [0, 2])
def test_retention_matches_direct_raw_input_recomputation(sinks: int) -> None:
    """Independent streaming reference retains raw inputs, not projected KV."""
    retention = ca.AttentionSinkRetention(5, sinks) if sinks else ca.SlidingWindowRetention(5)
    model = ca.Attention(
        ca.GroupedQueryProjection(16, 4, 2), ca.DenseKVCacheManager(retention)
    ).double()
    x = torch.randn(2, 21, 16, dtype=torch.float64)
    weights = dict(model.projection.named_parameters())
    kept: list[int] = []
    state: ca.DenseKVState | None = None
    for start, end in [(0, 9), (9, 15), (15, 21)]:
        current = list(range(start, end))
        available = kept + current
        expected = direct_attention(
            x[:, start:end],
            weights,
            4,
            2,
            keys=x[:, available],
            query_positions=current,
            key_positions=available,
        )
        result = model(x[:, start:end], state)
        state = result.state
        torch.testing.assert_close(result.output, expected, atol=1e-10, rtol=1e-8)
        kept = available if len(available) <= 5 else available[:sinks] + available[-(5 - sinks) :]
        assert state is not None and state.positions.tolist() == kept
