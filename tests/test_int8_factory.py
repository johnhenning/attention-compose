import pytest
import torch
from torch import Tensor

import attention_compose as ca


def test_int8_decode_and_storage() -> None:
    model = ca.QuantizedCachedAttention(32, 4).eval()
    dense = ca.CachedAttention(32, 4).eval()
    dense.projection.load_state_dict(model.projection.state_dict())
    x = torch.randn(2, 12, 32)
    state: ca.Int8KVState | None = None
    outputs: list[Tensor] = []
    with torch.no_grad():
        for chunk in x.split(3, dim=1):
            result = model(chunk, state)
            if state is not None:
                assert result.state is not None
                torch.testing.assert_close(
                    result.state.k.values[:, :, : state.positions.numel()], state.k.values
                )
            state = result.state
            outputs.append(result.output)
        torch.testing.assert_close(torch.cat(outputs, 1), dense(x).output, atol=0.008, rtol=0.04)
    assert state is not None
    assert state.k.values.dtype == torch.int8
    assert state.k.scales.dtype == torch.float32
    assert state.next_position == 12


def test_quantization_zero_and_error_bound() -> None:
    for x in [
        torch.zeros(2, 2, 3, 8),
        torch.randn(2, 2, 3, 8),
        torch.full((1, 1, 1, 4), 1e-7, dtype=torch.float16),
    ]:
        quantized = ca.QuantizedTensor.quantize(x)
        reconstructed = quantized.dequantize(torch.float32)
        assert torch.isfinite(reconstructed).all()
        assert ((reconstructed - x.float()).abs() <= quantized.scales / 2 + 1e-7).all()


def test_int8_requires_no_grad() -> None:
    with pytest.raises(ValueError, match="inference"):
        ca.QuantizedCachedAttention(8, 2)(torch.randn(1, 3, 8))


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("cache", ["none", "dense", "int8"])
def test_builder(kv_heads: int, cache: str) -> None:
    from typing import cast

    config = ca.AttentionConfig(
        d_model=16, num_query_heads=4, num_kv_heads=kv_heads, cache=cast(ca.CacheKind, cache)
    )
    model = ca.build_attention(config).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 16)).output.shape == (2, 3, 16)


def test_builder_rejects_invalid_combinations() -> None:
    with pytest.raises(ValueError):
        ca.build_attention(ca.AttentionConfig(16, 4, retention="sliding", window_size=3))
    with pytest.raises(ValueError):
        ca.build_attention(ca.AttentionConfig(16, 4, context="local"))
    with pytest.raises(ValueError):
        ca.GroupedQueryProjection(15, 4, 2)
    with pytest.raises(ValueError):
        ca.SlidingWindowRetention(0)
    with pytest.raises(ValueError):
        ca.AttentionSinkRetention(3, 3)


def test_quantization_rejects_values_outside_fp32_range() -> None:
    x = torch.full((1, 1, 1, 4), 1e100, dtype=torch.float64)
    with pytest.raises(ValueError, match="FP32"):
        ca.QuantizedTensor.quantize(x)


def test_sink_compositions_reject_nonzero_start() -> None:
    models = [
        ca.AttentionSinkAttention(8, 2, 5, 2),
        ca.build_attention(
            ca.AttentionConfig(
                8,
                2,
                cache="dense",
                retention="sinks",
                context="local_global",
                window_size=5,
                num_sink_tokens=2,
            )
        ),
        ca.build_attention(
            ca.AttentionConfig(
                8,
                2,
                cache="int8",
                retention="sinks",
                context="local_global",
                window_size=5,
                num_sink_tokens=2,
            )
        ),
    ]
    with torch.no_grad():
        for model in models:
            with pytest.raises(ValueError, match="zero"):
                model(torch.randn(1, 10, 8), positions=torch.arange(10, 20))


@pytest.mark.parametrize("sinks", [0, 2])
def test_int8_retention_chunk_equivalence(sinks: int) -> None:
    model = ca.Attention(
        ca.GroupedQueryProjection(16, 4, 2),
        ca.Int8KVCacheManager(ca.AttentionSinkRetention(5, sinks)),
        ca.LocalGlobalContext(5 - sinks, sinks),
    ).eval()
    x = torch.randn(2, 19, 16)
    with torch.inference_mode():
        expected = model(x).output
        state: ca.Int8KVState | None = None
        pieces: list[Tensor] = []
        for part in x.split(7, dim=1):
            result = model(part, state)
            state = result.state
            assert state is not None and state.positions.numel() <= 5
            pieces.append(result.output)
        torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=1e-6, rtol=1e-5)


def test_factory_rejects_wrong_state_kind() -> None:
    dense = ca.build_attention(ca.AttentionConfig(8, 2, cache="dense"))
    quantized = ca.build_attention(ca.AttentionConfig(8, 2, cache="int8"))
    stateless = ca.build_attention(ca.AttentionConfig(8, 2))
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        dense_state = dense(x).state
        int8_state = quantized(x).state
        with pytest.raises(ValueError, match="DenseKVState"):
            dense(x, int8_state)
        with pytest.raises(ValueError, match="Int8KVState"):
            quantized(x, dense_state)
        with pytest.raises(ValueError, match="state=None"):
            stateless(x, dense_state)
