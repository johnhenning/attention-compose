"""Run with `uv run python examples/quickstart.py`."""

import torch
from torch import Tensor

from attention_compose import (
    Attention,
    AttentionConfig,
    AttentionSinkAttention,
    DenseKVCacheManager,
    DenseKVState,
    GroupedQueryProjection,
    LocalContext,
    MultiHeadAttention,
    QuantizedCachedAttention,
    SlidingWindowRetention,
    build_attention,
)


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 12, 32)
    plain = MultiHeadAttention(32, 4).eval()
    assert plain(x).state is None

    # Composition determines the generic state type automatically.
    model = Attention(
        GroupedQueryProjection(32, 4, 2),
        DenseKVCacheManager(SlidingWindowRetention(5)),
        LocalContext(5),
    ).eval()
    state: DenseKVState | None = None
    pieces: list[Tensor] = []
    with torch.inference_mode():
        expected = model(x).output
        for chunk in x.split(3, dim=1):
            result = model(chunk, state)
            state = result.state
            pieces.append(result.output)
        torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=1e-6, rtol=1e-5)
        assert state is not None and state.positions.numel() == 5
        sinks = AttentionSinkAttention(32, 4, window_size=6, num_sink_tokens=2).eval()
        sink_result = sinks(x)
        assert sink_result.state is not None
        assert sink_result.state.positions.tolist() == [0, 1, 8, 9, 10, 11]
        quantized = QuantizedCachedAttention(32, 4, num_kv_heads=1).eval()
        assert quantized(x).state is not None
        built = build_attention(AttentionConfig(32, 4, num_kv_heads=2, cache="dense"))
        assert built(x).output.shape == x.shape
    print("Plain, GQA, sliding cache, attention sinks, INT8, and factory examples passed.")


if __name__ == "__main__":
    main()
