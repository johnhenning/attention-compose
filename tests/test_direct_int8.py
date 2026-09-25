import pytest
import torch
from reference_implementations import direct_attention
from torch import Tensor

import attention_compose as ca


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("chunk_size", [1, 5, 17])
def test_int8_composite_approximates_direct_dense(kv_heads: int, chunk_size: int) -> None:
    model = ca.QuantizedCachedAttention(32, 4, num_kv_heads=kv_heads).eval()
    x = torch.randn(2, 17, 32)
    state: ca.Int8KVState | None = None
    pieces: list[Tensor] = []
    with torch.inference_mode():
        expected = direct_attention(x, dict(model.projection.named_parameters()), 4, kv_heads)
        for part in x.split(chunk_size, dim=1):
            result = model(part, state)
            state = result.state
            pieces.append(result.output)
    actual = torch.cat(pieces, 1)
    # Quantization is approximate: constrain both worst-case and average error.
    assert float((actual - expected).abs().max()) < 0.01
    assert float((actual - expected).square().mean().sqrt()) < 0.003
