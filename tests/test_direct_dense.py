import pytest
import torch
from reference_implementations import direct_attention
from torch import Tensor

import attention_compose as ca


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("chunk_size", [1, 5, 13])
def test_cached_composite_matches_direct_full_sequence(kv_heads: int, chunk_size: int) -> None:
    model = ca.CachedAttention(16, 4, num_kv_heads=kv_heads).double()
    x = torch.randn(2, 13, 16, dtype=torch.float64, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    weights = {
        name: p.detach().clone().requires_grad_() for name, p in model.projection.named_parameters()
    }
    expected = direct_attention(ref_x, weights, 4, kv_heads)
    pieces: list[Tensor] = []
    state: ca.DenseKVState | None = None
    for part in x.split(chunk_size, dim=1):
        result = model(part, state)
        pieces.append(result.output)
        state = result.state
    actual = torch.cat(pieces, 1)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, atol=1e-10, rtol=1e-8)
    for name, parameter in model.projection.named_parameters():
        torch.testing.assert_close(parameter.grad, weights[name].grad, atol=1e-10, rtol=1e-8)
