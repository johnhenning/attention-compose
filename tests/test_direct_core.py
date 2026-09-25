"""Composite modules versus standalone equations, including parameter gradients."""

from typing import Literal

import pytest
import torch
from reference_implementations import direct_attention

import attention_compose as ca


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("kernel_name", ["reference", "sdpa"])
@pytest.mark.parametrize("causal", [True, False])
def test_composite_matches_direct_outputs_and_gradients(
    kv_heads: int, kernel_name: Literal["reference", "sdpa"], causal: bool
) -> None:
    kernel = (
        ca.ReferenceAttentionKernel() if kernel_name == "reference" else ca.SDPAAttentionKernel()
    )
    model = ca.Attention(
        ca.GroupedQueryProjection(16, 4, kv_heads, bias=True), kernel=kernel, causal=causal
    ).double()
    weights = {
        name: p.detach().clone().requires_grad_() for name, p in model.projection.named_parameters()
    }
    x = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = model(x).output
    expected = direct_attention(reference_x, weights, 4, kv_heads, causal=causal)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-10, rtol=1e-8)
    for name, parameter in model.projection.named_parameters():
        torch.testing.assert_close(parameter.grad, weights[name].grad, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("kind", ["mha", "gqa", "mqa"])
def test_convenience_classes_match_direct(kind: str) -> None:
    if kind == "mha":
        model = ca.MultiHeadAttention(16, 4)
        kv_heads = 4
    elif kind == "gqa":
        model = ca.GroupedQueryAttention(16, 4, 2)
        kv_heads = 2
    else:
        model = ca.MultiQueryAttention(16, 4)
        kv_heads = 1
    x = torch.randn(2, 9, 16)
    expected = direct_attention(x, dict(model.projection.named_parameters()), 4, kv_heads)
    torch.testing.assert_close(model(x).output, expected, atol=1e-6, rtol=1e-5)
