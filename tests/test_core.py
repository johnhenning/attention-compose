from copy import deepcopy

import pytest
import torch
from torch import Tensor, nn

import attention_compose as ca


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("causal", [True, False])
def test_reference_matches_sdpa(kv_heads: int, causal: bool) -> None:
    ref = ca.Attention(
        ca.GroupedQueryProjection(16, 4, kv_heads),
        kernel=ca.ReferenceAttentionKernel(),
        causal=causal,
    ).double()
    fast = deepcopy(ref)
    fast.kernel = ca.SDPAAttentionKernel()
    x = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
    y = ref(x).output
    z = fast(x).output
    torch.testing.assert_close(y, z)
    gy = torch.autograd.grad(y.square().sum(), x, retain_graph=True)[0]
    gz = torch.autograd.grad(z.square().sum(), x)[0]
    torch.testing.assert_close(gy, gz)


def test_matches_torch_mha() -> None:
    model = ca.MultiHeadAttention(16, 4).double()
    baseline = nn.MultiheadAttention(16, 4, bias=False, batch_first=True).double()
    p = model.projection
    assert baseline.in_proj_weight is not None
    with torch.no_grad():
        baseline.in_proj_weight.copy_(
            torch.cat([p.q_proj.weight, p.k_proj.weight, p.v_proj.weight])
        )
        baseline.out_proj.weight.copy_(p.out_proj.weight)
    x = torch.randn(2, 6, 16, dtype=torch.float64)
    mask = torch.ones(6, 6, dtype=torch.bool).triu(1)
    expected, _ = baseline(x, x, x, attn_mask=mask, need_weights=False)
    torch.testing.assert_close(model(x).output, expected)


@pytest.mark.parametrize("kernel", [ca.ReferenceAttentionKernel, ca.SDPAAttentionKernel])
def test_fully_masked_rows_are_zero(kernel: type[ca.AttentionKernel]) -> None:
    q = torch.randn(1, 2, 3, 4, requires_grad=True)
    context = ca.AttentionContext(q, q, torch.arange(3), torch.zeros(3, 3, dtype=torch.bool))
    output = kernel()(q, context, torch.arange(3))
    torch.testing.assert_close(output, torch.zeros_like(q))
    output.sum().backward()
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()


def test_hooks_are_ordered_registered_and_differentiable() -> None:
    events: list[str] = []

    class Hook(ca.AttentionHook):
        def __init__(self) -> None:
            super().__init__()
            self.gain = nn.Parameter(torch.tensor(2.0))

        def after_projection(self, qkv: ca.QKV, positions: Tensor) -> ca.QKV:
            events.append("projection")
            return qkv

        def before_attention(
            self, context: ca.AttentionContext, positions: Tensor
        ) -> ca.AttentionContext:
            events.append("context")
            return context

        def after_attention(self, attended: Tensor) -> Tensor:
            events.append("attention")
            return attended

        def after_output(self, output: Tensor) -> Tensor:
            events.append("output")
            return output * self.gain

    hook = Hook()
    model = ca.Attention(ca.MultiHeadProjection(8, 2), hooks=[hook])
    model(torch.randn(1, 3, 8)).output.sum().backward()
    assert events == ["projection", "context", "attention", "output"]
    assert hook.gain.grad is not None
    assert "hooks.0.gain" in model.state_dict()
    model.double().eval()
    assert hook.gain.dtype == torch.float64
    assert not hook.training


def test_dropout_is_disabled_in_eval() -> None:
    model = ca.Attention(ca.MultiHeadProjection(8, 2), dropout=0.5).eval()
    x = torch.randn(1, 5, 8)
    torch.testing.assert_close(model(x).output, model(x).output)
    model.train()
    assert not torch.equal(model(x).output, model(x).output)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_reference_sdpa(dtype: torch.dtype) -> None:
    q = torch.randn(1, 4, 5, 8, dtype=dtype)
    k = torch.randn(1, 2, 5, 8, dtype=dtype)
    v = torch.randn(1, 2, 5, 8, dtype=dtype)
    positions = torch.arange(5)
    context = ca.AttentionContext(k, v, positions)
    actual = ca.ReferenceAttentionKernel()(q, context, positions)
    expected = ca.SDPAAttentionKernel()(q, context, positions)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
