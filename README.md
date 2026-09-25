# Attention Compose

Composable, fully typed PyTorch self-attention for Python 3.12+.

## Development

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pyrefly check
pytest -q
python examples/quickstart.py
python examples/custom_hook.py
python -m build
```

## Plain attention

```python
import torch
from attention_compose import MultiHeadAttention

attention = MultiHeadAttention(64, 8)
result = attention(torch.randn(2, 10, 64))
assert result.output.shape == (2, 10, 64)
assert result.state is None
```

Compose `Attention(projection, state_manager, context_policy, kernel)` directly,
or use `MultiHeadAttention`, `GroupedQueryAttention`, or `MultiQueryAttention`.
The default is causal attention; set `causal=False` for bidirectional attention.
Inputs are [batch, tokens, d_model]; internal tensors are [batch, heads, tokens, dim].

Projection strategies are `MultiHeadProjection`, `GroupedQueryProjection`, and
`MultiQueryProjection`. `ReferenceAttentionKernel` and `SDPAAttentionKernel` use
absolute-position masks and return zero attended values for fully masked rows.
Low precision reference scores accumulate in FP32. SDPA backend selection is up
to PyTorch. GQA/MQA expand KV heads for kernel execution.

The orchestrator defaults to `StatelessManager` and `FullContext`. It is generic
over opaque state, defaults to None using typing_extensions on Python 3.12, and
does not inspect state fields. To supply custom state, implement
`AttentionStateManager[YourState].next_position` and `.prepare`; return
`PreparedAttention(current_context, next_state)`.

Subclass `AttentionHook` to transform `after_projection`, `before_attention`,
`after_attention`, or `after_output`. Hooks run in order and are registered
PyTorch modules, supporting parameters, state_dict, .to(), and .eval().
The typed `Attention.__call__` preserves ordinary PyTorch module hook dispatch.

The layer includes projections but no residual, normalization, feed-forward
layers, or positional encoding. Positions are shared across the batch; padding,
ragged batching and cross-attention are not included. Dropout is off in eval.
This package is not published to PyPI.

Tests compare composite attention against standalone functional equations with no
library imports, composition or inheritance, including output and gradient checks.
