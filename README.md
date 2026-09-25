# Attention Compose

Composable, fully typed PyTorch self-attention for Python 3.12+.

## Development

```sh
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pyrefly check
uv run --locked pytest -q
uv run --locked python examples/quickstart.py
uv run --locked python examples/custom_hook.py
uv build
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

## uv workflow

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
`uv sync --locked`. Python 3.12 is selected by `.python-version`; `uv.lock` pins
runtime and development dependencies. CI uses the same locked environment.
Use `uv add PACKAGE`, `uv add --dev TOOL`, or `uv lock --upgrade` when deliberately
updating dependencies, and commit the changed lockfile.

The project selects CPU PyTorch wheels on Linux/Windows to keep CI lightweight;
macOS uses PyPI wheels. GPU users can adapt the named PyTorch index to their CUDA
version and regenerate the lock, or install the wheel into an existing PyTorch
environment. The published package metadata does not force a CPU-only index.
