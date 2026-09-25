# Attention Compose

[![CI](https://github.com/johnhenning/attention-compose/actions/workflows/ci.yml/badge.svg)](https://github.com/johnhenning/attention-compose/actions/workflows/ci.yml)

Build attention from interchangeable PyTorch components. Choose projections,
state management, retention, context masks, execution kernels, and lifecycle
hooks independently. A single `Attention` orchestrator runs every composition.

Python **3.12+**, full public type hints, Ruff, Pyrefly, and a `py.typed` marker.

## Install

```sh
python -m pip install git+https://github.com/johnhenning/attention-compose.git
```

For development, with Python 3.12:

```sh
git clone https://github.com/johnhenning/attention-compose.git
cd attention-compose
uv sync --locked
```

The package is installed as `attention-compose` and imported as `attention_compose`.
It has not been published to PyPI.

## Notebook demo

Open the executed [hands-on demo](notebooks/attention_compose_demo.ipynb) for
composition, direct-reference comparisons, cached decoding, mask visualizations,
window/sink retention, INT8 storage/error plots, hooks, and the config factory.

```sh
uv sync --locked --group demo
uv run --group demo jupyter lab notebooks/attention_compose_demo.ipynb
```

Choose the project's Python kernel and run all cells in order. For automated
verification, run `uv run --group demo python notebooks/execute.py`.

## Start with plain attention

```python
import torch
from attention_compose import MultiHeadAttention

attention = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(2, 10, 64)  # [batch, tokens, d_model]
result = attention(x)
assert result.output.shape == x.shape
assert result.state is None
```

Attention is causal by default. Pass `causal=False` for bidirectional stateless
attention. Each module includes Q/K/V and output projections; it does not include
a residual connection, normalization, feed-forward layers, or positional encoding.

## Compose the pieces

```python
from attention_compose import (
    Attention,
    DenseKVCacheManager,
    GroupedQueryProjection,
    LocalContext,
    SDPAAttentionKernel,
    SlidingWindowRetention,
)

attention = Attention(
    projection=GroupedQueryProjection(64, num_query_heads=8, num_kv_heads=2),
    state_manager=DenseKVCacheManager(SlidingWindowRetention(window_size=128)),
    context_policy=LocalContext(window_size=128),
    kernel=SDPAAttentionKernel(),
).eval()
```

Each part has one responsibility:

- **Projection:** `MultiHeadProjection`, `GroupedQueryProjection`, or
  `MultiQueryProjection`. KV heads stay compact in stored state.
- **State:** `StatelessManager`, `DenseKVCacheManager`, or `Int8KVCacheManager`.
  State belongs to the caller, not the module, so separate requests can share weights.
- **Retention:** `FullRetention`, `SlidingWindowRetention`, or
  `AttentionSinkRetention`. Determines which entries survive for the next call.
- **Context:** `FullContext`, `LocalContext`, or `LocalGlobalContext`. Determines
  which available keys each query may attend to.
- **Kernel:** `ReferenceAttentionKernel` or `SDPAAttentionKernel`.
- **Hooks:** subclasses of `AttentionHook`, registered as PyTorch modules.

## Prefill and decode

```python
import torch
from attention_compose import CachedAttention, DenseKVState

attention = CachedAttention(64, num_query_heads=8, num_kv_heads=2).eval()
state: DenseKVState | None = None

with torch.inference_mode():
    prefill = attention(torch.randn(2, 20, 64), state)
    state = prefill.state
    next_token = attention(torch.randn(2, 1, 64), state)
    state = next_token.state
    assert state is not None and state.next_position == 21
```

Passing `state=None` starts a fresh sequence. Cache entries are not mutated by
subsequent calls, so a state can be reused to branch a sequence. Tensor payloads
are still mutable PyTorch tensors: treat them as read-only. State is external to
`state_dict()` and `.to()`; reset it if the batch, dtype, or device changes.

Dense caches retain the autograd graph by default. Use `detach=True` on
`DenseKVCacheManager` (or `CachedAttention`) to stop gradients through prior calls.
Use `.eval()` plus `torch.inference_mode()` for inference; `.eval()` alone does
not turn off autograd. Dropout is disabled automatically in eval mode.

`Attention` defaults its generic state parameter to `None`, using
`typing_extensions.TypeVar` for Python 3.12. Direct component composition infers
the state type, and both `attention(...)` and `attention.forward(...)` return a
typed `AttentionOutput[StateT]`.

## Window and sink semantics

```python
from attention_compose import SlidingWindowAttention, AttentionSinkAttention

local = SlidingWindowAttention(64, 8, window_size=128)
sinks = AttentionSinkAttention(64, 8, window_size=128, num_sink_tokens=4)
```

Local causal attention sees the current token plus the previous `window_size-1`
positions. Sink attention uses a **total** budget: four initial tokens plus 124
recent positions in this example. Sink convenience calls must start at position
zero, matching the initial global keys. Chunked and token-at-a-time decoding agree
when dropout is disabled and the context policy matches retention.

Retention alone is not a sliding attention mask. With `FullContext` and a bounded
cache, all keys available to the current call remain visible, then only the
selected keys are stored for the next call. Consequently that combination can
depend on chunk boundaries. Pair sliding retention with local context for sliding
attention, as the convenience class does.

The manager prepares current context **before** evicting entries for the next
state. A large prompt therefore cannot evict keys required by its early queries.
The persistent cache is bounded; temporary prefill tensors can be larger.

Masks use absolute positions, including after eviction. All batch members share
a one-dimensional position sequence. Explicit positions must be nonnegative,
strictly increasing `torch.int64` tensors on the input device; cached positions
must be contiguous and continue the prior state's `next_position`.

## INT8 cache

```python
import torch
from attention_compose import QuantizedCachedAttention

attention = QuantizedCachedAttention(64, 8, num_kv_heads=2).eval()
with torch.inference_mode():
    result = attention(torch.randn(2, 20, 64))
    result = attention(torch.randn(2, 1, 64), result.state)
```

INT8 is inference-only and rejects calls with gradient tracking enabled. It uses
symmetric per-token, per-head quantization with FP32 scales; old cache entries
are not repeatedly quantized. Current K/V are dequantized for the kernel. This
reduces persistent cache storage (depending on head dimension and scale overhead)
but does not promise faster execution or lower peak prefill memory. Outputs are
approximate. There are no custom INT8 attention kernels.

## Local + global sparse pattern

```python
from attention_compose import Attention, MultiHeadProjection, LocalGlobalContext

attention = Attention(
    MultiHeadProjection(64, 8),
    context_policy=LocalGlobalContext(
        window_size=32,
        num_global_tokens=2,
        global_positions=(100, 200),
        global_queries=False,
    ),
)
```

Initial global keys are positions `[0, num_global_tokens)`. Explicit global
positions add more global keys. Set `global_queries=True` to let those queries
see every available key. Causality still applies. Retention may remove global
keys: use full retention if arbitrary global positions must remain available.

“Sparse” describes the visibility pattern: both supplied kernels use dense
Boolean masks, **not** block-sparse computation. GQA/MQA heads are expanded for
kernel execution, preserving compact cache storage but not all possible kernel
memory savings. SDPA backend selection is left to PyTorch.

## Hooks and custom state

Subclass `AttentionHook` and override any stage:

```python
import torch
from torch import Tensor, nn
from attention_compose import AttentionHook, MultiHeadAttention


class Gain(AttentionHook):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(()))

    def after_output(self, output: Tensor) -> Tensor:
        return output * self.gain


attention = MultiHeadAttention(64, 8, hooks=[Gain()])
```

Hook order: `after_projection(QKV, positions)` →
`before_attention(AttentionContext, positions)` → `after_attention(Tensor)` →
`after_output(Tensor)`. Each method returns its transformed value; hooks run in
list order. Apply position-dependent Q/K transformations in `after_projection`
so cached keys are transformed only once. `before_attention` sees the selected
context and policy mask and does not rewrite the next stored state.

To add another cache, subclass `AttentionStateManager[YourState]` and implement
`next_position(state)` plus `prepare(qkv, positions, state)`. Return
`PreparedAttention(context, next_state)`. This replaces the sketch's separate
`update` and `context` methods with one atomic operation that can return a full
current context and a smaller retained state. The orchestrator never reads state
fields. See [the design](docs/design.md) and [hook example](examples/custom_hook.py).

## Config / factory

```python
from attention_compose import AttentionConfig, build_attention

config = AttentionConfig(
    d_model=64,
    num_query_heads=8,
    num_kv_heads=2,
    cache="dense",
    retention="sinks",
    context="local_global",
    window_size=128,
    num_sink_tokens=4,
    kernel="sdpa",
)
attention = build_attention(config)
```

Config validates incompatible dimensions, names, budgets, and cache options.
With sink retention plus local/global context it subtracts sink count from the
local width and includes the sinks as globals. Direct composition offers the
most precise static state type; dynamic factory construction covers all supported
state types. Serialize config with `dataclasses.asdict`.

Other convenience classes: `GroupedQueryAttention`, `MultiQueryAttention`,
`LocalAttention`, and `SparseAttention`. All share the same orchestrator.

## Verification

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pyrefly check
uv run --locked pytest -q
uv run --locked python examples/quickstart.py
uv run --locked python examples/custom_hook.py
uv build
```

CI runs these checks on Python 3.12 with CPU PyTorch. Tests cover independent
PyTorch MHA equivalence, reference/SDPA outputs and gradients, dense incremental
decoding, multi-token chunks larger than cache capacity, sink retention, INT8
accuracy, hooks, masks, invalid input, and generic types.

The suite also compares composite layers against standalone function-based
implementations in `tests/reference_implementations.py`. Those oracles import no
library components, inherit from no classes, and directly compute projections,
head grouping, masks, softmax, and outputs. Comparisons cover MHA/GQA/MQA outputs
and parameter/input gradients, cached decoding, local/global patterns, sliding
and sink attention, and bounded-cache streaming that recomputes keys from raw
inputs. INT8 is compared with the direct dense reference using explicit maximum
and RMS error bounds.

This is a self-attention building block, not an LLM serving engine. Padding masks,
per-example ragged positions, cross-attention, paged allocation, and built-in RoPE
are not included. Custom context hooks can supply a shared `[Q, K]` mask. Completely
masked rows yield zero attended values; an output projection bias or output hook
can subsequently change that value.

Implementation references: [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
and [Pyrefly configuration](https://pyrefly.org/en/docs/configuration/).

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
