# Composable attention

Implement the component architecture requested in Explain KV Cache as a Python
3.12 library. The orchestrator is generic over opaque state, defaults to None via
typing_extensions.TypeVar, and never examines cache fields. All components expose
typed interfaces. Projection and hooks are registered PyTorch modules.

Lifecycle: validate input and positions; project; transform QKV; prepare the
current context and next state; apply context policy; transform context; run the
kernel; transform attended values; project output; transform output. State
managers return both context and retained state atomically, avoiding eviction of
keys required by early queries in a multi-token prefill. Cached positions are
contiguous absolute positions; stateless calls may supply strictly increasing
positions. All batch members share the position sequence.

Dense cache preserves gradients by default, with an explicit detach option.
INT8 cache uses symmetric per-token, per-head scales computed in FP32 and is
inference-only; old quantized entries are concatenated without requantization.
The current context is dequantized. Retention limits future stored state, while
context policies determine which keys each current query sees. Sliding and sink
convenience classes pair matching retention and context masks for chunk-invariant
causal inference. Full retention, sliding retention, and sink-plus-recent retention
are separate from full, local, and local-plus-global context policies.

Both kernels use absolute-position causal masks; empty attention rows yield zero.
Sparse policies are dense Boolean masks, not computationally sparse kernels.
SDPA explicitly disables its implicit causal mask and disables dropout in eval.
The reference kernel computes FP16/BF16 scores in FP32 for stability.

Include direct composition, convenience classes, validated config/factory,
README and runnable examples, package typing marker, Ruff, Pyrefly, tests, and CI.
Publish to the user-created public johnhenning/attention-compose repository.
No PyPI publication or production performance claims are part of this work.
