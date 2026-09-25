# Implementation plan

1. Write behavioral tests for projection/kernel equivalence, causal chunked
   decoding, bounded retention and sinks, quantization error, hook registration,
   gradients, invalid arguments, config building, and static return types.
2. Implement typed records, projections, retention/context strategies, and kernels
   in separate modules. Compare against independent PyTorch MHA and manual masks.
3. Implement state managers, typed hooks, and the orchestrator. Cache preparation
   must expose all current keys before selecting the next retained state.
4. Add convenience classes and a config factory with early validation, README
   examples, package metadata, and Python 3.12 CI running Ruff, Pyrefly and pytest.
5. Run the complete checks, examples and wheel build; review edge cases and fix
   regression failures. Commit the verified package and publish to GitHub.

Review focus: long chunks exceeding capacity; nonzero positions after eviction;
fully masked rows; zero vectors in INT8 quantization; dtype/batch/device changes
between cache calls. Tests must assert outputs, state bounds and error behavior.
