"""The generic orchestrator knows state lifecycle, never cache fields."""

from collections.abc import Sequence
from typing import Generic, cast

import torch
from torch import Tensor, nn

from .context import ContextPolicy, FullContext
from .hooks import AttentionHook
from .kernels import AttentionKernel, SDPAAttentionKernel
from .projections import AttentionProjection
from .state import AttentionStateManager, StatelessManager
from .types import AttentionOutput, StateT


class Attention(nn.Module, Generic[StateT]):
    def __init__(
        self,
        projection: AttentionProjection,
        state_manager: AttentionStateManager[StateT] | None = None,
        context_policy: ContextPolicy | None = None,
        kernel: AttentionKernel | None = None,
        *,
        hooks: Sequence[AttentionHook] = (),
        causal: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.projection = projection
        # A missing manager is the default StateT=None specialization.
        self.state_manager: AttentionStateManager[StateT] = (
            state_manager
            if state_manager is not None
            else cast(AttentionStateManager[StateT], StatelessManager())
        )
        self.context_policy = context_policy if context_policy is not None else FullContext()
        self.kernel = kernel if kernel is not None else SDPAAttentionKernel()
        self.hooks = nn.ModuleList(hooks)
        self.causal = causal
        self.dropout = dropout

    def __call__(
        self, x: Tensor, state: StateT | None = None, *, positions: Tensor | None = None
    ) -> AttentionOutput[StateT]:
        """Keep PyTorch module hooks while preserving the generic return type."""
        return cast(AttentionOutput[StateT], super().__call__(x, state, positions=positions))

    def forward(
        self, x: Tensor, state: StateT | None = None, *, positions: Tensor | None = None
    ) -> AttentionOutput[StateT]:
        if x.ndim != 3 or x.shape[-1] != self.projection.d_model:
            raise ValueError("input must have shape [batch, tokens, d_model]")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("batch and token counts must be positive; empty input is unsupported")
        if not x.is_floating_point():
            raise ValueError("input must be floating point")
        if positions is None:
            start = self.state_manager.next_position(state)
            positions = torch.arange(start, start + x.shape[1], device=x.device)
        elif (
            positions.ndim != 1
            or positions.numel() != x.shape[1]
            or positions.dtype != torch.int64
            or positions.device != x.device
        ):
            raise ValueError("positions must be int64 [tokens] on the input device")
        if bool((positions < 0).any()) or bool((positions[1:] <= positions[:-1]).any()):
            raise ValueError("positions must be nonnegative and strictly increasing")
        qkv = self.projection.project(x)
        for module in self.hooks:
            hook = cast(AttentionHook, module)
            qkv = hook.after_projection(qkv, positions)
        prepared = self.state_manager.prepare(qkv, positions, state)
        context = self.context_policy.apply(prepared.context, positions)
        for module in self.hooks:
            hook = cast(AttentionHook, module)
            context = hook.before_attention(context, positions)
        attended = self.kernel(
            qkv.q,
            context,
            positions,
            causal=self.causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        for module in self.hooks:
            attended = cast(AttentionHook, module).after_attention(attended)
        output = self.projection.project_output(attended)
        for module in self.hooks:
            output = cast(AttentionHook, module).after_output(output)
        return AttentionOutput(output, prepared.state)
