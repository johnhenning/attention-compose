from abc import ABC, abstractmethod
from typing import Generic

from torch import Tensor

from .types import QKV, AttentionContext, PreparedAttention, StateT


class AttentionStateManager(ABC, Generic[StateT]):
    def next_position(self, state: StateT | None) -> int:
        return 0

    @abstractmethod
    def prepare(
        self, qkv: QKV, positions: Tensor, state: StateT | None
    ) -> PreparedAttention[StateT]:
        """Return the current context and next state without mutating previous state."""
        ...


class StatelessManager(AttentionStateManager[None]):
    def next_position(self, state: None) -> int:
        if state is not None:
            raise ValueError("stateless attention requires state=None")
        return 0

    def prepare(self, qkv: QKV, positions: Tensor, state: None) -> PreparedAttention[None]:
        self.next_position(state)
        return PreparedAttention(AttentionContext(qkv.k, qkv.v, positions), None)
