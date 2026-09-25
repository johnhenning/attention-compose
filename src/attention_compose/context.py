from abc import ABC, abstractmethod

from torch import Tensor

from .types import AttentionContext


class ContextPolicy(ABC):
    @abstractmethod
    def apply(self, context: AttentionContext, query_positions: Tensor) -> AttentionContext: ...


class FullContext(ContextPolicy):
    def apply(self, context: AttentionContext, query_positions: Tensor) -> AttentionContext:
        return context
