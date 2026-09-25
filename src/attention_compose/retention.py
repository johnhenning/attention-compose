from abc import ABC, abstractmethod

import torch
from torch import Tensor


class RetentionPolicy(ABC):
    @abstractmethod
    def indices(self, positions: Tensor) -> Tensor:
        """Return sorted, unique indices into the available cache."""
        ...


class FullRetention(RetentionPolicy):
    def indices(self, positions: Tensor) -> Tensor:
        return torch.arange(positions.numel(), device=positions.device)
