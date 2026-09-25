"""Select entries to preserve for the next call, not the current context."""

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


class SlidingWindowRetention(RetentionPolicy):
    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size

    def indices(self, positions: Tensor) -> Tensor:
        length = positions.numel()
        return torch.arange(max(0, length - self.window_size), length, device=positions.device)


class AttentionSinkRetention(RetentionPolicy):
    """Keep initial entries plus recent entries, within a total token budget."""

    def __init__(self, window_size: int, num_sink_tokens: int = 4) -> None:
        if window_size <= 0 or not 0 <= num_sink_tokens < window_size:
            raise ValueError("require 0 <= num_sink_tokens < positive window_size")
        self.window_size = window_size
        self.num_sink_tokens = num_sink_tokens

    def indices(self, positions: Tensor) -> Tensor:
        length = positions.numel()
        if length <= self.window_size:
            return torch.arange(length, device=positions.device)
        recent = self.window_size - self.num_sink_tokens
        return torch.cat(
            [
                torch.arange(self.num_sink_tokens, device=positions.device),
                torch.arange(length - recent, length, device=positions.device),
            ]
        )
