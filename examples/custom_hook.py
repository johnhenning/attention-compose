"""A trainable hook that participates in state_dict, .to(), and .eval()."""

import torch
from torch import Tensor, nn

from attention_compose import AttentionHook, MultiHeadAttention


class OutputGain(AttentionHook):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(()))

    def after_output(self, output: Tensor) -> Tensor:
        return output * self.gain


def main() -> None:
    model = MultiHeadAttention(32, 4, hooks=[OutputGain()])
    model(torch.randn(2, 6, 32)).output.square().mean().backward()
    parameter = dict(model.named_parameters())["hooks.0.gain"]
    assert parameter.grad is not None
    print("Trainable hook received a gradient.")


if __name__ == "__main__":
    main()
