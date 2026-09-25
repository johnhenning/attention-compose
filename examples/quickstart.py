import torch

from attention_compose import MultiHeadAttention


def main() -> None:
    model = MultiHeadAttention(32, 4).eval()
    result = model(torch.randn(2, 5, 32))
    assert result.output.shape == (2, 5, 32)
    assert result.state is None
    print("Stateless attention example passed.")


if __name__ == "__main__":
    main()
