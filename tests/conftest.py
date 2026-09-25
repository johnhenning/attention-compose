import pytest
import torch


@pytest.fixture(autouse=True)
def seed() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(1)
