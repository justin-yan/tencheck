import torch
import torch.nn as nn
from jaxtyping import Float


class SimpleLinReluModule(nn.Module):
    def __init__(self, out_features: int) -> None:
        super(SimpleLinReluModule, self).__init__()
        self.linear = nn.Linear(32, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: Float[torch.Tensor, "B 32"]) -> Float[torch.Tensor, "B O"]:
        x = self.linear(x)
        x = self.relu(x)
        return x


class BrokenModule(nn.Module):
    def __init__(self) -> None:
        super(BrokenModule, self).__init__()
        self.linear = nn.Linear(32, 4)
        self.relu = nn.ReLU()

    def forward(self, x: Float[torch.Tensor, "B 32"]) -> Float[torch.Tensor, "B 4"]:
        x = self.linear(x)
        raise Exception("Module is broken")
        x = self.relu(x)  # type: ignore[unreachable]
        return x
