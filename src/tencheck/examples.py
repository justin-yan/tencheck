from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Bool, Float


class SimpleLinReluModule(nn.Module):
    def __init__(self, out_features: int) -> None:
        super(SimpleLinReluModule, self).__init__()
        self.linear = nn.Linear(32, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: Float[torch.Tensor, "B 32"]) -> Float[torch.Tensor, "B O"]:
        x = self.linear(x)
        x = self.relu(x)
        return x


class SpecifiedLinReluModule(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super(SpecifiedLinReluModule, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.linear = nn.Linear(c_in, c_out)
        self.relu = nn.ReLU()

    def forward(self, x: Float[torch.Tensor, "B c_in"]) -> Float[torch.Tensor, "B c_out"]:
        x = self.linear(x)
        x = self.relu(x)
        return x


class VariadicLinReluModule(nn.Module):
    def __init__(self, out_features: int) -> None:
        super(VariadicLinReluModule, self).__init__()
        self.linear = nn.Linear(32, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: Float[torch.Tensor, "... 32"]) -> Float[torch.Tensor, "... O"]:
        x = self.linear(x)
        x = self.relu(x)
        return x


class BoolModule(nn.Module):
    def __init__(self, out_features: int) -> None:
        super(BoolModule, self).__init__()
        self.linear = nn.Linear(32, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: Float[torch.Tensor, "B 32"], mask: Bool[torch.Tensor, "B"]) -> Float[torch.Tensor, "B O"]:
        x = x * mask.unsqueeze(1)
        x = self.linear(x)
        x = self.relu(x)
        return x


@dataclass
class Features:
    one: Float[torch.Tensor, "B 32"]
    two: Float[torch.Tensor, "C D"]


class DataclassLinReluModule(nn.Module):
    def __init__(self, out_features: int) -> None:
        super(DataclassLinReluModule, self).__init__()
        self.linear = nn.Linear(32, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: Features) -> Float[torch.Tensor, "B O"]:
        y = self.linear(x.one)
        y = self.relu(y)
        return y


class CasedLinReluModule(nn.Module):
    _tencheck_cases = [{"out_features": 10}, {"out_features": 20}]

    def __init__(self, out_features: int) -> None:
        super(CasedLinReluModule, self).__init__()
        self.linear = nn.Linear(32, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: Float[torch.Tensor, "B 32"]) -> Float[torch.Tensor, "B O"]:
        x = self.linear(x)
        x = self.relu(x)
        return x


class UnusedParamsModule(nn.Module):
    def __init__(self, out_features: int) -> None:
        super(UnusedParamsModule, self).__init__()
        self.linear = nn.Linear(32, out_features)
        self.unused_linear = nn.Linear(32, out_features)
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


class MistypedModule(nn.Module):
    def __init__(self, out_features: int) -> None:
        super(MistypedModule, self).__init__()
        self.linear = nn.Linear(32, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: Float[torch.Tensor, "B 32"]) -> Float[torch.Tensor, "B 1025"]:
        x = self.linear(x)
        x = self.relu(x)
        return x


class HardcodedDeviceModule(nn.Module):
    def __init__(self, out_features: int) -> None:
        super(HardcodedDeviceModule, self).__init__()
        self.linear = nn.Linear(32, out_features)

    def forward(self, x: Float[torch.Tensor, "B 32"]) -> Float[torch.Tensor, "B O"]:
        x = self.linear(x)
        x = x + torch.randn(1, device="cpu")
        return x
