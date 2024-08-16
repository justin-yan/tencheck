from torch import Tensor
from torch import nn as nn

from tencheck.type import TensorContainerTypes


def unused_params_check(layer: nn.Module) -> None:
    """
    This check is run after a backward pass is completed.

    If any unused parameters are found, an exception is thrown with the named parameters.
    """
    unused_parameters = []
    for name, param in layer.named_parameters():
        if param.grad is None:
            unused_parameters.append(name)

    assert len(unused_parameters) == 0, f"Unused parameters: {unused_parameters} detected."


def device_meta_check(out: TensorContainerTypes) -> None:
    """
    This check runs after the meta device forward/backward pass, and validates that the output tensor is on the meta device.

    This serves as a quick sanity check that any tensors initialized within the forward pass respect the device of
      the layers and inputs
    """
    match out:
        case Tensor():
            assert out.device.type == "meta", "Layer output was not correctly on meta device."
        case list() | tuple():
            device_meta_check(out[0])
        case set():
            device_meta_check(out.pop())
        case dict():
            device_meta_check([v for k, v in out.items()][0])
        case _:
            raise NotImplementedError("Unexpected input format")
