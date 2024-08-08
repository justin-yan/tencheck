from torch import nn as nn


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
