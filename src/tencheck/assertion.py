from typing import Optional

import torch.nn as nn

from tencheck.input import input_gen
from tencheck.loss import trivial_loss


def layer_object_assertion(layers: list[nn.Module], seed: Optional[int] = None) -> None:
    """
    This method receives a *concrete* list of layer objects, and asserts the relevant properties.
    """
    for layer in layers:
        in_tens = input_gen(layer, seed)
        out = layer.forward(**in_tens)
        loss = trivial_loss(out)
        loss.backward()
