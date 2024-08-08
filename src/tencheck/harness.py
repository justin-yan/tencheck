import time
from typing import Optional, Type

import torch
from torch import nn as nn

from tencheck.checks import unused_params_check
from tencheck.input import input_gen
from tencheck.loss import trivial_loss
from tencheck.ttypes import CaseDefined, LayerStats


def check_layers(layers: list[nn.Module | Type[CaseDefined]], seed: Optional[int] = None) -> None:
    """
    This method receives a *concrete* list of layer objects, and asserts the relevant properties.
    """
    for layer in layers:
        if isinstance(layer, CaseDefined):  # This works even though layer is a class, not an obj, due to @runtime_checkable
            for case in layer._tencheck_cases:
                layer_obj = layer(**case)
                _single_layer_assert_all(layer_obj, seed)
        else:
            _single_layer_assert_all(layer, seed)  # type: ignore[arg-type]


def _single_layer_assert_all(layer: nn.Module, seed: Optional[int] = None) -> None:
    in_tens = input_gen(layer, seed)
    layer.zero_grad(set_to_none=True)

    # throws TypeCheckError for shapecheck
    # throws Exception for generic issues
    out = layer.forward(**in_tens)
    loss = trivial_loss(out)
    loss.backward()

    unused_params_check(layer)


def profile_layer(layer: nn.Module) -> LayerStats:
    """
    This runs a basic profiling setup for a single layer.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise Exception("GPU is not available")

    layer = layer.to(device)
    layer.zero_grad(set_to_none=True)
    in_tens = input_gen(layer, device=device)

    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    out = layer.forward(**in_tens)
    loss = trivial_loss(out)
    loss.backward()
    elapsed = time.perf_counter() - start
    peak_memory_gbs = torch.cuda.max_memory_allocated() / 1024**3
    gigaflops = 0.0

    return LayerStats(elapsed, peak_memory_gbs, gigaflops)


if __name__ == "__main__":
    from tencheck.examples import SimpleLinReluModule

    print(profile_layer(SimpleLinReluModule(5)))
