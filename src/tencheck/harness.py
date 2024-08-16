import time
from typing import Optional, Type

import torch
from torch import nn as nn

from tencheck.checks import device_meta_check, unused_params_check
from tencheck.input import input_gen
from tencheck.loss import trivial_loss
from tencheck.torchviz import make_dot  # type: ignore
from tencheck.type import CaseDefined, LayerStats, TensorContainerTypes


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
    # throws Exception for generic issues
    # throws TypeCheckError for shapecheck
    in_tens = input_gen(layer, seed, device="cpu")
    out, loss = _forward_backward(layer, in_tens)
    unused_params_check(layer)

    layer.to(device="meta")
    m_in_tens = input_gen(layer, seed, device="meta")
    m_out, m_loss = _forward_backward(layer, m_in_tens)
    device_meta_check(m_out)


def _forward_backward(layer: nn.Module, in_tens: dict[str, torch.Tensor]) -> tuple[TensorContainerTypes, torch.Tensor]:
    layer.zero_grad(set_to_none=True)
    out = layer.forward(**in_tens)
    loss = trivial_loss(out)
    loss.backward()
    return out, loss


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


def graph_layer(layer: nn.Module) -> str:
    """
    Requires graphviz as a general dependency.
    """
    in_tens = input_gen(layer)
    out, loss = _forward_backward(layer, in_tens)
    d = make_dot(loss, params=dict(layer.named_parameters()), show_attrs=True, show_saved=True)

    output_name = f"/tmp/{layer._get_name()}.gv"
    d.render(output_name, format="png")
    return output_name


if __name__ == "__main__":
    from tencheck.examples import SimpleLinReluModule

    d = graph_layer(SimpleLinReluModule(5))
    print(d)
