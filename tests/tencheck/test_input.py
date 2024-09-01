import inspect

import torch

from tencheck.examples import (
    DataclassLinReluModule,
    ListLinReluModule,
    OptionalSimpleLinReluModule,
    SimpleLinReluModule,
    SpecifiedLinReluModule,
    VariadicLinReluModule,
)
from tencheck.input import _extract_dim_names, _resolve_signature, input_gen


def test_dim_name_extraction():
    layer = SimpleLinReluModule(5)
    names = _extract_dim_names(inspect.signature(layer.forward))
    assert names == set("B")

    layer = DataclassLinReluModule(5)
    names = _extract_dim_names(inspect.signature(layer.forward))
    assert names == {"B", "C", "D"}


def test_resolution():
    layer = SimpleLinReluModule(5)
    kwargs = _resolve_signature(inspect.signature(layer.forward), assigned_dimensions={"B": 3}, device=torch.device("cpu"))
    assert kwargs["x"].shape == (3, 32)

    layer = DataclassLinReluModule(5)
    kwargs = _resolve_signature(inspect.signature(layer.forward), assigned_dimensions={"B": 3, "C": 4, "D": 5}, device=torch.device("cpu"))
    assert kwargs["x"].one.shape == (3, 32)


def test_shape_size():
    layer = SimpleLinReluModule(5)
    kwargs = input_gen(layer)
    x = kwargs["x"]
    assert x.shape[0] < 17
    assert x.shape[1] == 32

    layer = OptionalSimpleLinReluModule(5)
    kwargs = input_gen(layer)
    x = kwargs["x"]
    assert x.shape[0] < 17
    assert x.shape[1] == 32

    layer = ListLinReluModule(5)
    kwargs = input_gen(layer)
    x = kwargs["x"]
    assert x[0].shape[0] < 17
    assert x[0].shape[1] == 32
    assert len(x) == 2

    layer = DataclassLinReluModule(5)
    kwargs = input_gen(layer)
    x = kwargs["x"]
    assert x.one.shape[0] < 17
    assert x.one.shape[1] == 32
    assert len(x.two) == 2

    layer = SpecifiedLinReluModule(17, 3)
    kwargs = input_gen(layer)
    x = kwargs["x"]
    assert x.shape[1] == 17

    layer = VariadicLinReluModule(5)
    kwargs = input_gen(layer)
    x = kwargs["x"]
    assert x.shape[0] < 17
    assert x.shape[1] == 32
