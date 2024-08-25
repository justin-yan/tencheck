import pytest
from jaxtyping import TypeCheckError

from tencheck.examples import (
    BoolModule,
    BrokenModule,
    CasedLinReluModule,
    HardcodedDeviceModule,
    MistypedModule,
    SimpleLinReluModule,
    UnusedParamsModule,
)
from tencheck.harness import _single_layer_assert_all, check_layers


def test_single_layer():
    _single_layer_assert_all(SimpleLinReluModule(5))

    with pytest.raises(Exception, match="Module is broken"):
        _single_layer_assert_all(BrokenModule())

    with pytest.raises(TypeCheckError):
        _single_layer_assert_all(MistypedModule(5))

    with pytest.raises(AssertionError, match="Unused parameters"):
        _single_layer_assert_all(UnusedParamsModule(5))

    with pytest.raises(RuntimeError, match="is not on the expected device meta"):
        _single_layer_assert_all(HardcodedDeviceModule(5))


def test_layers_check():
    check_layers([SimpleLinReluModule(5)])

    with pytest.raises(Exception, match="Module is broken"):
        check_layers([BrokenModule()])

    with pytest.raises(TypeCheckError):
        check_layers([MistypedModule(5)])


def test_cased_layers():
    check_layers([SimpleLinReluModule(5), CasedLinReluModule, BoolModule(5)])
