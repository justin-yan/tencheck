import pytest

from tencheck.assertion import layer_object_assertion
from tencheck.examples import BrokenModule, SimpleLinReluModule


def test_layer_check():
    layer_object_assertion([SimpleLinReluModule(5)])


def test_layer_check_broken():
    with pytest.raises(Exception):
        layer_object_assertion([BrokenModule()])
