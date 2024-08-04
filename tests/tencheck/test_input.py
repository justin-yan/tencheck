from tencheck.examples import SimpleLinReluModule
from tencheck.input import input_gen


def test_shape_size():
    layer = SimpleLinReluModule(5)
    kwargs = input_gen(layer)
    x = kwargs["x"]
    assert x.shape[0] < 65
    assert x.shape[1] == 32
