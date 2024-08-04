import torch
from torch.testing import assert_close, make_tensor

from tencheck.loss import flatten_tensors, trivial_loss


def test_flatten_tensors_preserves_tensor():
    t1 = make_tensor((4, 4), dtype=torch.int8, device="cpu")
    tl = flatten_tensors(t1)
    assert_close(tl[0], t1)


def test_flatten_tensors_preserves_tensor_tuple():
    t1 = make_tensor((4, 4), dtype=torch.int8, device="cpu")
    t2 = make_tensor((4, 4), dtype=torch.int8, device="cpu")
    tl = flatten_tensors((t1, t2))
    assert len(tl) == 2
    assert_close(tl[0], t1)
    assert_close(tl[1], t2)


def test_flatten_tensors_preserves_nested_tensor_list():
    t1 = make_tensor((4, 4), dtype=torch.int8, device="cpu")
    t2 = make_tensor((4, 4), dtype=torch.int8, device="cpu")
    nested_tensor_list = [t1, [t2]]
    tl = flatten_tensors(nested_tensor_list)
    assert len(tl) == 2
    assert_close(tl[0], t1)
    assert_close(tl[1], t2)


def test_flatten_tensors_preserves_nested_tensor_dict():
    t1 = make_tensor((4, 4), dtype=torch.int8, device="cpu")
    t2 = make_tensor((4, 4), dtype=torch.int8, device="cpu")
    nested_tensor_dict = {"a": t1, "b": {"c": t2}}
    tl = flatten_tensors(nested_tensor_dict)
    assert len(tl) == 2
    assert_close(tl[0], t1)
    assert_close(tl[1], t2)


def test_trivial_loss():
    t1 = make_tensor((2, 2), dtype=torch.int8, device="cpu")
    t2 = make_tensor((2, 2), dtype=torch.int8, device="cpu")
    nested_tensor_dict = {"a": t1, "b": {"c": t2}}
    assert_close(trivial_loss(nested_tensor_dict), t1.sum() + t2.sum())
