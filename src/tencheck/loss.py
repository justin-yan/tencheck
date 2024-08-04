import torch
from torch import Tensor

TensorContainerTypes = Tensor | list | set | tuple | dict


def flatten_tensors(tct: TensorContainerTypes) -> list[Tensor]:
    """
    This method will recursively traverse any potential container structures and extract any tensors as a flat list.
    """
    tensor_list: list[Tensor] = []
    match tct:
        case Tensor():
            tensor_list.append(tct)
        case list():
            for e in tct:
                match e:
                    case Tensor():
                        tensor_list.append(e)
                    case _:
                        tensor_list.extend(flatten_tensors(e))
        case set() | tuple():
            tensor_list.extend(flatten_tensors(list(tct)))
        case dict():
            tensor_list.extend(flatten_tensors([v for k, v in tct.items()]))
        case _:
            raise NotImplementedError("Unexpected input format")

    return tensor_list


def trivial_loss(tct: TensorContainerTypes) -> Tensor:
    partial_sums = [t.sum() for t in flatten_tensors(tct)]
    return torch.stack(partial_sums).sum()
