import dataclasses
import inspect
import logging
import random
from typing import Any, Optional

import torch
import torch.nn as nn
from jaxtyping._array_types import _FixedDim, _NamedDim, _NamedVariadicDim, _SymbolicDim
from torch import Tensor
from torch.testing import make_tensor

dtype_mapping = {
    "bool": torch.bool,
    "bool_": torch.bool,
    "uint4": torch.int32,
    "uint8": torch.int32,
    "uint16": torch.int32,
    "uint32": torch.int32,
    "uint64": torch.int32,
    "int4": torch.int32,
    "int8": torch.int32,
    "int16": torch.int32,
    "int32": torch.int32,
    "int64": torch.int32,
    "bfloat16": torch.float32,
    "float16": torch.float32,
    "float32": torch.float32,
    "float64": torch.float32,
    "complex64": torch.float32,
    "complex128": torch.float32,
}


def input_gen(layer: nn.Module, seed: Optional[int] = None, device: str | torch.device = "cpu") -> dict[str, Tensor]:
    """
    For a given layer that is type annotated with jaxtyping, produce a map of mock tensors that can be used like so:

    in_tens = input_gen(layer)
    layer.forward(**in_tens)
    """
    signature = inspect.signature(layer.forward)
    dim_names = _extract_dim_names(signature)

    if seed:
        torch.manual_seed(seed)
        random.seed(seed)

    assigned_dimensions = {}
    for dim in dim_names:
        if hasattr(layer, dim) and isinstance(getattr(layer, dim), int):
            assigned_dimensions[dim] = getattr(layer, dim)
        else:
            assigned_dimensions[dim] = random.randint(4, 16)

    return _resolve_signature(signature, assigned_dimensions, device)


def _extract_dim_names(s: inspect.Signature) -> list[str]:
    dim_names: list[str] = []
    for name, param_obj in s.parameters.items():
        obj_type = param_obj.annotation

        if hasattr(obj_type, "dims"):  # This is a very imperfect check for jaxtyped torch tensors
            # A jaxtyped tensor can be one of `_NamedDim`, `_FixedDim`, `_NamedVariadicDim`, or `_SymbolicDim`
            for dim in obj_type.dims:
                match dim:
                    case object() if type(dim) is object:
                        dim_names.append("ellipsis")
                    case _NamedDim(nm, _, _):
                        dim_names.append(nm)
                    case _FixedDim(_, _):
                        pass
                    case _NamedVariadicDim() | _SymbolicDim():
                        raise NotImplementedError("Don't yet handle these dimension cases")
                    case _:
                        raise Exception("Unknown Type")
        elif dataclasses.is_dataclass(obj_type):
            dim_names.extend(_extract_dim_names(inspect.signature(obj_type)))
        else:
            raise Exception("Unhandled Type")

    return dim_names


def _resolve_signature(s: inspect.Signature, assigned_dimensions: dict[str, int], device: str | torch.device) -> dict[str, Any]:
    resolved_kwargs: dict[str, Any] = {}

    for name, param_obj in s.parameters.items():
        obj_type = param_obj.annotation

        if hasattr(obj_type, "dims"):  # This is a very imperfect check for jaxtyped torch tensors
            dtype = dtype_mapping[param_obj.annotation.dtypes[0]]
            shape: list[int] = []

            # A jaxtyped tensor can be one of `_NamedDim`, `_FixedDim`, `_NamedVariadicDim`, or `_SymbolicDim`
            for dim in obj_type.dims:
                match dim:
                    case object() if type(dim) is object:
                        shape.append(assigned_dimensions["ellipsis"])
                    case _NamedDim(nm, _, _):
                        shape.append(assigned_dimensions[nm])
                    case _FixedDim(sz, _):
                        shape.append(sz)  # type: ignore[arg-type]
                    case _NamedVariadicDim() | _SymbolicDim() | _:
                        raise NotImplementedError("Don't yet handle these dimension cases")

            resolved_kwargs[name] = _make_tensor(shape, dtype, device)
        elif dataclasses.is_dataclass(obj_type):
            recursive_kwargs = _resolve_signature(inspect.signature(obj_type), assigned_dimensions, device)
            resolved_kwargs[name] = obj_type(**recursive_kwargs)
        else:
            raise Exception("Unhandled Type")

    return resolved_kwargs


def _make_tensor(shape: list[int], dtype: torch.dtype, device: str | torch.device) -> torch.Tensor:
    match dtype:
        case torch.float32:
            mock_ten = make_tensor(tuple(shape), dtype=dtype, device=device, low=-1, high=1)
        case torch.int32:
            mock_ten = make_tensor(tuple(shape), dtype=dtype, device=device, low=-10, high=10)
        case torch.bool:
            mock_ten = make_tensor(tuple(shape), dtype=dtype, device=device)
        case _:
            logging.error(dtype)
            raise NotImplementedError("Don't yet handle these dtypes")
    return mock_ten
