from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class CaseDefined(Protocol):
    @property
    def _tencheck_cases(self) -> List[Dict[str, Any]]:
        raise NotImplementedError()


@dataclass
class LayerStats:
    total_time: float
    peak_mem_gigs: float
    giga_flop_count: float


TensorContainerTypes = Tensor | list | set | tuple | dict
