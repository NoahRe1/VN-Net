import math
from typing import Any

import torch
from torch import Tensor


def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, "parameters") else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, "buffers") else []:
            glorot(v)


def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, "parameters") else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, "buffers") else []:
            constant(v, fill_value)


def zeros(value: Any):
    constant(value, 0.0)
