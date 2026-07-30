"""limn: a small, readable deep learning framework. Lazy tensors, autograd, numpy reference device."""

from limn.device import set_device
from limn.jit import capture
from limn.ops import float16, float32, float64, int8, int16, int32
from limn.tensor import Tensor, no_grad, realize, set_seed

__all__ = [
    "Tensor",
    "capture",
    "no_grad",
    "realize",
    "set_device",
    "set_seed",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
]
