"""limn: a small, readable deep learning framework. Lazy tensors, autograd, numpy reference device."""

from limn.device import set_device
from limn.ops import float32, int32
from limn.tensor import Tensor, no_grad, realize, set_seed

__all__ = ["Tensor", "no_grad", "realize", "set_device", "set_seed", "float32", "int32"]
