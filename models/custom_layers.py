import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Any

class TransposeLayer(nn.Module):
    """
    This layer is useful for adding transpose operations to a pipeline to
    make operations easier. It takes the two dimensions to be swapped
    as parameters for the layer.
    """
    def __init__(self, dim1: int, dim2: int) -> None:
        super(TransposeLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Expected input:
        - x : a tensor whose number of dimensions is compatible with
            dim1 and dim2.
        """
        return torch.transpose(x, self.dim1, self.dim2)