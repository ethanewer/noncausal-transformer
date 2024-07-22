from typing import Callable
import torch
from torch import Tensor
from .base_generator import BaseGenerator


class LinearGenerator(BaseGenerator):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        x_rand=torch.randn,
        function_rand=torch.randn,
    ) -> None:
        """Generates data using a linear mapping from `input_dim` to `output_dim`."""
        super().__init__(input_dim, x_rand)
        assert input_dim >= output_dim
        self.output_dim = output_dim
        self.function_rand = function_rand

    def pad(self, x: Tensor) -> Tensor:
        """Zero pads vector `x` to length `self.input_dim`."""
        if len(x) == self.input_dim:
            return x
        zeros = torch.zeros(self.input_dim - len(x))
        return torch.cat((x, zeros), dim=0)

    def get_function(self) -> Callable[[Tensor], Tensor]:
        """Return linear mapping from `input_dim` to `output_dim`."""
        w = self.function_rand(self.input_dim, self.output_dim)
        # b = self.function_rand(self.output_dim)
        # return lambda a: self.pad(a @ w + b)
        return lambda a: self.pad(a @ w)
