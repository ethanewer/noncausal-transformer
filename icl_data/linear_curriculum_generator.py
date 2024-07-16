from typing import Callable
import torch
from torch import Tensor
from .curriculum_generator import CurriculumGenerator


class LinearCurriculumGenerator(CurriculumGenerator):
    def __init__(
        self,
        start_dim: int,
        end_dim: int,
        interval: int,
        increment: int,
        x_rand: Callable[[int], Tensor] = torch.randn,
        function_rand: Callable[[int, int], Tensor] = torch.randn,
    ) -> None:
        """Generates data using a linear mapping from `input_dim` to `output_dim`."""
        super().__init__(start_dim, end_dim, interval, increment, rand=x_rand)
        self.function_rand = function_rand

    def get_function(self, n_dim: int) -> Callable[[Tensor], Tensor]:
        """Return linear mapping from `input_dim` to `output_dim`."""
        w = self.function_rand(n_dim, 1)
        return lambda a: a @ w
