from typing import Callable
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class CurriculumGenerator:
    start_dim: int
    end_dim: int
    interval: int
    increment: int
    rand: Callable[[int], Tensor]
    n_iters: int = 0


    def get_function(self, n_dim: int) -> Callable[[Tensor], Tensor]:
        """Returns function for model to learn."""
        raise NotImplementedError()

    def generate(self, increment_n_iters=True) -> Tensor:
        """Returns `x`, where `x` has `2 * dim` examples of function inputs and outputs."""
        if self.n_iters == self.interval and self.start_dim < self.end_dim:
            self.start_dim += self.increment
            self.n_iters = 0

        if increment_n_iters:
            self.n_iters += 1

        f = self.get_function(self.start_dim)
        x = []
        for _ in range(2 * self.start_dim):
            a = self.rand(self.start_dim)
            x.append(self.pad(a))
            x.append(self.pad(f(a)))

        return torch.stack(x)

    def generate_batch(self, n_batch: int) -> Tensor:
        """Returns a batch from `generate` method."""
        batch = torch.stack(
            [self.generate(increment_n_iters=False) for _ in range(n_batch)]
        )
        self.n_iters += 1
        return batch

    def pad(self, x: Tensor) -> Tensor:
        """Zero pads vector `x` to length `self.input_dim`."""
        if len(x) == self.end_dim:
            return x
        zeros = torch.zeros(self.end_dim - len(x))
        return torch.cat((x, zeros), dim=0)
