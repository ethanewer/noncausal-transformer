from typing import Callable
import torch
from torch import Tensor


class CurriculumGenerator:
    def __init__(self) -> None:
        """Base class for data generator."""
        self.start_dim = None
        self.end_dim = None
        self.interval = None
        self.increment = None
        self.rand = None
        self.dim = None
        self.n_iters = None

    def get_function(self, n_dim: int) -> Callable[[Tensor], Tensor]:
        """Returns function for model to learn."""
        raise NotImplementedError()

    def generate(self, increment_n_iters=True) -> tuple[Tensor, Tensor]:
        """Returns `x`, where `x` has `2 * dim` examples of function inputs and outputs."""
        if self.n_iters == self.interval and self.dim < self.end_dim:
            self.dim += self.increment
            self.n_iters = 0

        if increment_n_iters:
            self.n_iters += 1

        f = self.get_function(self.dim)
        x = []
        for _ in range(2 * self.dim):
            a = self.rand(self.dim)
            x.append(self.pad(a))
            x.append(self.pad(f(a)))

        return torch.stack(x)

    def generate_batch(self, n_batch: int) -> tuple[Tensor, Tensor]:
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
