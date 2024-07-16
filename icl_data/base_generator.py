from typing import Callable
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class BaseGenerator:
    input_dim: int
    rand: Callable[[int], Tensor] = torch.randn

    def get_function(self) -> Callable[[Tensor], Tensor]:
        """Returns function for model to learn."""
        raise NotImplementedError()

    def generate(self, n_examples: int) -> Tensor:
        """Returns `x`, where `x` has `n_examples` examples of function inputs and outputs."""
        f = self.get_function()
        x = []
        for _ in range(n_examples):
            x.append(self.rand(self.input_dim))
            x.append(f(x[-1]))

        x.append(self.rand(self.input_dim))
        return torch.stack(x)

    def generate_batch(self, n_batch: int, n_examples: int) -> Tensor:
        """Returns a batch from `generate` method."""
        return torch.stack([self.generate(n_examples) for _ in range(n_batch)])
