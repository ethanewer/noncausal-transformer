from typing import Callable
import torch
from torch import Tensor

torch.manual_seed(0)


class BaseGenerator:
    def __init__(self, rand=torch.randn) -> None:
        """Base class for data generator."""
        self.input_dim = None
        self.rand = rand

    def get_function(self) -> Callable[[Tensor], Tensor]:
        """Returns function for model to learn."""
        raise NotImplementedError()

    def generate(self, n_examples: int) -> tuple[Tensor, Tensor]:
        """
        Returns `x, y`. `x` has `n_examples` examples of function inputs and outputs,
        followed by a single function input. `y` has the function output for the last input in `x`.
        """
        f = self.get_function()
        x = []
        for _ in range(n_examples):
            x.append(self.rand(self.input_dim))
            x.append(f(x[-1]))

        x.append(self.rand(self.input_dim))
        y = f(x[-1])
        x = torch.stack(x)
        return x, y

    def generate_batch(self, n_batch: int, n_examples: int) -> tuple[Tensor, Tensor]:
        """Returns a batch of batch of `x, y` pairs from `generate` method."""
        x_batch = []
        y_batch = []
        for _ in range(n_batch):
            x, y = self.generate(n_examples)
            x_batch.append(x)
            y_batch.append(y)

        return torch.stack(x_batch), torch.stack(y_batch)


class LinearGenerator(BaseGenerator):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        x_rand=torch.randn,
        function_rand=torch.randn,
    ) -> None:
        """Generates data using a linear mapping from `input_dim` to `output_dim`."""
        super().__init__()
        assert input_dim >= output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rand = x_rand
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
        b = self.function_rand(self.output_dim)
        return lambda a: self.pad(a @ w + b)
