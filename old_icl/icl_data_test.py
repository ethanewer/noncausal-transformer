from icl_data import LinearCurriculumGenerator
import torch
import torch.nn.functional as F
import sys


def test_lcg_zeros():
    lcg = LinearCurriculumGenerator(1, 3, 1, 1)
    x = torch.cat([lcg.generate() for _ in range(4)], dim=0)

    zeros = torch.tensor(
        [
            # first generate
            [False, True, True],
            [False, True, True],
            [False, True, True],
            [False, True, True],
            # second generate
            [False, False, True],
            [False, True, True],
            [False, False, True],
            [False, True, True],
            [False, False, True],
            [False, True, True],
            [False, False, True],
            [False, True, True],
            # third generate
            [False, False, False],
            [False, True, True],
            [False, False, False],
            [False, True, True],
            [False, False, False],
            [False, True, True],
            [False, False, False],
            [False, True, True],
            [False, False, False],
            [False, True, True],
            [False, False, False],
            [False, True, True],
            # fourth generate
            [False, False, False],
            [False, True, True],
            [False, False, False],
            [False, True, True],
            [False, False, False],
            [False, True, True],
            [False, False, False],
            [False, True, True],
            [False, False, False],
            [False, True, True],
            [False, False, False],
            [False, True, True],
        ]
    )

    assert torch.all(x[zeros] == 0)


def test_lcg_linear_model(verbose=False):
    lcg = LinearCurriculumGenerator(100, 1000, 1, 100)

    for _ in range(25):
        z = lcg.generate()
        x = z[::2, : lcg.start_dim]
        y = z[1::2, 0]
        w, *_ = torch.linalg.lstsq(x, y)
        y_hat = x @ w
        if verbose:
            print(f"R^2: {(1 - F.mse_loss(y, y_hat) / torch.var(y)).item():.4f}")
        assert torch.allclose(y, y_hat, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    test_lcg_zeros()
    test_lcg_linear_model(verbose)
