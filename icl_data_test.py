from icl_data import LinearCurriculumGenerator
import torch
import torch.nn.functional as F

def test_lcg_zeros():
    lcg = LinearCurriculumGenerator(1, 3, 1, 1)
    x = torch.cat([lcg.generate() for _ in range(4)], dim=0)
    
    zeros = torch.tensor([
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
    ])

    assert torch.all(x[zeros] == 0)


def test_lcg_linear_model():
    lcg = LinearCurriculumGenerator(25, 100, 1, 25)

    for _ in range(5):
        z = lcg.generate()
        x = z[::2]
        y = z[1::2, 0]
        w, *_ = torch.linalg.lstsq(x, y)
        y_hat = x @ w
        assert torch.allclose(y, y_hat)


if __name__ == "__main__":
    test_lcg_zeros()
    test_lcg_linear_model()
