import torch
from torch import Tensor
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
    def __init__(self, text_path: str, block_size: int) -> None:
        self.block_size = block_size

        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char2int = {c: i for i, c in enumerate(chars)}
        self.int2char = {i: c for i, c in enumerate(chars)}

        encoded_text = torch.tensor(self.encode(text), dtype=torch.int64)
        self.idxs, self.target_idxs = self.block_data(encoded_text)

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, i) -> tuple[Tensor, Tensor]:
        return self.idxs[i], self.target_idxs[i]

    def encode(self, s: str) -> list[int]:
        return [self.char2int[c] for c in s if c in self.char2int]

    def decode(self, y: list[int] | Tensor) -> str:
        return "".join([self.int2char[int(i)] for i in y if int(i) in self.int2char])

    def block_data(self, data: Tensor) -> tuple[Tensor, Tensor]:
        n_blocks = len(data) - self.block_size - 1
        x = torch.stack([data[i : i + self.block_size] for i in range(n_blocks)])
        y = torch.stack([data[i : i + self.block_size] for i in range(1, n_blocks + 1)])
        return x, y
