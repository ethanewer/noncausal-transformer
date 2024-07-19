from contextlib import nullcontext
import time
import argparse
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from model import DecoderTransformerStackICL, DecoderTransformerConfig
from icl_data import LinearCurriculumGenerator


OUT_DIR = "out"

MAX_ITERS = 500000
EVAL_INTERVAL = 500

N_DIM = 20

BLOCK_SIZE = N_DIM * 4
BATCH_SIZE = 128

MIN_LR = 1e-5
MAX_LR = 1e-4
WARMUP_ITERS = 1000
LR_DECAY_ITERS = MAX_ITERS // 2

MODEL_ARCHITECTURES = {
    "tiny": {
        "n_embd": 64,
        "n_layer": 3,
        "n_head": 2,
    },
    "small": {
        "n_embd": 128,
        "n_layer": 6,
        "n_head": 4,
    },
    "standard": {
        "n_embd": 256,
        "n_layer": 12,
        "n_head": 8,
    },
}

MODEL_ARCHITECTURE = MODEL_ARCHITECTURES["tiny"]


def get_lr(iter_num: int) -> float:
    if iter_num < WARMUP_ITERS:
        return MAX_LR * iter_num / WARMUP_ITERS

    if iter_num > LR_DECAY_ITERS:
        return MIN_LR

    decay_ratio = (iter_num - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    assert 0 <= decay_ratio and decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    i: int,
    checkpoint_name: str,
) -> None:
    """Saves model, optimizer, and iteration number of checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "i": i,
    }
    torch.save(checkpoint, f"{OUT_DIR}/checkpoints/{checkpoint_name}.pt")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_name: str,
) -> tuple[nn.Module, optim.Optimizer, int, float]:
    """Returns model, optimizer, iteration number of and checkpoint."""
    checkpoint = torch.load(f"{OUT_DIR}/checkpoints/{checkpoint_name}.pt")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    i = checkpoint["i"]
    return model, optimizer, i


def save_results(
    losses: list[int],
    pointwise_losses: list[list[int]],
    results_name: str,
) -> None:
    """
    Saves losses and pointwise losses. Pointwise losses are padded with NaN and saved as a matrix.
    """
    losses = np.array(losses)
    max_len = max(len(row) for row in pointwise_losses)
    for row in pointwise_losses:
        row.extend([float("nan")] * (max_len - len(row)))
    pointwise_losses = np.array(pointwise_losses)
    np.savez(
        f"{OUT_DIR}/results/{results_name}.npz",
        losses=losses,
        pointwise_losses=pointwise_losses,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ICL training with curriculum learning"
    )

    parser.add_argument(
        "-checkpoint",
        action="store_const",
        const=True,
        default=False,
        help="starts training from checkpoint",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    context = nullcontext() if device == "mps" else torch.autocast(device)
    print(f"using {device} device")

    causal_config = DecoderTransformerConfig(
        block_size=BLOCK_SIZE,
        n_layer=MODEL_ARCHITECTURE["n_layer"],
        n_head=MODEL_ARCHITECTURE["n_head"],
        n_embd=MODEL_ARCHITECTURE["n_embd"],
        n_dim=N_DIM,
        is_causal=True,
        loss_fn=F.mse_loss,
    )

    noncausal_config = DecoderTransformerConfig(
        block_size=BLOCK_SIZE,
        n_layer=MODEL_ARCHITECTURE["n_layer"],
        n_head=MODEL_ARCHITECTURE["n_head"],
        n_embd=MODEL_ARCHITECTURE["n_embd"],
        n_dim=N_DIM,
        is_causal=False,
        loss_fn=F.mse_loss,
    )

    data_generator = LinearCurriculumGenerator(
        start_dim=5, end_dim=20, interval=2000, increment=1
    )

    causal_model = DecoderTransformerStackICL(causal_config).to(device)
    noncausal_model = DecoderTransformerStackICL(noncausal_config).to(device)

    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(
            causal_model.named_parameters(),
            noncausal_model.named_parameters(),
        ):
            assert n1 == n2
            p1.copy_(p2)

    causal_optimizer = causal_model.configure_optimizers(
        weight_decay=0.1, learning_rate=MIN_LR, betas=(0.9, 0.99), device_type=device
    )

    noncausal_optimizer = noncausal_model.configure_optimizers(
        weight_decay=0.1, learning_rate=MIN_LR, betas=(0.9, 0.99), device_type=device
    )

    if args.checkpoint:
        print("loading from checkpoint")
        causal_model, causal_optimizer, i1 = load_checkpoint(
            causal_model, causal_optimizer, "causal"
        )
        noncausal_model, noncausal_optimizer, i2 = load_checkpoint(
            noncausal_model, noncausal_optimizer, "noncausal"
        )
        start_i = min(i1, i2)
    else:
        print("starting from scratch")
        start_i = 0

    models_and_optimizers = [
        (causal_model, causal_optimizer),
        (noncausal_model, noncausal_optimizer),
    ]
    losses = [[], []]
    pointwise_losses = [[], []]

    t0 = time.time()

    for i in range(2):
        data = data_generator.generate_batch(BATCH_SIZE).to(device)
        forward_idxs = [i for i in range(data.shape[1]) if i % 2 == 0]

        x = data[:, :-1, :]
        y = data[:, 1:, :]

        lr = get_lr(i)
        for k, (model, optimizer) in enumerate(models_and_optimizers):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            with context:
                y_hat, loss = model(
                    x, y, backward=True, forward_idxs=forward_idxs, fast_backward=True
                )

            optimizer.step()
            optimizer.zero_grad()

            losses[k].append(loss.detach().cpu())

            unreduced_loss = F.mse_loss(
                y[:, forward_idxs, 0], y_hat[:, :, 0], reduction="none"
            )
            pointwise_loss = torch.mean(unreduced_loss, dim=0)
            pointwise_losses[k].append(pointwise_loss.cpu().tolist())

        if (i + 1) % EVAL_INTERVAL == 0:
            print(f"{f'[{i + 1}]':9}", end="")

            dt = time.time() - t0
            t0 = time.time()

            for k, name in enumerate(["causal", "noncausal"]):
                model, optimizer = models_and_optimizers[k]
                save_checkpoint(model, optimizer, i, name)
                save_results(losses[k], pointwise_losses[k], name)
                mean_loss = np.mean(losses[k][-EVAL_INTERVAL:])
                print(f"{name} loss: {mean_loss:.3f}", end=", ")

            print(f"time: {dt:.1f}s")
