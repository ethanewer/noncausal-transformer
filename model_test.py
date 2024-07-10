import sys
import time
import torch
import torch.nn.functional as F
from model import (
    DecoderTransformer,
    DecoderTransformerStack,
    DecoderTransformerConfig,
)


def test_decoder_transformer_stack(verbose=False) -> None:
    config = DecoderTransformerConfig(
        n_embd=512, n_head=4, n_layer=8, is_causal=True, loss_fn=F.mse_loss
    )

    model = DecoderTransformerStack(config)

    x = torch.randn(10, 20, 512)
    y = torch.randn(10, 20, 512)

    y1, l1 = model._DecoderTransformerStack__causal_forward(
        x, y, backward=False, forward_idxs=None
    )
    y2, l2 = model._DecoderTransformerStack__noncausal_forward(
        x, y, backward=False, forward_idxs=None
    )

    if verbose:
        print(
            f"DecoderTransformerStack outputs mse: {F.mse_loss(y1, y2).detach().item()}"
        )
        print(
            f"DecoderTransformerStack losess mse: {F.mse_loss(l1, l2).detach().item()}"
        )

    assert torch.allclose(
        y1, y2, rtol=1e-4, atol=1e-5
    ), "DecoderTransformerStack outputs don't match"
    assert torch.allclose(l1, l2), "DecoderTransformerStack losses don't match"


def test_decoder_transformer(verbose=False) -> None:
    config = DecoderTransformerConfig(n_embd=512, n_head=4, n_layer=8, is_causal=True)

    model = DecoderTransformer(config)

    x = torch.randn(10, 20, 512)
    y = torch.randint(config.vocab_size, (10, 20))

    y1, l1 = model._DecoderTransformer__causal_forward(x, y, backward=False)
    y2, l2 = model._DecoderTransformer__noncausal_forward(x, y, backward=False)

    if verbose:
        print(f"DecoderTransformer outputs mse: {F.mse_loss(y1, y2).detach().item()}")
        print(f"DecoderTransformer losess mse: {F.mse_loss(l1, l2).detach().item()}")

    assert torch.allclose(
        y1, y2, rtol=1e-4, atol=1e-5
    ), "DecoderTransformer outputs don't match"
    assert torch.allclose(l1, l2), "DecoderTransformer losses don't match"


def test_decoder_transformer_stack_gradients(verbose=False) -> None:
    config = DecoderTransformerConfig(
        n_embd=512, n_head=4, n_layer=8, is_causal=True, loss_fn=F.mse_loss
    )

    model = DecoderTransformerStack(config)

    x = torch.randn(10, 20, 512)
    y = torch.randn(10, 20, 512)

    model._DecoderTransformerStack__causal_forward(
        x, y, backward=True, forward_idxs=None
    )
    grads1 = {}
    for p_name, p in model.named_parameters():
        if p.grad is not None:
            grads1[p_name] = p.grad.clone()
            p.grad.zero_()

    model._DecoderTransformerStack__noncausal_forward(
        x, y, backward=True, forward_idxs=None
    )
    grads2 = {}
    for p_name, p in model.named_parameters():
        if p.grad is not None:
            grads2[p_name] = p.grad.clone()
            p.grad.zero_()

    for p_name, _ in model.named_parameters():
        if p.grad is not None:
            g1 = grads1[p_name]
            g2 = grads2[p_name]
            if verbose:
                mse = F.mse_loss(g1, g2).item()
                print(f"DecoderTransformerStack[{p_name}] grad mse: {mse}")

            assert torch.allclose(
                g1, g2, rtol=1e-4, atol=1e-5
            ), "DecoderTransformerStack gradients don't match"


def test_decoder_transformer_gradients(verbose=False) -> None:
    config = DecoderTransformerConfig(n_embd=512, n_head=4, n_layer=8, is_causal=True)

    model = DecoderTransformer(config)

    x = torch.randn(10, 20, 512)
    y = torch.randint(config.vocab_size, (10, 20))

    model._DecoderTransformer__causal_forward(x, y, backward=True)
    grads1 = {}
    for p_name, p in model.named_parameters():
        if p.grad is not None:
            grads1[p_name] = p.grad.clone()
            p.grad.zero_()

    model._DecoderTransformer__noncausal_forward(x, y, backward=True)
    grads2 = {}
    for p_name, p in model.named_parameters():
        if p.grad is not None:
            grads2[p_name] = p.grad.clone()
            p.grad.zero_()

    for p_name, p in model.named_parameters():
        if p.grad is not None:
            g1 = grads1[p_name]
            g2 = grads2[p_name]
            if verbose:
                print(
                    f"DecoderTransformer[{p_name}] grad mse: {F.mse_loss(g1, g2).item()}"
                )

            assert torch.allclose(
                g1, g2, rtol=1e-4, atol=1e-5
            ), "DecoderTransformer gradients don't match"


def test_decoder_transformer_stack_forward_idxs(verbose=False) -> None:
    config = DecoderTransformerConfig(
        n_embd=512, n_head=4, n_layer=8, is_causal=True, loss_fn=F.mse_loss
    )

    model = DecoderTransformerStack(config)

    x = torch.randn(10, 20, 512)
    y = torch.randn(10, 20, 512)
    forward_idxs = [i for i in range(20) if i % 2]

    y1, l1 = model._DecoderTransformerStack__causal_forward(
        x, y, backward=False, forward_idxs=forward_idxs
    )
    y2, l2 = model._DecoderTransformerStack__noncausal_forward(
        x, y, backward=False, forward_idxs=forward_idxs
    )
    y3 = model(x)[0][:, forward_idxs, :]
    l3 = F.mse_loss(y3, y[:, forward_idxs, :])

    if verbose:
        print(
            f"DecoderTransformerStack outputs mse: {F.mse_loss(y1, y2).detach().item()}"
        )
        print(
            f"DecoderTransformerStack losess mse: {F.mse_loss(l1, l2).detach().item()}"
        )
        print(
            f"DecoderTransformerStack outputs mse: {F.mse_loss(y2, y3).detach().item()}"
        )
        print(
            f"DecoderTransformerStack losess mse: {F.mse_loss(l2, l3).detach().item()}"
        )

    assert torch.allclose(
        y1, y2, rtol=1e-4, atol=1e-5
    ), "DecoderTransformerStack outputs don't match"
    assert torch.allclose(l1, l2), "DecoderTransformerStack losses don't match"
    assert torch.allclose(
        y2, y3, rtol=1e-4, atol=1e-5
    ), "DecoderTransformerStack outputs don't match"
    assert torch.allclose(l2, l3), "DecoderTransformerStack losses don't match"


def test_decoder_transformer_stack_gradients_forward_idxs(verbose=False) -> None:
    config = DecoderTransformerConfig(
        n_embd=512, n_head=4, n_layer=8, is_causal=True, loss_fn=F.mse_loss
    )

    model = DecoderTransformerStack(config)

    x = torch.randn(10, 20, 512)
    y = torch.randn(10, 20, 512)
    forward_idxs = [i for i in range(20) if i % 2]

    model._DecoderTransformerStack__causal_forward(
        x, y, backward=True, forward_idxs=forward_idxs
    )
    grads1 = {}
    for p_name, p in model.named_parameters():
        if p.grad is not None:
            grads1[p_name] = p.grad.clone()
            p.grad.zero_()

    model._DecoderTransformerStack__noncausal_forward(
        x, y, backward=True, forward_idxs=forward_idxs
    )
    grads2 = {}
    for p_name, p in model.named_parameters():
        if p.grad is not None:
            grads2[p_name] = p.grad.clone()
            p.grad.zero_()

    for p_name, _ in model.named_parameters():
        if p.grad is not None:
            g1 = grads1[p_name]
            g2 = grads2[p_name]
            if verbose:
                mse = F.mse_loss(g1, g2).item()
                print(f"DecoderTransformerStack[{p_name}] grad mse: {mse}")

            assert torch.allclose(
                g1, g2, rtol=1e-4, atol=1e-5
            ), "DecoderTransformerStack gradients don't match"


def time_transformer_iter(device: str) -> None:
    config = DecoderTransformerConfig(
        block_size=256,
        n_layer=8,
        n_head=4,
        n_embd=512,
        is_causal=False,
    )

    model = DecoderTransformer(config).to(device)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=0.01, betas=(0.9, 0.99), device_type=device
    )

    x = torch.randint(config.vocab_size, (10, 20), device=device)
    y = torch.randint(config.vocab_size, (10, 20), device=device)
    t0 = time.time()

    model(x, y, backward=True)
    optimizer.step()
    optimizer.zero_grad()

    print(f"transformer iteration time: {time.time() - t0:.2f}s")


def time_transformer_stack_iter(device: str) -> None:
    config = DecoderTransformerConfig(
        block_size=256,
        n_layer=8,
        n_head=4,
        n_embd=512,
        is_causal=False,
        loss_fn=F.mse_loss,
    )

    model = DecoderTransformerStack(config).to(device)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=0.01, betas=(0.9, 0.99), device_type=device
    )

    x = torch.randn(10, 20, 512, device=device)
    y = torch.randn(10, 20, 512, device=device)
    t0 = time.time()

    model(x, y, backward=True)
    optimizer.step()
    optimizer.zero_grad()

    print(f"transformer stack iteration time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    verbose = "-v" in sys.argv

    test_decoder_transformer_stack(verbose)
    test_decoder_transformer(verbose)
    test_decoder_transformer_stack_gradients(verbose)
    test_decoder_transformer_gradients(verbose)
    test_decoder_transformer_stack_forward_idxs(verbose)
    test_decoder_transformer_stack_gradients_forward_idxs(verbose)
    print("Numeric tests passed")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    time_transformer_iter(device)
    time_transformer_stack_iter(device)
