import sys
import torch
import torch.nn.functional as F
from model import DecoderTransformer, DecoderTransformerStack, DecoderTransformerConfig


def test_decoder_transformer_stack(verbose=False) -> None:
    config = DecoderTransformerConfig(n_embd=512, n_head=4, n_layer=8, 
                                    is_causal=True, loss_fn=F.mse_loss)

    model = DecoderTransformerStack(config)

    x = torch.randn(10, 20, 512)
    y = torch.randn(10, 20, 512)

    y1, l1 = model._DecoderTransformerStack__causal_forward(x, y)
    y2, l2 = model._DecoderTransformerStack__noncausal_forward(x, y)

    if verbose:
        print(f"DecoderTransformerStack outputs mse: {F.mse_loss(y1, y2).detach().item()}")
        print(f"DecoderTransformerStack losess mse: {F.mse_loss(l1, l2).detach().item()}")

    assert torch.allclose(y1, y2, rtol=1e-4, atol=1e-5), "DecoderTransformerStack outputs don't match"
    assert torch.allclose(l1, l2), "DecoderTransformerStack losses don't match"


def test_decoder_transformer(verbose=False) -> None:
    config = DecoderTransformerConfig(n_embd=512, n_head=4, n_layer=8, is_causal=True)

    model = DecoderTransformer(config)

    x = torch.randint(512, (10, 20))
    y = torch.randint(512, (10, 20))

    y1, l1 = model._DecoderTransformer__causal_forward(x, y)
    y2, l2 = model._DecoderTransformer__noncausal_forward(x, y)

    if verbose:
        print(f"DecoderTransformer outputs mse: {F.mse_loss(y1, y2).detach().item()}")
        print(f"DecoderTransformer losess mse: {F.mse_loss(l1, l2).detach().item()}")

    assert torch.allclose(y1, y2, rtol=1e-4, atol=1e-5), "DecoderTransformer outputs don't match"
    assert torch.allclose(l1, l2), "DecoderTransformer losses don't match"


def test_decoder_transformer_stack_gradients(verbose=False) -> None:
    config = DecoderTransformerConfig(n_embd=512, n_head=4, n_layer=8, 
                                    is_causal=True, loss_fn=F.mse_loss)

    model = DecoderTransformerStack(config)

    x = torch.randn(10, 20, 512)
    y = torch.randn(10, 20, 512)

    model._DecoderTransformerStack__causal_forward(x, y, backward=True)
    grads1 = {}
    for p_name, p in model.named_parameters():
        if p.grad is not None:
            grads1[p_name] = p.grad.clone()
            p.grad.zero_()

    model._DecoderTransformerStack__noncausal_forward(x, y, backward=True)
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

            assert torch.allclose(g1, g2, rtol=1e-4, atol=1e-5), \
                "DecoderTransformerStack gradients don't match"

    
def test_decoder_transformer_gradients(verbose=False) -> None:
    config = DecoderTransformerConfig(n_embd=512, n_head=4, n_layer=8, is_causal=True)

    model = DecoderTransformer(config)

    x = torch.randint(512, (10, 20))
    y = torch.randint(512, (10, 20))

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
                print(f"DecoderTransformer[{p_name}] grad mse: {F.mse_loss(g1, g2).item()}")

            assert torch.allclose(g1, g2, rtol=1e-4, atol=1e-5), \
                "DecoderTransformer gradients don't match"


if __name__ == "__main__":
    verbose = "-v" in sys.argv

    test_decoder_transformer_stack(verbose)
    test_decoder_transformer(verbose)
    test_decoder_transformer_stack_gradients(verbose)
    test_decoder_transformer_gradients(verbose)

    print("Tests passed")