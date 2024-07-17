import math
import torch
from torch import nn, optim, Tensor
from .base import *


class DecoderTransformerStack_ICL(nn.Module):
    """Decoder transformer model without embeddings or lm-head, with optional causality."""

    def __init__(self, config: DecoderTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.is_causal = config.is_causal
        self.loss_fn = self.config.loss_fn
        self.stack = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.stack[0].attn.is_causal = True
        
        # new updates based on the paper
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.read_in = nn.Linear(config.n_dim, config.n_embd)
        self.read_out = nn.Linear(config.n_embd, 1)
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        
        self.apply(self.__init_weights)
        for p_name, p in self.named_parameters():
            if p_name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    @staticmethod
    def __init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def __causal_forward(
        self,
        x: Tensor,
        y: Tensor | None,
        backward: bool,
        forward_idxs: list[int] | None,
    ) -> tuple[Tensor, float | None]:
        """Helper method for forward. Causal forward is O(N^2.)"""
        for block in self.stack:
            x = block(x)

        if forward_idxs is not None:
            x = x[:, forward_idxs, :] 
        
        x = self.ln_f(x)
        x = self.read_out(x)

        if y is not None and self.loss_fn is not None:
            if forward_idxs is not None:
                y = y[:, forward_idxs, :]

            loss = self.loss_fn(x[:,:,0], y[:,:,0])

            if backward:
                loss.backward()
        else:
            loss = None

        return x, loss

    def __noncausal_forward(
        self,
        x: Tensor,
        y: Tensor | None,
        backward: bool,
        forward_idxs: list[int] | None,
    ) -> tuple[Tensor, float | None]:
        """Helper method for forward. Noncausal forward is O(N^3.)"""
        if forward_idxs is None:
            forward_idxs = range(x.shape[1])

        logits = []
        loss_sum = 0
        x = self.stack[0](x)
        for t in forward_idxs:
            x_t = x[:, : t + 1, :]
            for block in self.stack[1:]:  # type: ignore
                x_t = block(x_t)
                
            x_t = self.ln_f(x_t)
            x_t = self.read_out(x_t)

            if y is not None and self.loss_fn is not None:
                # loss = self.loss_fn(x_t[:, -1, :], y[:, t]) / len(forward_idxs)
                loss = self.loss_fn(x_t[:, -1, 0], y[:, t, 0]) / len(forward_idxs)
                loss_sum += loss.detach()
                if backward:
                    loss.backward(retain_graph=True)

            logits.append(x_t[:, -1, :].detach())

        logits = torch.stack(logits, dim=1)
        if y is not None and self.loss_fn is not None:
            loss = loss_sum
        else:
            loss = None
        return logits, loss

    def forward(
        self,
        x: Tensor,
        y=None,
        backward=False,
        forward_idxs: list[int] | None = None,
    ) -> tuple[Tensor, float | None]:
        assert (
            x.shape[1] <= self.config.block_size
        ), f"Cannot forward sequence of length {x.shape[1]} > {self.config.block_size}"

        if backward:
            assert y is not None

        T = x.shape[1]
        pos_idxs = torch.arange(0, T, dtype=torch.int64, device=x.device)
        pos_emb = self.wpe(pos_idxs)
        x = self.read_in(x)
        x = x + pos_emb
        
        if self.is_causal:
            return self.__causal_forward(x, y, backward, forward_idxs)
        else:
            return self.__noncausal_forward(x, y, backward, forward_idxs)

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> optim.Optimizer:
        """Configures AdamW optimizer."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        return optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
