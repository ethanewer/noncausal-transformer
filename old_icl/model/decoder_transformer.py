import math
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from .base import *


class DecoderTransformer(nn.Module):
    """Decoder transformer with optional causality."""

    def __init__(self, config: DecoderTransformerConfig) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.is_causal = config.is_causal
        self.loss_fn = self.config.loss_fn

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        
        self.transformer.h[0].attn.is_causal = True

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

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
    ) -> tuple[Tensor, float | None]:
        """Helper method for forward. Causal forward is O(N^2.)"""
        for block in self.transformer.h[1:]:
            x = block(x)

        x = self.transformer.ln_f(x)
        x = self.lm_head(x)

        if y is not None and self.loss_fn is not None:
            loss = self.loss_fn(x, y)
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
    ) -> tuple[Tensor, float | None]:
        """Helper method for forward. Noncausal forward is O(N^3.)"""
        T = x.shape[1]
        logits = []
        loss_sum = 0

        for t in range(T):
            x_t = x[:, : t + 1, :]
            for block in self.transformer.h[1:]:
                x_t = block(x_t)

            x_t = x_t[:, -1:, :]
            x_t = self.transformer.ln_f(x_t)
            x_t = self.lm_head(x_t)

            if y is not None and self.loss_fn is not None:
                loss = self.loss_fn(x_t, y[:, t]) / T
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
        idxs: Tensor,
        target_idxs=None,
        backward=False,
    ) -> tuple[Tensor, float | None]:
        T = idxs.shape[1]

        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T} > {self.config.block_size}"

        if backward:
            assert target_idxs is not None

        pos_idxs = torch.arange(0, T, dtype=torch.int64, device=idxs.device)
        pos_emb = self.transformer.wpe(pos_idxs)
        tok_emb = self.transformer.wte(idxs)
        x = self.transformer.drop(tok_emb + pos_emb)
        x = self.transformer.h[0](x)

        if self.is_causal:
            return self.__causal_forward(x, target_idxs, backward)
        else:
            return self.__noncausal_forward(x, target_idxs, backward)

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

    @torch.no_grad()
    def generate(self, idxs: Tensor, max_new_tokens: int, temperature=1.0) -> Tensor:
        """
        Take a conditioning sequence of indices `idxs`, `int64` tensor with shape `[B, T]`, and
        completes the sequence `max_new_tokens` times, feeding the predictions back into the model
        each time.
        """
        for _ in range(max_new_tokens):
            if idxs.shape[1] <= self.config.block_size:
                cropped_idxs = idxs
            else:
                cropped_idxs = idxs[:, -self.config.block_size :]

            logits, _ = self(cropped_idxs)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idxs_next = torch.multinomial(probs, num_samples=1)
            idxs = torch.cat((idxs, idxs_next), dim=1)

        return idxs
