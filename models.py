import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
    


class SelfAttention(nn.Module):
    def __init__(
        self, n_embed: int, n_head: int, 
        dropout: float, is_causal: bool, bias=True,
    ) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        assert n_embed % n_head == 0 
        self.D = n_embed // n_head
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=bias)
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)
        self.dropout = dropout
        self.res_dropout = nn.Dropout(dropout)
        self.is_causal = is_causal
        self.attn_scale = 1.0 / math.sqrt(self.D)
    

    def forward(self, x: Tensor) -> Tensor:
        B, T, n_embed = x.shape
        assert n_embed == self.n_embed

        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        q = q.view(B, T, self.n_head, self.D).transpose(1, 2) # gives shape (B, N, T, D)
        k = k.view(B, T, self.n_head, self.D).transpose(1, 2) # gives shape (B, N, T, D)
        v = v.view(B, T, self.n_head, self.D).transpose(1, 2) # gives shape (B, N, T, D)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, 
                                           is_causal=self.is_causal, scale=self.attn_scale)
        
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embed) # concat head outputs 
        y = self.c_proj(y)
        y = self.res_dropout(y)
        return y



class MLP(nn.Module):
    def __init__(self, n_embed: int, dropout: float, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed, bias=bias)
        self.c_proj = nn.Linear(4 * n_embed, n_embed, bias=bias)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x



class Block(nn.Module):
    def __init__(
        self, n_embed: int, n_head: int, 
        dropout: float, is_causal: bool,  bias=True,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed, bias=bias)
        self.attn = SelfAttention(n_embed, n_head, dropout, is_causal, bias)
        self.ln_2 = nn.LayerNorm(n_embed, bias=bias)
        self.mlp = MLP(n_embed, dropout, bias)


    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class SlowTransformer(nn.Module):
    def __init__(
        self, n_embed: int, n_head: int, block_size: int,
        n_layer: int, dropout: float, is_causal: bool, bias=True,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        args = (n_embed, n_head, dropout, is_causal, bias)
        self.h = nn.ModuleList([Block(*args) for _ in range(n_layer)])
        self.apply(self.init_weights)
    

    @staticmethod
    def init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 
                

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] <= self.block_size, \
            f"cannot forward sequence of length {x.shape[1]}, block size is only {self.block_size}"

        for blk in self.h:
            x = blk(x)

        return x[:, -1, :]  
    


class SlowLanguageTransformer(nn.Module):
    def __init__(
        self, n_embed: int, n_head: int, block_size: int, n_layer: int,
        vocab_size: int, dropout: float, is_causal: bool, bias=True,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embed)
        self.wpe = nn.Embedding(block_size, n_embed)
        self.drop = nn.Dropout(dropout)
        self.transformer = SlowTransformer(n_embed, n_head, block_size, 
                                           n_layer, dropout, is_causal, bias)
        
        self.ln_f = nn.LayerNorm(n_embed, bias=bias)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self.transformer.init_weights)


    def forward(self, x_idx: Tensor) -> Tensor:
        device = x_idx.device
        _, T = x_idx.shape

        assert T <= self.block_size, \
            f"cannot forward sequence of length {T}, block size is only {self.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.int64, device=device)

        tok_emb = self.wte(x_idx) # shape (B, T, C)
        pos_emb = self.wpe(pos) # shape (T, C)

        # (B, T, C) + (T, C) = (B, T, C)
        # elementwise addition for each batch
        x = self.drop(tok_emb + pos_emb)
        x = self.transformer(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
    

    @torch.no_grad()
    def generate(self, x_idx: Tensor, max_new_tokens: int, temperature=1.0) -> Tensor:
        # Take a conditioning sequence of indices x_idx (int64 tensor of shape (B, T)) and 
        # complete the sequence max_new_tokens times, feeding the predictions back into 
        # the model each time. Most likely you"ll want to make sure to be in model.eval() 
        # mode of operation for this.
        for _ in range(max_new_tokens):
            if x_idx.shape[1] <= self.block_size:
                x_idx_cropped = x_idx 
            else:
                x_idx_cropped = x_idx[:, -self.block_size:]

            logits = self(x_idx_cropped) / temperature

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x_idx = torch.cat((x_idx, idx_next), dim=1)

        return x_idx