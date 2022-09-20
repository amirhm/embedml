from embedml.nn import Module
from embedml.nn import Linear
from embedml.nn import Embedding
from embedml.nn import Softmax
from embedml.nn import LayerNorm
from embedml.nn import ModuleDict
from collections import namedtuple
from embedml.tensor import Tensor
import numpy as np


class SelfAttention(Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn0 = Linear(config.n_embd, config.n_embd)
        self.c_attn1 = Linear(config.n_embd, config.n_embd)
        self.c_attn2 = Linear(config.n_embd, config.n_embd)
        self.sm = Softmax(dim=-1)
        # output projection
        self.c_proj = Linear(config.n_embd, config.n_embd)
        #  causal mask to ensure that attention is only applied to the left in the input sequence
        #  self.bias = Tensor(torch.tril(torch.ones(config.block_size, config.block_size))
        #                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attn_pdrop = config.attn_pdrop
        self.resid_pdrop = config.resid_pdrop

    def forward(self, x):

        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.c_attn0(x).reshape((B, T, self.n_head, C // self.n_head)).transpose(1, 2)  # (B, nh, T, hs)
        k = self.c_attn1(x).reshape((B, T, self.n_head, C // self.n_head)).transpose(1, 2)  # (B, nh, T, hs)
        v = self.c_attn2(x).reshape((B, T, self.n_head, C // self.n_head)).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = q.matmul(k.transpose(-2, -1)) * (1.0 / np.sqrt(k.shape[-1]))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = self.sm(att)
        att = att.dropout(self.attn_pdrop)
        y = att.matmul(v)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).reshape((B, T, C))  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y).dropout(self.resid_pdrop)
        return y


class Block(Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd)
        self.config = config

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + (self.c_proj(self.c_fc(self.ln_2(x)).gelu())).dropout(self.config.resid_pdrop)
        return x


class GPT(Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        cfg = config()
        gpt_nano = dict(n_layer=3, n_head=3, n_embd=48)
        cfg = cfg._replace(**gpt_nano)
        self.block_size = cfg.block_size

        self.transformer = ModuleDict(
            wte=Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=Embedding(cfg.block_size, cfg.n_embd),
            h=[Block(cfg) for _ in range(cfg.n_layer)],
            ln_f=LayerNorm(cfg.n_embd),
        )
        self.lm_head = Linear(cfg.n_embd, cfg.vocab_size)
        self.cfg = cfg

    def forward(self, idx, targets=None):

        b, t = idx.shape
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = np.arange(0, t, dtype=int).reshape((1, -1))  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = (tok_emb + pos_emb).dropout(self.cfg.embd_pdrop)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


def main():
    config = namedtuple(
        'config',
        ['n_layer', 'n_embd', 'n_head', 'resid_pdrop', 'attn_pdrop', 'vocab_size', 'block_size', 'embd_pdrop'],
        defaults=[1, 48, 3, 0.1, 0.1, 10, 19, 0.1]
    )
    gpt = GPT(config)
    bs, T = 5, 4
    inp = Tensor(np.ones((bs, T), dtype=int))
    logits = gpt(inp)
    assert logits.shape == (bs, T, gpt.cfg.vocab_size)
    logits.backward()
    assert gpt.transformer["wte"].weight.shape == (gpt.cfg.vocab_size, gpt.cfg.n_embd)
    assert gpt.transformer["wte"].weight.grad.shape == (gpt.cfg.vocab_size, gpt.cfg.n_embd)

    steps = list(map(lambda x: (x.ctx, x.ctx.parents[0].shape, x.ctx.parents[1].shape if len(x.ctx.parents) > 1 else None), logits.get_topo_graph()))
    print(*steps, sep="\n")


if __name__ == "__main__":
    exit(main())
