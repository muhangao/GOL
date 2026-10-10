"""A small, non-recurrent decoder: RMSNorm, RoPE, causal SDPA and SwiGLU."""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig

IO_PREFIXES = ("embedding.", "lm_head.")


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        return (x.float() * scale).to(x.dtype) * self.weight


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.dim // cfg.heads
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.attn_out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [y.view(b, t, self.heads, self.head_dim).transpose(1, 2) for y in (q, k, v)]

        def rotate(y):
            even, odd = y[..., ::2], y[..., 1::2]
            c, s = cos.to(y.dtype), sin.to(y.dtype)
            return torch.stack((even * c - odd * s, even * s + odd * c), -1).flatten(-2)

        y = F.scaled_dot_product_attention(rotate(q), rotate(k), v, dropout_p=0.0, is_causal=True)
        x = x + self.attn_out(y.transpose(1, 2).contiguous().view(b, t, d))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, vocab_size, bias=False)
        if cfg.tied_embeddings:
            self.lm_head.weight = self.embedding.weight
        head_dim = cfg.dim // cfg.heads
        inv = cfg.rope_theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        angle = torch.outer(torch.arange(cfg.seq_len).float(), inv)
        self.register_buffer("rope_cos", angle.cos()[None, None], persistent=False)
        self.register_buffer("rope_sin", angle.sin()[None, None], persistent=False)

    def forward(self, tokens, targets=None):
        if tokens.shape[1] > self.cfg.seq_len:
            raise ValueError("Input exceeds configured sequence length")
        x = self.embedding(tokens)
        cos = self.rope_cos[:, :, :tokens.shape[1]].to(x.dtype)
        sin = self.rope_sin[:, :, :tokens.shape[1]].to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())


def build_model(cfg: ModelConfig, vocab: int, body_seed: int, io_seed: int) -> Decoder:
    """Initialize body and I/O independently, so vocabulary size cannot shift the body RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(body_seed)
        model = Decoder(cfg, vocab)
        torch.manual_seed(body_seed)
        for name, p in model.named_parameters():
            if not name.startswith(IO_PREFIXES):
                if p.ndim == 1:
                    nn.init.ones_(p)
                else:
                    std = 0.02 / math.sqrt(2 * cfg.layers) if name.endswith(("attn_out.weight", "down.weight")) else 0.02
                    nn.init.normal_(p, std=std)
        torch.manual_seed(io_seed)
        nn.init.normal_(model.embedding.weight, std=0.02)
        if not cfg.tied_embeddings:
            nn.init.normal_(model.lm_head.weight, std=0.02)
    return model


def transfer_body(source: Decoder, vocab: int, body_seed: int, io_seed: int) -> Decoder:
    target = build_model(source.cfg, vocab, body_seed, io_seed)
    state = {k: v for k, v in source.state_dict().items() if not k.startswith(IO_PREFIXES)}
    result = target.load_state_dict(state, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != {"embedding.weight", "lm_head.weight"}:
        raise RuntimeError(f"Unexpected transfer keys: {result}")
    return target


def fingerprint(model: Decoder, io: bool = False) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith(IO_PREFIXES) == io:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
