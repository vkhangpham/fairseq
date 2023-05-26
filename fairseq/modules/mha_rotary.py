import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos[...,:q.shape[-2],:], sin[...,:q.shape[-2],:]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class MHA_rotary(nn.Module):
    def __init__(self, n_attn, num_heads, n_embd, ctx_len, time_shift = False):
        super().__init__()
        assert n_attn % num_heads == 0
        self.num_heads = num_heads
        self.ctx_len = ctx_len
        self.head_dim = n_attn // num_heads

        if time_shift:
            self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.query = nn.Linear(n_embd, n_attn)
        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)

        self.register_buffer("mask", torch.tril(torch.ones(ctx_len, ctx_len)))
        
        self.rotary_ndims = int(self.head_dim * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.output = nn.Linear(n_attn, n_embd)

    def forward(self, x):
        B, T, C = x.size()

        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)

        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)         # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                                     # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))                 # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))                     # causal mask
        att = F.softmax(att, dim = -1)                                                  # softmax

        x = att @ v                                                                     # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)                               # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x)
        return x, None
    
class MHA_pro(nn.Module):
    def __init__(self, n_attn, num_heads, n_embd, ctx_len):
        super().__init__()
        assert n_attn % num_heads == 0
        self.num_heads = num_heads
        self.ctx_len = ctx_len
        self.head_dim = n_attn // num_heads

        self.time_w = nn.Parameter(torch.ones(self.num_heads, ctx_len))
        self.time_alpha = nn.Parameter(torch.ones(self.num_heads, 1, ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.num_heads, ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(ctx_len, 1))
        self.register_buffer("mask", torch.tril(torch.ones(ctx_len, ctx_len)))

        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.query = nn.Linear(n_embd, n_attn)
        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)
        
        self.rotary_ndims = int(self.head_dim * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.head_mix = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, bias=False)  # talking heads

        self.output = nn.Linear(n_attn, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:] # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)      # time-shift mixing
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)         # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                                     # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)  
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))                 # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))                     # causal mask
        att = F.softmax(att, dim = -1)                                                  # softmax
        att = att * w                                                                   # time-weighting
        att = self.head_mix(att)                                                        # talking heads

        x = att @ v                                                                     # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)                               # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x) * self.time_gamma[:T, :]
        return x, None