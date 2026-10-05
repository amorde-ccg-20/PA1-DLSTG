from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("moving-average kernel must be odd and at least 3")
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = (self.kernel_size - 1) // 2
        trend = F.avg_pool1d(F.pad(x.transpose(1, 2), (pad, pad), mode="replicate"), self.kernel_size, stride=1).transpose(1, 2)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: float, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factor = factor
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        correlation = torch.fft.irfft(
            torch.fft.rfft(q.transpose(-1, -2), dim=-1) * torch.conj(torch.fft.rfft(k.transpose(-1, -2), dim=-1)),
            n=length, dim=-1,
        ).mean(dim=-2)
        top_k = min(length, max(1, int(self.factor * math.log(length))))
        scores, delays = correlation.topk(top_k, dim=-1)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        positions = torch.arange(length, device=x.device).view(1, 1, 1, length)
        gather_index = (positions - delays.unsqueeze(-1)) % length
        shifted = torch.gather(v.unsqueeze(2).expand(-1, -1, top_k, -1, -1), 3,
                               gather_index.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim))
        mixed = (weights.unsqueeze(-1).unsqueeze(-1) * shifted).sum(dim=2)
        return self.out(mixed.transpose(1, 2).reshape(batch, length, width))


class AutoformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: float, moving_avg: int, dropout: float):
        super().__init__()
        self.auto_correlation = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.decomposition_1 = SeriesDecomposition(moving_avg)
        self.decomposition_2 = SeriesDecomposition(moving_avg)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.decomposition_1(x + self.dropout(self.auto_correlation(self.norm_1(x))))
        x, _ = self.decomposition_2(x + self.dropout(self.ffn(self.norm_2(x))))
        return x


class AutoformerForecaster(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, input_dim: int, external_dim: int, d_model: int,
                 n_heads: int, e_layers: int, moving_avg: int, factor: float, dropout: float):
        super().__init__()
        if pred_len != 168:
            raise ValueError("Task 2 requires pred_len=168")
        self.pred_len = pred_len
        self.external_dim = external_dim
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList([AutoformerBlock(d_model, n_heads, factor, moving_avg, dropout) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.seasonal_head = nn.Linear(seq_len * d_model, pred_len)
        self.trend_head = nn.Linear(seq_len, pred_len)
        self.future_head = nn.Linear(external_dim, 1) if external_dim else None
        self.input_decomposition = SeriesDecomposition(moving_avg)

    def forward(self, x: torch.Tensor, future_external: torch.Tensor | None = None) -> torch.Tensor:
        seasonal, trend = self.input_decomposition(x[..., :1])
        encoded_input = torch.cat([seasonal, x[..., 1:]], dim=-1)
        encoded = self.input_projection(encoded_input) + self.position
        for block in self.blocks:
            encoded = block(encoded)
        output = self.seasonal_head(self.norm(encoded).flatten(1)) + self.trend_head(trend.squeeze(-1))
        if self.future_head is not None:
            if future_external is None or future_external.shape[1:] != (self.pred_len, self.external_dim):
                raise ValueError("future external features are required with the configured shape")
            output = output + self.future_head(future_external).squeeze(-1)
        return output
