"""Crossformer (Zhang & Yan, ICLR'23) — paper-faithful univariate port.

Reference
---------
Y. Zhang, J. Yan, "Crossformer: Transformer Utilizing Cross-Dimension
Dependency for Multivariate Time Series Forecasting", ICLR 2023.
Official code (cached in ``papers/literlature/crossformer_official/``,
github.com/Thinklab-SJTU/Crossformer):
    cross_former.py     — top-level
    cross_embed.py      — DSW_embedding
    cross_encoder.py    — Encoder, scale_block, SegMerging
    cross_decoder.py    — Decoder, DecoderLayer
    attn.py             — FullAttention, AttentionLayer, TwoStageAttentionLayer

This module mirrors the official structure with the following deliberate
adaptations:

- **univariate** — UMass apt-level kW is single-channel. We fix the
  ``data_dim`` (a.k.a. ``ts_d``) axis to 1, but we keep the 4D tensor
  shape ``[B, ts_d=1, seg_num, d_model]`` everywhere so the official
  einops paths translate verbatim. The cross-dimension stage of TSA
  becomes trivial (1 router suffices), but we keep the code path so
  the architecture remains paper-faithful and ready for v05+ multivariate.
- **no PyTorch-Lightning / extra plumbing** — plain ``nn.Module`` only.
- input/output **2-D**: forward takes ``[B, L]`` and returns ``[B, H]``.
  Internally we unsqueeze to add the singleton ts_d=1.
- ``baseline=False`` (mean-subtraction not used): the v01-v03 protocol
  already applies per-apt z-norm, so adding another mean-subtract here
  would double-correct.

Algorithmic invariants kept verbatim from the official source:

1. **DSW embedding** with learnable position embedding
   ``[1, ts_d, seg_num, d_model]``.
2. **Hierarchical encoder** with ``e_blocks`` ``scale_block``s; the first
   has ``win_size=1`` (no merging), subsequent ones have ``win_size=2``
   (paper default; the official top-level passes ``win_size=4`` but the
   inner scale_block doc states "we set win_size=2 in our paper").
3. **TwoStageAttentionLayer (TSA)** — cross-time then cross-dimension
   via a learnable router (factor=10).
4. **Hierarchical decoder** — ``d_layers = e_layers + 1`` decoder layers,
   each cross-attending to a different scale of the encoder output.
   Per-layer ``layer_predict``s are summed for the final forecast.
5. ``pad_in_len = ceil(L / seg_len) * seg_len`` with first-segment
   replication padding when ``L % seg_len != 0``.
"""

from __future__ import annotations

from math import ceil, sqrt

import torch
import torch.nn as nn
from einops import rearrange, repeat

from config import HORIZON, INPUT_SIZE


# ---------------------------------------------------------------------------
# Attention (verbatim port of attn.py, module-private helpers)
# ---------------------------------------------------------------------------


class _FullAttention(nn.Module):
    """Standard scaled-dot-product attention (single, multi-head outside)."""

    def __init__(self, scale: float | None = None, attention_dropout: float = 0.1) -> None:
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        v = torch.einsum("bhls,bshd->blhd", attn, values)
        return v.contiguous()


class _AttentionLayer(nn.Module):
    """Multi-head self-attention with linear projections (Q, K, V, output)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_keys: int | None = None,
        d_values: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = _FullAttention(scale=None, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        q = self.query_projection(queries).view(B, L, H, -1)
        k = self.key_projection(keys).view(B, S, H, -1)
        v = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(q, k, v).view(B, L, -1)
        return self.out_projection(out)


class _TwoStageAttentionLayer(nn.Module):
    """Cross-time + cross-dimension attention (TSA).

    Input/output shape: ``[B, ts_d, seg_num, d_model]``.

    Stage 1 (cross-time): rearrange to ``(b ts_d) seg_num d_model``,
    self-attend across segments, FFN.

    Stage 2 (cross-dimension): rearrange to ``(b seg_num) ts_d d_model``,
    use a learnable ``router [seg_num, factor, d_model]`` to first
    summarise variates into ``factor`` slots, then redistribute.
    """

    def __init__(
        self,
        seg_num: int,
        factor: int,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = _AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_sender = _AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = _AttentionLayer(d_model, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, ts_d, seg_num, d_model]
        batch = x.shape[0]
        # Stage 1: cross-time MSA per dimension.
        time_in = rearrange(x, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model")
        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = self.norm1(time_in + self.dropout(time_enc))
        dim_in = self.norm2(dim_in + self.dropout(self.MLP1(dim_in)))
        # Stage 2: cross-dimension via router.
        dim_send = rearrange(
            dim_in, "(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model", b=batch
        )
        batch_router = repeat(
            self.router, "seg_num factor d_model -> (repeat seg_num) factor d_model", repeat=batch
        )
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = self.norm3(dim_send + self.dropout(dim_receive))
        dim_enc = self.norm4(dim_enc + self.dropout(self.MLP2(dim_enc)))
        return rearrange(
            dim_enc, "(b seg_num) ts_d d_model -> b ts_d seg_num d_model", b=batch
        )


# ---------------------------------------------------------------------------
# Embedding (verbatim port of cross_embed.py)
# ---------------------------------------------------------------------------


class _DSWEmbedding(nn.Module):
    """Dimension-Segment-Wise embedding: [B, L, ts_d] → [B, ts_d, seg_num, d_model]."""

    def __init__(self, seg_len: int, d_model: int) -> None:
        super().__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, ts_d]
        batch, ts_len, ts_dim = x.shape
        x_segment = rearrange(
            x, "b (seg_num seg_len) d -> (b d seg_num) seg_len", seg_len=self.seg_len
        )
        x_embed = self.linear(x_segment)
        return rearrange(
            x_embed,
            "(b d seg_num) d_model -> b d seg_num d_model",
            b=batch,
            d=ts_dim,
        )


# ---------------------------------------------------------------------------
# Encoder (verbatim port of cross_encoder.py)
# ---------------------------------------------------------------------------


class _SegMerging(nn.Module):
    """Merge ``win_size`` adjacent segments into one (coarser scale)."""

    def __init__(self, d_model: int, win_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = nn.LayerNorm(win_size * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, ts_d, seg_num, d_model]
        _, _, seg_num, _ = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)
        seg_to_merge = [x[:, :, i :: self.win_size, :] for i in range(self.win_size)]
        x = torch.cat(seg_to_merge, dim=-1)            # [B, ts_d, seg_num/win, win*d_model]
        x = self.norm(x)
        return self.linear_trans(x)


class _ScaleBlock(nn.Module):
    """One encoder scale: optional SegMerging then ``depth`` TSA layers."""

    def __init__(
        self,
        win_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        depth: int,
        dropout: float,
        seg_num: int,
        factor: int,
    ) -> None:
        super().__init__()
        self.merge_layer: _SegMerging | None = (
            _SegMerging(d_model, win_size) if win_size > 1 else None
        )
        self.encode_layers = nn.ModuleList(
            [
                _TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, d_ff, dropout)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merge_layer is not None:
            x = self.merge_layer(x)
        for layer in self.encode_layers:
            x = layer(x)
        return x


class _Encoder(nn.Module):
    """Hierarchical encoder: ``e_blocks`` scales, returns list of all scales."""

    def __init__(
        self,
        e_blocks: int,
        win_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        block_depth: int,
        dropout: float,
        in_seg_num: int,
        factor: int,
    ) -> None:
        super().__init__()
        # First scale: no merging (win_size=1, full segment count).
        self.encode_blocks = nn.ModuleList(
            [
                _ScaleBlock(
                    win_size=1,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    depth=block_depth,
                    dropout=dropout,
                    seg_num=in_seg_num,
                    factor=factor,
                )
            ]
        )
        # Subsequent scales: merge by win_size, segment count divides accordingly.
        for i in range(1, e_blocks):
            self.encode_blocks.append(
                _ScaleBlock(
                    win_size=win_size,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    depth=block_depth,
                    dropout=dropout,
                    seg_num=ceil(in_seg_num / (win_size ** i)),
                    factor=factor,
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Returns [x_scale0, x_scale1, ...] including the pre-merge embedding.
        encode_x = [x]
        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)
        return encode_x


# ---------------------------------------------------------------------------
# Decoder (verbatim port of cross_decoder.py)
# ---------------------------------------------------------------------------


class _DecoderLayer(nn.Module):
    """One decoder layer: TSA self-attention + cross-attention + per-layer linear_pred."""

    def __init__(
        self,
        seg_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        out_seg_num: int = 10,
        factor: int = 10,
    ) -> None:
        super().__init__()
        self.self_attention = _TwoStageAttentionLayer(
            out_seg_num, factor, d_model, n_heads, d_ff, dropout
        )
        self.cross_attention = _AttentionLayer(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, ts_d, out_seg_num, d_model],  cross: [B, ts_d, in_seg_num, d_model]
        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, "b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model")
        cross = rearrange(cross, "b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model")
        tmp = self.cross_attention(x, cross, cross)
        x = x + self.dropout(tmp)
        y = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)
        dec_output = rearrange(
            dec_output, "(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model", b=batch
        )
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(
            layer_predict, "b out_d seg_num seg_len -> b (out_d seg_num) seg_len"
        )
        return dec_output, layer_predict


class _Decoder(nn.Module):
    """Hierarchical decoder: per-layer cross-attends to encoder scale i, sums predictions."""

    def __init__(
        self,
        seg_len: int,
        d_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        out_seg_num: int,
        factor: int,
    ) -> None:
        super().__init__()
        self.decode_layers = nn.ModuleList(
            [
                _DecoderLayer(
                    seg_len=seg_len,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    out_seg_num=out_seg_num,
                    factor=factor,
                )
                for _ in range(d_layers)
            ]
        )

    def forward(self, x: torch.Tensor, cross: list[torch.Tensor]) -> torch.Tensor:
        ts_d = x.shape[1]
        final_predict = None
        for i, layer in enumerate(self.decode_layers):
            x, layer_predict = layer(x, cross[i])
            final_predict = layer_predict if final_predict is None else final_predict + layer_predict
        return rearrange(
            final_predict, "b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d", out_d=ts_d
        )


# ---------------------------------------------------------------------------
# Top-level Crossformer (univariate adaptation of cross_former.py)
# ---------------------------------------------------------------------------


class Crossformer(nn.Module):
    """Univariate Crossformer (paper-faithful port).

    Defaults match the official top-level (cross_former.py) trimmed for our
    L=96 / H=24 setting. The official top-level uses ``win_size=4`` while
    ``scale_block`` doc says "we set win_size=2 in our paper"; we keep
    ``win_size=2`` for paper consistency.

    Hyperparameters (defaults)
    --------------------------
    seg_len    = 12     # 96/12 = 8 input segments, 24/12 = 2 output segments
    d_model    = 256    # paper default for short horizon
    d_ff       = 512
    n_heads    = 4
    e_layers   = 3      # encoder scale_blocks
    d_layers   = 4      # = e_layers + 1
    win_size   = 2
    factor     = 10     # router slots in TSA cross-dim stage
    dropout    = 0.0    # match v01-v03 (no dropout in NBEATSx training)
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        horizon: int = HORIZON,
        data_dim: int = 1,
        seg_len: int = 12,
        win_size: int = 2,
        factor: int = 10,
        d_model: int = 256,
        d_ff: int = 512,
        n_heads: int = 4,
        e_layers: int = 3,
        dropout: float = 0.0,
        block_depth: int = 1,
    ) -> None:
        super().__init__()
        if data_dim != 1:
            raise NotImplementedError(
                "v04 Crossformer is univariate (data_dim=1); multivariate is v05+."
            )
        self.input_size = input_size
        self.horizon = horizon
        self.data_dim = data_dim
        self.seg_len = seg_len

        # Pad input/horizon up to a multiple of seg_len.
        self.pad_in_len = ceil(input_size / seg_len) * seg_len
        self.pad_out_len = ceil(horizon / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - input_size

        # Encoder.
        self.enc_value_embedding = _DSWEmbedding(seg_len=seg_len, d_model=d_model)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, data_dim, self.pad_in_len // seg_len, d_model)
        )
        self.pre_norm = nn.LayerNorm(d_model)
        self.encoder = _Encoder(
            e_blocks=e_layers,
            win_size=win_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            block_depth=block_depth,
            dropout=dropout,
            in_seg_num=self.pad_in_len // seg_len,
            factor=factor,
        )

        # Decoder.
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, data_dim, self.pad_out_len // seg_len, d_model)
        )
        self.decoder = _Decoder(
            seg_len=seg_len,
            d_layers=e_layers + 1,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            out_seg_num=self.pad_out_len // seg_len,
            factor=factor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, L] (univariate) and add the singleton ts_d=1 axis to
        # match the official 3-D input expectation [B, L, ts_d].
        if x.dim() == 2:
            x_seq = x.unsqueeze(-1)
        elif x.dim() == 3:
            x_seq = x
        else:
            raise ValueError(f"Crossformer expects [B, L] or [B, L, 1], got {tuple(x.shape)}")

        batch_size = x_seq.shape[0]
        if self.in_len_add != 0:
            # Replicate the first time step in_len_add times to pad up to pad_in_len.
            x_seq = torch.cat(
                (x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1
            )

        x_seq = self.enc_value_embedding(x_seq)              # [B, ts_d, seg_num, d_model]
        x_seq = x_seq + self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)                        # list[len = e_layers + 1]
        dec_in = repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=batch_size,
        )
        predict_y = self.decoder(dec_in, enc_out)            # [B, pad_out_len, ts_d]

        # Truncate to the true horizon and squeeze the univariate ts_d axis.
        predict_y = predict_y[:, : self.horizon, :]
        return predict_y.squeeze(-1)                         # [B, H]
