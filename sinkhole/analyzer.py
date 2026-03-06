from __future__ import annotations

import numpy as np

from sinkhole.models import (
    AnalysisReport,
    SinkResult,
    SinkToken,
    SpikeResult,
    SpikeToken,
)


def find_spikes(
    hidden: np.ndarray,
    token_texts: list[str],
    threshold: float = 10.0,
) -> SpikeResult:
    """Detect tokens with massive activation outliers.

    Args:
        hidden: [n_layers, seq_len, d_model]
        token_texts: decoded token strings
        threshold: spike score cutoff (multiple of mean norm)
    """
    # Per-token L2 norm across channels, per layer
    norms = np.linalg.norm(hidden, axis=2)  # [n_layers, seq_len]

    # Aggregate across layers: max norm per token
    max_norms = norms.max(axis=0)  # [seq_len]
    median_norm = np.median(max_norms)

    if median_norm == 0:
        return SpikeResult(tokens=[], norms_per_layer=norms, threshold=threshold)

    scores = max_norms / median_norm  # [seq_len]

    spike_tokens = []
    for pos in np.where(scores > threshold)[0]:
        pos = int(pos)
        # Find spike channels: channels where |h| > threshold * std across all layers
        all_h = hidden[:, pos, :]  # [n_layers, d_model]
        h_abs = np.abs(all_h)
        channel_max = h_abs.max(axis=0)  # [d_model]
        channel_std = channel_max.std()
        channel_mean = channel_max.mean()
        if channel_std > 0:
            spike_channels = np.where(channel_max > channel_mean + threshold * channel_std)[0].tolist()
        else:
            spike_channels = []

        spike_tokens.append(SpikeToken(
            position=pos,
            text=token_texts[pos],
            score=float(scores[pos]),
            channels=spike_channels,
        ))

    spike_tokens.sort(key=lambda t: t.score, reverse=True)
    return SpikeResult(tokens=spike_tokens, norms_per_layer=norms, threshold=threshold)


def find_sinks(
    attn: np.ndarray,
    token_texts: list[str],
    top_k: float = 0.5,
) -> SinkResult:
    """Detect tokens that attract disproportionate attention mass.

    Args:
        attn: [n_layers, n_heads, seq_len, seq_len]
        token_texts: decoded token strings
        top_k: fraction of max score to qualify as sink
    """
    n_layers, n_heads, seq_len, _ = attn.shape

    # Per-token: mean attention received across all layers and heads
    # attn[l, h, i, j] = attention from token i to token j
    # Sum over source dimension (axis=2) to get attention received by each token
    attn_received = attn.sum(axis=2)  # [n_layers, n_heads, seq_len]
    # Normalize: divide by seq_len to get mean attention received
    attn_received = attn_received / seq_len
    # Average across layers and heads
    attn_mass_per_token = attn_received.mean(axis=(0, 1))  # [seq_len]

    max_mass = attn_mass_per_token.max()
    if max_mass == 0:
        return SinkResult(tokens=[], sink_heads=[], attn_mass_per_token=attn_mass_per_token)

    # Find sink heads: heads where top-2 tokens receive > 50% of attention mass
    sink_heads = []
    for l in range(n_layers):
        for h in range(n_heads):
            head_attn_received = attn[l, h].sum(axis=0)  # [seq_len] — total attn received
            total = head_attn_received.sum()
            if total > 0:
                top2 = np.sort(head_attn_received)[-2:].sum()
                if top2 / total > 0.5:
                    sink_heads.append((l, h))

    # Find sink tokens
    sink_tokens = []
    for pos in np.where(attn_mass_per_token > top_k * max_mass)[0]:
        pos = int(pos)
        # Count how many heads this token dominates
        head_count = 0
        for l in range(n_layers):
            for h in range(n_heads):
                head_received = attn[l, h, :, pos].sum()
                head_total = attn[l, h].sum(axis=1).max()  # max attention any token receives
                if head_total > 0 and head_received / (attn[l, h].sum(axis=1).sum()) > 1.0 / seq_len * 2:
                    head_count += 1

        sink_tokens.append(SinkToken(
            position=pos,
            text=token_texts[pos],
            attn_mass=float(attn_mass_per_token[pos]),
            head_count=head_count,
            total_heads=n_layers * n_heads,
        ))

    sink_tokens.sort(key=lambda t: t.attn_mass, reverse=True)
    return SinkResult(tokens=sink_tokens, sink_heads=sink_heads, attn_mass_per_token=attn_mass_per_token)


def analyze(
    hidden: np.ndarray,
    attn: np.ndarray,
    token_texts: list[str],
    model_name: str,
    prompt: str,
    threshold: float = 10.0,
) -> AnalysisReport:
    """Run full spike + sink analysis."""
    n_layers, n_heads, seq_len, _ = attn.shape

    spikes = find_spikes(hidden, token_texts, threshold=threshold)
    sinks = find_sinks(attn, token_texts)

    spike_positions = {t.position for t in spikes.tokens}
    sink_positions = {t.position for t in sinks.tokens}

    union = spike_positions | sink_positions
    intersection = spike_positions & sink_positions
    overlap_jaccard = len(intersection) / len(union) if union else 0.0
    overlap_tokens = [token_texts[p] for p in sorted(intersection)]

    # KV waste: fraction of sequence that are sink tokens
    kv_waste_fraction = len(sinks.tokens) / seq_len if seq_len > 0 else 0.0

    # KV waste attention mass: total attention mass absorbed by sink tokens
    sink_mass_total = sum(t.attn_mass for t in sinks.tokens)
    total_mass = float(sinks.attn_mass_per_token.sum())
    kv_waste_attn_mass = sink_mass_total / total_mass if total_mass > 0 else 0.0

    return AnalysisReport(
        model_name=model_name,
        prompt=prompt,
        token_texts=token_texts,
        spikes=spikes,
        sinks=sinks,
        overlap_tokens=overlap_tokens,
        overlap_jaccard=overlap_jaccard,
        kv_waste_fraction=kv_waste_fraction,
        kv_waste_attn_mass=kv_waste_attn_mass,
        n_layers=n_layers,
        n_heads=n_heads,
        seq_len=seq_len,
    )
