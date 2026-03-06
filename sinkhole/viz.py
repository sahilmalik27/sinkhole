from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sinkhole.models import AnalysisReport


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _fig_to_file(fig: plt.Figure, path: Path):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_attention_heatmap(report: AnalysisReport, attn: np.ndarray) -> str:
    """Create [layer x head] grid showing max attention to sink tokens.

    Returns base64-encoded PNG.
    """
    n_layers, n_heads, seq_len, _ = attn.shape

    sink_positions = [t.position for t in report.sinks.tokens]
    if not sink_positions:
        sink_positions = [0]

    # For each (layer, head), compute max attention directed at any sink token
    grid = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            # Mean attention received by sink tokens from all source tokens
            sink_attn = attn[l, h, :, sink_positions].sum(axis=1).mean()
            grid[l, h] = sink_attn

    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.3), max(6, n_layers * 0.2)))
    im = ax.imshow(grid, aspect="auto", cmap="Reds", interpolation="nearest")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Attention to Sink Tokens (per layer/head)")
    fig.colorbar(im, ax=ax, label="Mean attention mass")
    return _fig_to_base64(fig)


def plot_spike_norms(report: AnalysisReport) -> str:
    """Per-token norm across layers (line chart).

    Returns base64-encoded PNG.
    """
    norms = report.spikes.norms_per_layer  # [n_layers, seq_len]
    n_layers, seq_len = norms.shape

    fig, ax = plt.subplots(figsize=(max(10, seq_len * 0.3), 5))

    # Plot a few representative layers
    layer_indices = np.linspace(0, n_layers - 1, min(8, n_layers), dtype=int)
    for l in layer_indices:
        ax.plot(range(seq_len), norms[l], alpha=0.6, label=f"Layer {l}")

    # Mark spike tokens
    for spike in report.spikes.tokens:
        ax.axvline(spike.position, color="red", linestyle="--", alpha=0.5)

    ax.set_xlabel("Token position")
    ax.set_ylabel("Hidden state L2 norm")
    ax.set_title("Activation Norms per Token")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(report.token_texts, rotation=90, fontsize=7)
    ax.legend(fontsize=7, loc="upper right")
    return _fig_to_base64(fig)


def plot_sink_mass(report: AnalysisReport) -> str:
    """Per-token attention mass received (bar chart).

    Returns base64-encoded PNG.
    """
    mass = report.sinks.attn_mass_per_token  # [seq_len]
    seq_len = len(mass)

    fig, ax = plt.subplots(figsize=(max(10, seq_len * 0.3), 5))

    colors = []
    sink_positions = {t.position for t in report.sinks.tokens}
    for i in range(seq_len):
        colors.append("#d32f2f" if i in sink_positions else "#1976d2")

    ax.bar(range(seq_len), mass, color=colors)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Mean attention received")
    ax.set_title("Attention Mass per Token (red = sink token)")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(report.token_texts, rotation=90, fontsize=7)
    return _fig_to_base64(fig)
