"""Generate plots from raw evaluation results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RAW_RESULTS_PATH = RESULTS_DIR / "raw_results.jsonl"


def load_results() -> list[dict]:
    rows = []
    with open(RAW_RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_histogram(values, title, xlabel, filename, color="steelblue", bins=50):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.axvline(np.mean(values), color="red", linestyle="--", label=f"Mean: {np.mean(values):.3f}")
    ax.axvline(np.median(values), color="orange", linestyle="--", label=f"Median: {np.median(values):.3f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_boxplot_by_category(rows, filename):
    categories = {}
    for r in rows:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["sink_count"])

    cats = sorted(categories.keys())
    data = [categories[c] for c in cats]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=cats, patch_artist=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for patch, color in zip(bp["boxes"], colors * 3):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Sink Token Count by Prompt Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Sink Count")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_spike_position_heatmap(rows, filename):
    n_bins = 50
    counts = np.zeros(n_bins)
    for r in rows:
        for rp in r.get("spike_rel_positions", []):
            bin_idx = min(int(rp * n_bins), n_bins - 1)
            counts[bin_idx] += 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.linspace(0, 1, n_bins), counts, width=1.0 / n_bins, color="steelblue", edgecolor="white")
    ax.set_title("Spike Token Relative Position Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Relative Position (0 = start, 1 = end)")
    ax.set_ylabel("Spike Count")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_scatter_with_regression(rows, filename):
    seq_lens = [r["seq_len"] for r in rows]
    sink_mass = [r["sink_attn_mass_total"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(seq_lens, sink_mass, alpha=0.3, s=10, color="steelblue")

    # Regression line
    x = np.array(seq_lens, dtype=float)
    y = np.array(sink_mass, dtype=float)
    if len(x) > 2 and np.std(x) > 0:
        from scipy import stats
        slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2,
                label=f"r={r_val:.3f}, p={p_val:.2e}")
        ax.legend()

    ax.set_title("Sequence Length vs. Sink Attention Mass", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Total Sink Attention Mass")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_results()
    print(f"Loaded {len(rows)} results. Generating plots...")

    spike_scores = [r["spike_score_max"] for r in rows]
    sink_masses = [r["sink_attn_mass_total"] for r in rows]
    jaccards = [r["jaccard"] for r in rows]

    plot_histogram(spike_scores, "Max Spike Score Distribution", "Max Spike Score",
                   "spike_score_distribution.png", color="#C44E52")
    plot_histogram(sink_masses, "Total Sink Attention Mass Distribution", "Sink Attention Mass",
                   "sink_mass_distribution.png", color="#4C72B0")
    plot_histogram(jaccards, "Jaccard Overlap Distribution (Spike ∩ Sink)", "Jaccard Index",
                   "jaccard_distribution.png", color="#55A868")
    plot_boxplot_by_category(rows, "sink_count_by_category.png")
    plot_spike_position_heatmap(rows, "spike_position_heatmap.png")
    plot_scatter_with_regression(rows, "seq_len_vs_sink_mass.png")

    print("All plots saved.")


if __name__ == "__main__":
    main()
