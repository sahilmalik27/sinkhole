"""Compute aggregate statistics and hypothesis tests from raw results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_RESULTS_PATH = RESULTS_DIR / "raw_results.jsonl"
STATS_PATH = RESULTS_DIR / "stats.json"


def load_results() -> list[dict]:
    rows = []
    with open(RAW_RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_descriptive(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0,
                "p5": 0, "p25": 0, "p75": 0, "p95": 0, "n": 0,
                "ci95_low": 0, "ci95_high": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    se = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    ci95_low = mean - 1.96 * se
    ci95_high = mean + 1.96 * se
    return {
        "mean": mean,
        "std": std,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "n": len(arr),
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
    }


def main():
    rows = load_results()
    n = len(rows)
    print(f"Loaded {n} results")

    # Extract metric arrays
    metrics = {
        "spike_count": [r["spike_count"] for r in rows],
        "spike_score_max": [r["spike_score_max"] for r in rows],
        "spike_score_mean": [r["spike_score_mean"] for r in rows],
        "sink_count": [r["sink_count"] for r in rows],
        "sink_attn_mass_max": [r["sink_attn_mass_max"] for r in rows],
        "sink_attn_mass_total": [r["sink_attn_mass_total"] for r in rows],
        "sink_head_coverage": [r["sink_head_coverage"] for r in rows],
        "jaccard": [r["jaccard"] for r in rows],
        "overlap_count": [r["overlap_count"] for r in rows],
        "seq_len": [r["seq_len"] for r in rows],
    }

    # Descriptive stats
    descriptive = {k: compute_descriptive(v) for k, v in metrics.items()}

    # Hypothesis tests
    hypothesis_tests = {}

    # 1. One-sample t-test: mean Jaccard > 0
    jaccard = np.array(metrics["jaccard"])
    if np.std(jaccard) > 0:
        t_stat, p_val = stats.ttest_1samp(jaccard, 0)
        # One-sided: we want jaccard > 0
        p_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
    else:
        t_stat = float("inf") if np.mean(jaccard) > 0 else 0.0
        p_one_sided = 0.0 if np.mean(jaccard) > 0 else 1.0
    hypothesis_tests["jaccard_gt_0"] = {
        "test": "one-sample t-test (one-sided)",
        "h0": "mean Jaccard == 0",
        "h1": "mean Jaccard > 0",
        "t_statistic": float(t_stat),
        "p_value": float(p_one_sided),
        "significant_at_001": bool(p_one_sided < 0.01),
        "mean": float(np.mean(jaccard)),
    }

    # 2. One-sample t-test: mean sink_attn_mass_total > 0.3
    sink_mass = np.array(metrics["sink_attn_mass_total"])
    if np.std(sink_mass) > 0:
        t_stat2, p_val2 = stats.ttest_1samp(sink_mass, 0.3)
        p_one_sided2 = p_val2 / 2 if t_stat2 > 0 else 1 - p_val2 / 2
    else:
        t_stat2 = float("inf") if np.mean(sink_mass) > 0.3 else float("-inf")
        p_one_sided2 = 0.0 if np.mean(sink_mass) > 0.3 else 1.0
    hypothesis_tests["sink_mass_gt_03"] = {
        "test": "one-sample t-test (one-sided)",
        "h0": "mean sink_attn_mass_total <= 0.3",
        "h1": "mean sink_attn_mass_total > 0.3",
        "t_statistic": float(t_stat2),
        "p_value": float(p_one_sided2),
        "significant_at_001": bool(p_one_sided2 < 0.01),
        "mean": float(np.mean(sink_mass)),
    }

    # 3. Chi-square: spike positions concentrated at structural tokens
    all_spike_types = []
    for r in rows:
        all_spike_types.extend(r.get("spike_position_types", []))

    type_counts = {}
    for t in all_spike_types:
        type_counts[t] = type_counts.get(t, 0) + 1

    if len(type_counts) > 1:
        observed = np.array(list(type_counts.values()))
        # Expected: uniform distribution
        expected = np.full_like(observed, dtype=float, fill_value=observed.sum() / len(observed))
        chi2_stat, chi2_p = stats.chisquare(observed, expected)
    else:
        chi2_stat, chi2_p = 0.0, 1.0

    structural_count = sum(type_counts.get(t, 0) for t in ["special", "newline", "punctuation", "whitespace"])
    content_count = type_counts.get("content", 0)
    total_spikes = structural_count + content_count

    hypothesis_tests["spike_position_chi2"] = {
        "test": "chi-square goodness-of-fit",
        "h0": "spike positions uniformly distributed across token types",
        "h1": "spike positions concentrated at structural tokens",
        "chi2_statistic": float(chi2_stat),
        "p_value": float(chi2_p),
        "significant_at_005": bool(chi2_p < 0.05),
        "type_counts": type_counts,
        "structural_fraction": structural_count / total_spikes if total_spikes > 0 else 0.0,
    }

    # 4. Pearson correlation: seq_len vs sink_attn_mass_total
    seq_lens = np.array(metrics["seq_len"], dtype=np.float64)
    if np.std(seq_lens) > 0 and np.std(sink_mass) > 0:
        r_val, r_p = stats.pearsonr(seq_lens, sink_mass)
    else:
        r_val, r_p = 0.0, 1.0

    hypothesis_tests["seq_len_vs_sink_mass_correlation"] = {
        "test": "Pearson correlation",
        "h0": "no linear correlation between seq_len and sink_attn_mass_total",
        "r": float(r_val),
        "p_value": float(r_p),
        "significant_at_005": bool(r_p < 0.05),
    }

    # Per-category stats
    categories = {}
    for r in rows:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["sink_count"])
    category_stats = {cat: compute_descriptive(vals) for cat, vals in categories.items()}

    # Build output
    output = {
        "n_prompts": n,
        "descriptive": descriptive,
        "hypothesis_tests": hypothesis_tests,
        "category_sink_count": category_stats,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Stats saved to {STATS_PATH}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"STATISTICAL SUMMARY ({n} prompts)")
    print(f"{'='*60}")
    for metric, desc in descriptive.items():
        print(f"  {metric}: mean={desc['mean']:.4f} std={desc['std']:.4f} "
              f"median={desc['median']:.4f} [P5={desc['p5']:.4f}, P95={desc['p95']:.4f}]")
    print(f"\nHypothesis tests:")
    for name, test in hypothesis_tests.items():
        sig = "***" if test["p_value"] < 0.001 else "**" if test["p_value"] < 0.01 else "*" if test["p_value"] < 0.05 else "ns"
        print(f"  {name}: p={test['p_value']:.6f} {sig}")


if __name__ == "__main__":
    main()
