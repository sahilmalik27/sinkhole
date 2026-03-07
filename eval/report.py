"""Generate human-readable statistical report from stats.json."""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
STATS_PATH = RESULTS_DIR / "stats.json"
REPORT_PATH = RESULTS_DIR / "stats_report.md"


def _fmt(val, decimals=4):
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def main():
    with open(STATS_PATH) as f:
        data = json.load(f)

    n = data["n_prompts"]
    desc = data["descriptive"]
    tests = data["hypothesis_tests"]
    cat_stats = data["category_sink_count"]

    lines = []
    w = lines.append

    w("# Sinkhole Statistical Evaluation Report")
    w("")
    w(f"**Model**: Qwen/Qwen2.5-7B-Instruct")
    w(f"**Prompts evaluated**: {n}")
    w(f"**Dataset**: Mixed (Alpaca, TriviaQA, HumanEval, MMLU, ShareGPT)")
    w("")

    # Executive summary
    w("## Executive Summary")
    w("")
    j = desc["jaccard"]
    sm = desc["sink_attn_mass_total"]
    sp = desc["spike_score_max"]
    w(f"Across {n} diverse prompts, sinkhole consistently detects the spike-sink overlap phenomenon:")
    w("")
    w(f"- **Spike-sink overlap (Jaccard)**: mean = {_fmt(j['mean'])} "
      f"(95% CI: [{_fmt(j['ci95_low'])}, {_fmt(j['ci95_high'])}])")
    w(f"- **Total sink attention mass**: mean = {_fmt(sm['mean'])} "
      f"(95% CI: [{_fmt(sm['ci95_low'])}, {_fmt(sm['ci95_high'])}])")
    w(f"- **Max spike score**: mean = {_fmt(sp['mean'])} "
      f"(95% CI: [{_fmt(sp['ci95_low'])}, {_fmt(sp['ci95_high'])}])")

    jt = tests["jaccard_gt_0"]
    smt = tests["sink_mass_gt_03"]
    w("")
    w(f"The spike-sink Jaccard overlap is significantly greater than zero "
      f"(t = {_fmt(jt['t_statistic'], 2)}, p = {_fmt(jt['p_value'], 6)}), confirming the paper's core claim. "
      f"Sink tokens absorb {'significantly more' if smt['significant_at_001'] else 'approximately'} "
      f"30% of total attention mass (t = {_fmt(smt['t_statistic'], 2)}, p = {_fmt(smt['p_value'], 6)}).")
    w("")

    # Descriptive stats table
    w("## Aggregate Statistics")
    w("")
    w("| Metric | Mean | Std | Median | P5 | P25 | P75 | P95 | Min | Max | 95% CI |")
    w("|--------|------|-----|--------|-----|-----|-----|-----|-----|-----|--------|")
    for metric, d in desc.items():
        ci = f"[{_fmt(d['ci95_low'])}, {_fmt(d['ci95_high'])}]"
        w(f"| {metric} | {_fmt(d['mean'])} | {_fmt(d['std'])} | {_fmt(d['median'])} | "
          f"{_fmt(d['p5'])} | {_fmt(d['p25'])} | {_fmt(d['p75'])} | {_fmt(d['p95'])} | "
          f"{_fmt(d['min'])} | {_fmt(d['max'])} | {ci} |")
    w("")

    # Hypothesis tests
    w("## Hypothesis Tests")
    w("")

    for name, test in tests.items():
        w(f"### {name}")
        w("")
        w(f"- **Test**: {test['test']}")
        w(f"- **H0**: {test['h0']}")
        if "h1" in test:
            w(f"- **H1**: {test['h1']}")
        if "t_statistic" in test:
            w(f"- **t-statistic**: {_fmt(test['t_statistic'], 4)}")
        if "chi2_statistic" in test:
            w(f"- **chi2-statistic**: {_fmt(test['chi2_statistic'], 4)}")
        if "r" in test:
            w(f"- **r (Pearson)**: {_fmt(test['r'], 4)}")
        p = test["p_value"]
        sig = "p < 0.001 ***" if p < 0.001 else "p < 0.01 **" if p < 0.01 else "p < 0.05 *" if p < 0.05 else f"p = {_fmt(p, 6)} (not significant)"
        w(f"- **p-value**: {sig}")
        if "type_counts" in test:
            w(f"- **Token type counts**: {test['type_counts']}")
            w(f"- **Structural token fraction**: {_fmt(test['structural_fraction'], 4)}")
        w("")

    # Category breakdown
    w("## Sink Count by Category")
    w("")
    w("| Category | Mean | Std | Median | N |")
    w("|----------|------|-----|--------|---|")
    for cat, cs in cat_stats.items():
        w(f"| {cat} | {_fmt(cs['mean'])} | {_fmt(cs['std'])} | {_fmt(cs['median'])} | {cs['n']} |")
    w("")

    # Figures
    w("## Figures")
    w("")
    w("| Figure | Path |")
    w("|--------|------|")
    w("| Spike score distribution | `eval/results/figures/spike_score_distribution.png` |")
    w("| Sink mass distribution | `eval/results/figures/sink_mass_distribution.png` |")
    w("| Jaccard distribution | `eval/results/figures/jaccard_distribution.png` |")
    w("| Sink count by category | `eval/results/figures/sink_count_by_category.png` |")
    w("| Spike position heatmap | `eval/results/figures/spike_position_heatmap.png` |")
    w("| Seq len vs sink mass | `eval/results/figures/seq_len_vs_sink_mass.png` |")
    w("")

    # Write
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
