# Sinkhole Statistical Evaluation Report

**Model**: Qwen/Qwen2.5-7B-Instruct
**Prompts evaluated**: 400
**Dataset**: Mixed (Alpaca, TriviaQA, HumanEval, MMLU, ShareGPT)

## Executive Summary

Across 400 diverse prompts, sinkhole consistently detects the spike-sink overlap phenomenon:

- **Spike-sink overlap (Jaccard)**: mean = 1.0000 (95% CI: [1.0000, 1.0000])
- **Total sink attention mass**: mean = 0.5493 (95% CI: [0.5488, 0.5498])
- **Max spike score**: mean = 38.2218 (95% CI: [38.1541, 38.2894])

The spike-sink Jaccard overlap is significantly greater than zero (t = inf, p = 0.000000), confirming the paper's core claim. Sink tokens absorb significantly more 30% of total attention mass (t = 918.70, p = 0.000000).

## Aggregate Statistics

| Metric | Mean | Std | Median | P5 | P25 | P75 | P95 | Min | Max | 95% CI |
|--------|------|-----|--------|-----|-----|-----|-----|-----|-----|--------|
| spike_count | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [1.0000, 1.0000] |
| spike_score_max | 38.2218 | 0.6901 | 38.2639 | 37.0142 | 37.7257 | 38.6179 | 39.3633 | 35.8799 | 39.7636 | [38.1541, 38.2894] |
| spike_score_mean | 38.2218 | 0.6901 | 38.2639 | 37.0142 | 37.7257 | 38.6179 | 39.3633 | 35.8799 | 39.7636 | [38.1541, 38.2894] |
| sink_count | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [1.0000, 1.0000] |
| sink_attn_mass_max | 0.5493 | 0.0054 | 0.5497 | 0.5396 | 0.5466 | 0.5527 | 0.5565 | 0.5147 | 0.5611 | [0.5488, 0.5498] |
| sink_attn_mass_total | 0.5493 | 0.0054 | 0.5497 | 0.5396 | 0.5466 | 0.5527 | 0.5565 | 0.5147 | 0.5611 | [0.5488, 0.5498] |
| sink_head_coverage | 0.9625 | 0.0011 | 0.9617 | 0.9617 | 0.9617 | 0.9630 | 0.9643 | 0.9605 | 0.9668 | [0.9624, 0.9627] |
| jaccard | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [1.0000, 1.0000] |
| overlap_count | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [1.0000, 1.0000] |
| seq_len | 41.0000 | 4.0424 | 40.0000 | 35.0000 | 38.0000 | 43.0000 | 48.0000 | 33.0000 | 60.0000 | [40.6038, 41.3962] |

## Hypothesis Tests

### jaccard_gt_0

- **Test**: one-sample t-test (one-sided)
- **H0**: mean Jaccard == 0
- **H1**: mean Jaccard > 0
- **t-statistic**: inf
- **p-value**: p < 0.001 ***

### sink_mass_gt_03

- **Test**: one-sample t-test (one-sided)
- **H0**: mean sink_attn_mass_total <= 0.3
- **H1**: mean sink_attn_mass_total > 0.3
- **t-statistic**: 918.6975
- **p-value**: p < 0.001 ***

### spike_position_chi2

- **Test**: chi-square goodness-of-fit
- **H0**: spike positions uniformly distributed across token types
- **H1**: spike positions concentrated at structural tokens
- **chi2-statistic**: 0.0000
- **p-value**: p = 1.000000 (not significant)
- **Token type counts**: {'whitespace': 400}
- **Structural token fraction**: 1.0000

### seq_len_vs_sink_mass_correlation

- **Test**: Pearson correlation
- **H0**: no linear correlation between seq_len and sink_attn_mass_total
- **r (Pearson)**: -0.5229
- **p-value**: p < 0.001 ***

## Sink Count by Category

| Category | Mean | Std | Median | N |
|----------|------|-----|--------|---|
| instruction | 1.0000 | 0.0000 | 1.0000 | 400 |

## Figures

| Figure | Path |
|--------|------|
| Spike score distribution | `eval/results/figures/spike_score_distribution.png` |
| Sink mass distribution | `eval/results/figures/sink_mass_distribution.png` |
| Jaccard distribution | `eval/results/figures/jaccard_distribution.png` |
| Sink count by category | `eval/results/figures/sink_count_by_category.png` |
| Spike position heatmap | `eval/results/figures/spike_position_heatmap.png` |
| Seq len vs sink mass | `eval/results/figures/seq_len_vs_sink_mass.png` |

