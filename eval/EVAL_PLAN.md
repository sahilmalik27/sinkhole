# sinkhole — Statistical Evaluation Plan

## Goal

Replace single-example claims with statistically significant results across 1000 diverse prompts on Qwen2.5-7B-Instruct.

## What to measure (per prompt)

### Spike metrics
- `spike_count` — number of spike tokens detected (threshold=10x)
- `spike_score_max` — highest spike score in the sequence
- `spike_score_mean` — mean spike score across spike tokens
- `spike_positions` — which positions spike (BOS, EOS, \n, punctuation, content?)
- `spike_token_text` — actual token text for each spike token
- `spike_channels` — which channels spike (are they consistent across prompts?)

### Sink metrics
- `sink_count` — number of sink tokens
- `sink_attn_mass_max` — highest attention mass on a single token
- `sink_attn_mass_total` — total attention absorbed by all sink tokens
- `sink_head_coverage` — fraction of heads dominated by sinks
- `sink_positions` — which positions become sinks
- `sink_token_text` — actual token text

### Overlap metrics
- `jaccard` — spike ∩ sink / spike ∪ sink (the paper's core claim)
- `overlap_count` — tokens that are both spike and sink

### Sequence characteristics
- `seq_len` — prompt length in tokens
- `prompt_category` — category of the prompt (factual, reasoning, creative, code, etc.)

## Dataset

Use a mix of 1000 prompts from public datasets (no private data):

1. **Alpaca** (200 prompts) — `tatsu-lab/alpaca` from HuggingFace datasets, `instruction` field
2. **TriviaQA** (200 prompts) — `trivia_qa`, `unfiltered`, question field
3. **HumanEval** (100 prompts) — `openai/openai_humaneval`, prompt field
4. **MMLU** (200 prompts) — `cais/mmlu`, `all` subset, question field
5. **ShareGPT-style** (300 prompts) — `anon8231489123/ShareGPT_Vicuna_unfiltered`, first user turn

All datasets are public on HuggingFace. Sample deterministically (seed=42) for reproducibility.

Apply the Qwen2.5-7B-Instruct chat template to each prompt before tokenizing.

## Statistical analysis

For each metric, compute:
- Mean ± std
- Median, P5, P25, P75, P95
- Distribution histogram

Key hypothesis tests:
1. **Jaccard overlap > 0** — one-sample t-test: mean Jaccard significantly > 0
2. **Sink tokens absorb majority of attention** — test: mean sink_attn_mass_total > 0.3
3. **Spike position consistency** — chi-square test: are spikes concentrated at specific structural positions (BOS, \n, punctuation)?
4. **Sink count stability** — is sink_count consistent across diverse prompts?

## Output files

- `eval/results/raw_results.jsonl` — one JSON per prompt (all metrics)
- `eval/results/stats.json` — aggregate statistics with confidence intervals
- `eval/results/stats_report.md` — human-readable statistical report
- `eval/results/figures/` — distribution plots (spike score, sink mass, Jaccard, etc.)

## Script structure

```
eval/
  run_eval.py        — main evaluation loop (1000 prompts, progress bar)
  dataset.py         — load and sample from HuggingFace datasets
  stats.py           — compute statistics, run hypothesis tests
  plot.py            — generate distribution plots
  report.py          — write stats_report.md
  EVAL_PLAN.md       — this file
```

## Runtime estimate

Qwen2.5-7B-Instruct forward pass with output_attentions=True: ~2-5 seconds per prompt on GPU.
1000 prompts × 3s avg = ~50 minutes. Run with tqdm progress bar.

## Success criteria

- [ ] 1000 prompts processed without OOM
- [ ] Jaccard mean > 0.5 with p < 0.01 (paper's core claim)
- [ ] Sink tokens absorb > 30% attention on average (p < 0.01)
- [ ] Spike positions shown to be non-random (chi-square p < 0.05)
- [ ] All results in eval/results/, committed to repo
- [ ] stats_report.md updates README claims with real numbers
