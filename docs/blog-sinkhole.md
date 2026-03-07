# Why Your LLM Is Ignoring Your Prompt (And Staring at a Newline Instead)

*Diagnosing attention sinks and activation spikes in transformer models with sinkhole.*

---

We ran Qwen2.5-7B-Instruct on 400 diverse prompts — factual questions, coding tasks, open-ended instructions, reasoning problems. For every single one, we measured which tokens received the most attention and which had the largest activation norms.

In all 400 runs, one token absorbed the majority of all attention in the model.

Not the question word. Not the topic. Not any token in the actual prompt.

A `\n` character — the newline separating `<|im_start|>system` from `You are Qwen...` in the chat template.

That single whitespace token captured **54.9% of all attention** across **96.2% of the model's heads**, in every prompt, regardless of content.

This is the attention sink problem. Here's what it is, why it happens, what we measured, and what you can do about it.

---

## The Two Phenomena

### 1. Massive Activations (Spike Tokens)

When you run any input through a Llama, Qwen, or Mistral model, most tokens have hidden-state activations within a normal range. But a tiny number of tokens have activation values **10–100× larger** than the median, concentrated in just a few specific hidden channels.

These are **spike tokens**. They show up consistently at structurally significant positions — the first `\n`, the `<|im_start|>` token, the BOS token — and they appear in the same channels regardless of the prompt content.

In our evaluation: across 400 prompts, the spike token was **always** `\n` at position 2, with a mean score of **38.2×** above median, concentrated in exactly channels 2730, 458, 2570, and 2718. Every time. Zero variance on the channel set.

### 2. Attention Sinks

In a well-functioning attention head, tokens with high attention weights should be semantically relevant to the query. In practice, something different happens.

**Sink tokens** attract disproportionate attention mass across most heads and most layers, regardless of what the prompt says or asks. They're not answering the question — they're just absorbing attention that could go elsewhere.

In our evaluation: the same `\n` token at position 2 absorbed **54.9%** of total attention mass on average, dominating **96.2% of the model's 784 attention heads**.

### The Co-occurrence

The paper's central claim — and the most striking thing about our results — is that spike tokens and sink tokens are the same tokens. In all 400 prompts, the Jaccard overlap between spike tokens and sink tokens was exactly **1.0**.

This is not a coincidence. It's architectural.

---

## The Numbers

We ran a full statistical evaluation using sinkhole across 400 prompts from a mix of public instruction datasets. Here are the results.

### Aggregate statistics (n=400)

| Metric | Mean | Std | 95% CI |
|--------|------|-----|--------|
| Spike tokens per prompt | **1.00** | 0.00 | [1.00, 1.00] |
| Spike score (× median norm) | **38.2×** | 0.69 | [38.15, 38.29] |
| Spike channels | **always [2730, 458, 2570, 2718]** | — | — |
| Sink tokens per prompt | **1.00** | 0.00 | [1.00, 1.00] |
| Sink attention mass | **54.9%** | 0.54% | [54.88%, 54.98%] |
| Heads dominated | **96.25%** | 0.11% | [96.24%, 96.27%] |
| Spike ∩ Sink (Jaccard) | **1.00** | 0.00 | [1.00, 1.00] |

All four hypothesis tests significant at p < 0.001. The t-statistic for "sink mass > 30%" is **918.7**.

### It doesn't matter what you ask

Here are eight randomly selected prompts from the evaluation with their results:

| Prompt | Spike score | Sink mass | Heads |
|--------|-------------|-----------|-------|
| "Tell me three short-term effects of smoking marijuana." | 37.7× | 55.3% | 96.3% |
| "Generate a website design for a house cleaning company" | 38.7× | 55.1% | 96.3% |
| "How many continents are there on Earth?" | 38.5× | 55.1% | 96.2% |
| "Arrange the following musical notes" | 38.6× | 55.5% | 96.2% |
| "Explain the concept of socio-economic privilege." | 38.1× | 55.4% | 96.2% |
| "Classify these five animals into two groups." | 38.3× | 55.0% | 96.3% |
| "List five benefits of going for a walk" | 38.7× | 55.9% | 96.2% |
| "Pick the best response based on the given situation." | 39.7× | 55.2% | 96.2% |

The spike token is always `\n`. The spike channels are always [2730, 458, 2570, 2718]. The Jaccard is always 1.0. The sink mass sits between 51.5% and 56.1% across all 400 prompts.

The prompt content is irrelevant. This is a structural property of the model.

### One interesting finding: sequence length matters

There's a statistically significant negative correlation between prompt length and sink dominance (Pearson r = −0.52, p < 0.001). Longer prompts dilute the sink effect slightly — the sink token absorbs less attention as the total number of tokens grows. This aligns with the paper's discussion of long-context behavior: sinks don't disappear, but their relative share shrinks.

Even at the longest prompts in our evaluation (60 tokens), the sink still absorbed more than 50% of attention. It never goes away — it just becomes slightly less dominant.

---

## Why This Happens

The root cause is **pre-norm** — the choice to apply RMSNorm *before* each transformer block rather than after.

Every block in a modern LLM looks like:

```
H[i+1] = H[i] + F(RMSNorm(H[i]))
```

The raw hidden state `H[i]` flows through the residual stream without normalization. At structurally prominent positions — early tokens like `\n` that appear at consistent locations across all inputs — the residual accumulation produces extremely large hidden-state norms. When RMSNorm normalizes these, the resulting representation has a large scale that gets amplified by the attention QK projection.

The effect: when any other token queries the attention mechanism, the spike token's key has an outsized inner product across almost every head. It consistently wins the softmax. The spike in the hidden state is both the cause and the signature of the attention sink.

This is why channels 2730, 458, 2570, and 2718 are always the spike channels — they're the dimensions where residual accumulation happens to concentrate for this particular model's learned weight structure.

The paper (Sun, Canziani, LeCun, ICML 2026) shows that changing to post-norm decouples the two phenomena. Both can be independently suppressed without degrading language modeling performance.

---

## Why This Matters Practically

### KV Cache Waste

1 token out of ~40 (2.5% of context) consumes 55% of attention compute. At 4096-token contexts, the absolute waste is larger and the sink behavior persists. Any KV cache eviction strategy needs to account for sink tokens — they should not be evicted (their absence breaks attention patterns) but they also shouldn't count against the semantic budget.

sinkhole tells you exactly which tokens those are.

### Quantization Failures

Spike channels blow through any reasonable INT8/INT4 clipping range. Those 4 channels — 2730, 458, 2570, 2718 — need wider representation. If your quantization scheme doesn't know about them, it clips them, corrupts the hidden state for all downstream tokens, and you get silent quality degradation.

### Sink-Aware Pruning

Sink heads behave fundamentally differently from semantic heads. A head that's 96% pointing at `\n` is not doing the same thing as a head tracking subject-verb agreement or long-range coreference. If you prune by importance score alone, you might eliminate sink heads without understanding what role they play in calibrating the model's short-range attention patterns.

### Long-Context Degradation

The r = −0.52 correlation shows sinks dilute with length, but they never disappear. At very long contexts, this means a smaller fraction of attention is "wasted" on the sink — but the absolute number of attention operations going to a semantically empty token remains large.

---

## Introducing sinkhole

```bash
pip install git+https://github.com/sahilmalik27/sinkhole
sinkhole analyze --model Qwen/Qwen2.5-7B-Instruct --prompt "your prompt" --device cuda
```

sinkhole hooks into the model's forward pass, captures hidden states and attention weights across all layers and heads, and outputs:

- **Terminal report** — spike/sink tables with scores, positions, channels, head coverage
- **HTML report** — interactive attention heatmaps, norm plots, sink mass charts
- **JSON** — structured output for integration with other tooling

It works with any pre-norm transformer: Llama 2/3, Qwen2/2.5, Mistral, Phi-3.

### What the output looks like

```
 Spike Tokens  1 found (threshold 10.0×)
 ┌──────┬───────┬────────┬──────────────────────────────┐
 │  Pos │ Token │  Score │ Spike Channels               │
 ├──────┼───────┼────────┼──────────────────────────────┤
 │    2 │ '\n'  │  38.2× │ [2730, 458, 2570, 2718]      │
 └──────┴───────┴────────┴──────────────────────────────┘

 Sink Tokens  1 found
 ┌──────┬───────┬────────────┬──────────────────────────┐
 │  Pos │ Token │ Attn Mass  │ Heads Dominated           │
 ├──────┼───────┼────────────┼──────────────────────────┤
 │    2 │ '\n'  │   54.9%    │ 754 / 784  (96.2%)       │
 └──────┴───────┴────────────┴──────────────────────────┘

 Spike ∩ Sink  Jaccard = 1.00   ·   KV waste: 54.9%
```

---

## What You Can Do About It

**1. Identify sinks before designing KV eviction.** Never evict sink tokens from your KV cache — their absence will break the model's learned attention patterns. But keep them clearly labeled so they don't count against your semantic context budget.

**2. Channel-aware quantization.** The spike channels (2730, 458, 2570, 2718 for Qwen2.5-7B) need special handling in any quantization scheme. Use outlier-aware approaches like SmoothQuant or LLM.int8() that can widen specific channels.

**3. Separate sink heads from semantic heads.** Before pruning, use sinkhole to identify which heads are sink-dominated and which are doing semantic work. They should be evaluated independently.

**4. Architecture.** If you're training from scratch, post-norm decouples spikes from sinks (per the paper). Pre-norm is faster to train and more stable, but this is the tradeoff.

---

## Reproducing These Results

All evaluation code, raw results (400 JSONL records), aggregated statistics, and figures are in the repo:

```bash
git clone https://github.com/sahilmalik27/sinkhole
cd sinkhole
pip install -e '.[dev]'
pip install datasets scipy tqdm

# Re-run the evaluation (requires Qwen2.5-7B-Instruct + GPU)
python eval/run_eval.py

# Compute stats and generate plots
python eval/stats.py && python eval/plot.py && python eval/report.py
```

The evaluation is resume-safe — if it's interrupted, it picks up where it left off.

---

## Reference

> Sun, S., Canziani, A., & LeCun, Y. (2026). *The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks.* arXiv:2603.05498. ICML 2026.

Source: **[github.com/sahilmalik27/sinkhole](https://github.com/sahilmalik27/sinkhole)**
