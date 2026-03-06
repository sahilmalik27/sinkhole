# Why Your LLM Is Ignoring Your Prompt (And Staring at a Newline Instead)

*Diagnosing attention sinks and activation spikes in transformer models with sinkhole.*

---

We ran Qwen2.5-7B-Instruct on a simple prompt:

> *"Explain the theory of relativity in simple terms."*

40 tokens. 28 layers. 784 attention heads.

One `\n` token — a newline after the system prompt — absorbed **55% of all attention** across **96% of the model's heads**.

Not the word "relativity." Not "explain." Not even the beginning-of-sequence token. A newline.

This is the attention sink problem. It's real, it's measurable, and it affects every pre-norm transformer you're running. Here's what it is, why it happens, and how to see it in your own models.

---

## The Two Phenomena

There are two things going on, and they're related:

### 1. Massive Activations (Spike Tokens)

Run any input through a Llama, Qwen, or Mistral model and look at the hidden states at each layer. Most tokens have activations in a normal range. But a handful of tokens — often just one or two — have activation values that are **10–100× larger** than the median, concentrated in a few specific channels.

These are called **spike tokens**. They're not random. They consistently appear at structurally significant positions: the `<|im_start|>` token, the first `\n`, the BOS token. And they appear in the same channels across different prompts.

In our Qwen2.5-7B run, the `\n` at position 2 had a spike score of **38.6×** above the median norm, concentrated in channels 458, 2570, 2718, and 2730.

### 2. Attention Sinks

In the attention mechanism, each head produces a distribution over tokens for each query. In theory, high-attention tokens should be semantically relevant. In practice, a paper from Yann LeCun's group at NYU showed something different:

> *"Certain tokens attract disproportionate attention mass regardless of semantic relevance."*

These are **sink tokens** — tokens that consistently receive high attention from most heads across most layers, independent of what the prompt actually says.

In our run: that same `\n` at position 2 received 55.1% of total attention mass, dominating 754 of 784 heads (96%).

### The Co-occurrence

The paper's core finding — and the most striking result from our own test — is that spike tokens and sink tokens are almost always the same tokens. In our Qwen2.5-7B run, the Jaccard overlap was exactly **1.0**.

This is not a coincidence. The mechanism is architectural.

---

## Why This Happens

The root cause is **pre-norm** — the design choice where RMSNorm is applied *before* each transformer block rather than after.

In modern LLMs, every block looks like this:

```
H[i+1] = H[i] + F(RMSNorm(H[i]))
```

RMSNorm normalizes each token's hidden state to unit norm before it goes into the attention or FFN block. But it operates on the *normalized* input — the raw hidden state `H[i]` flows through the residual stream unnormalized.

Here's what happens at spike tokens: their raw hidden states develop extremely large norms from the residual accumulation. When RMSNorm normalizes them, the scale is huge. The subsequent linear projection amplifies this into the attention QK computation. The result: when *any* other token queries, the spike token's key has an outsized inner product — it wins the softmax, consistently, across heads.

This is the mechanism. The spike in the hidden state directly produces the attention sink. They're two views of the same artifact.

The paper also shows you can suppress one without the other by changing the normalization configuration — which confirms they're coupled through the norm, not inherently linked.

---

## Why This Matters Practically

### KV Cache Waste

Every token in the sequence occupies a slot in the KV cache. In our example, 1 token out of 40 (2.5% of tokens) consumed 55% of attention mass. That attention budget is effectively wasted — it's going to a structurally fixed anchor, not to semantically relevant context.

In a 4096-token context, if 2–4 tokens are perpetual sinks, they take up slots and compute that could go to actual content. At scale, this adds up.

### Quantization Failures

INT8 and INT4 quantization schemes work by clipping activations to a fixed range. Spike channels — the few hidden dimensions where activation values are 10–100× the mean — blow through any reasonable clip range. This is why LLM quantization often fails silently on certain inputs: the spike token hits an extreme, clips, and corrupts the hidden representation for everything downstream.

### Pruning and Distillation

Sink heads — attention heads that are dominated by sink behavior — are functionally different from semantic heads. If you're pruning attention heads by importance score, you'll likely want to treat sink heads differently. Removing them might change the model's behavior in non-obvious ways even though they appear "unimportant" on standard metrics.

### Long-Context Performance

Sink token behavior compounds with context length. The longer your input, the more queries there are pointing at the sink, and the more compute is diverted from real context. This partially explains why some models degrade in quality at long contexts in ways that aren't fully explained by position encoding issues.

---

## Introducing sinkhole

**sinkhole** is a diagnostic CLI that makes all of this visible in one command:

```bash
sinkhole analyze \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompt "Explain the theory of relativity in simple terms." \
  --output report.html \
  --device cuda
```

It hooks into the model's forward pass, captures hidden states and attention weights across all layers and heads, and produces:

**Terminal output:**
```
╭─ sinkhole ─────────────────────────────────────────────────────╮
│ Model   Qwen/Qwen2.5-7B-Instruct                               │
│ Tokens  40   Layers  28   Heads  28                            │
╰────────────────────────────────────────────────────────────────╯

 Spike Tokens  1 found (threshold 10.0×)
 ┌──────┬───────┬────────┬──────────────────────────────┐
 │  Pos │ Token │  Score │ Spike Channels               │
 ├──────┼───────┼────────┼──────────────────────────────┤
 │    2 │ '\n'  │  38.6× │ [458, 2570, 2718, 2730]      │
 └──────┴───────┴────────┴──────────────────────────────┘

 Sink Tokens  1 found
 ┌──────┬───────┬────────────┬─────────────────────────┐
 │  Pos │ Token │ Attn Mass  │ Heads Dominated          │
 ├──────┼───────┼────────────┼─────────────────────────┤
 │    2 │ '\n'  │   55.1%    │ 754 / 784  (96%)        │
 └──────┴───────┴────────────┴─────────────────────────┘

 Spike ∩ Sink  Jaccard = 1.00
 KV Impact     55.1% of attention budget on sink tokens
```

**HTML report** with layer-by-layer attention heatmaps and activation norm plots. **JSON output** for integration with other tooling.

### Architecture support

sinkhole works with any pre-norm transformer exposed through HuggingFace:
- Llama 2, Llama 3, Llama 3.1
- Qwen2, Qwen2.5 (all sizes)
- Mistral 7B, Mixtral
- Phi-3

If your model uses RMSNorm + residual connections in the standard pre-norm configuration, you'll see these phenomena. The only question is *how strong* they are.

---

## What We Found on Qwen2.5-7B

The model we tested: `Qwen/Qwen2.5-7B-Instruct` (7 billion parameters, instruction-tuned, bfloat16).

Prompt: *"Explain the theory of relativity in simple terms."* (with the standard Qwen chat template applied — so the actual input to the model is 40 tokens including the system message).

| Metric | Value |
|--------|-------|
| Spike tokens detected | 1 |
| Spike token | `\n` at position 2 |
| Spike score | 38.6× above median |
| Spike channels | 458, 2570, 2718, 2730 |
| Sink tokens detected | 1 |
| Sink token | `\n` at position 2 |
| Attention mass on sink | 55.1% |
| Heads dominated | 754 / 784 (96%) |
| Spike ∩ Sink Jaccard | 1.0 |

The `\n` at position 2 is the newline that separates `<|im_start|>system` from `You are Qwen...` in the system prompt. It sits in a structurally conspicuous position early in the sequence and becomes the anchor that every head in the model refers back to.

This is exactly what the paper predicts. The pre-norm configuration, the structural position, the residual accumulation — it all adds up to one token vacuuming up more than half the model's attention.

---

## What You Can Do About It

The paper shows that both phenomena can be suppressed independently without degrading language modeling performance. Some practical directions:

**1. Identify your sinks before optimizing.** If you're designing a KV cache eviction policy, don't evict sink tokens — they're not semantically meaningful, but their absence breaks the model's learned attention patterns. sinkhole tells you which ones they are.

**2. Channel-aware quantization.** Know which channels are spike channels before choosing your quantization scheme. Spike channels need wider representation; the rest can be compressed aggressively.

**3. Architecture choices.** If you're training from scratch or fine-tuning with architectural modifications, the paper shows that post-norm configurations decouple spikes from sinks. Worth knowing before you commit to a design.

**4. Sink-aware pruning.** sinkhole reports which heads are sink-dominated. In a head pruning experiment, you'd want to separate "sink heads" from "semantic heads" and evaluate them separately.

---

## Get Started

```bash
pip install git+https://github.com/sahilmalik27/sinkhole
sinkhole analyze --model Qwen/Qwen2.5-7B-Instruct --prompt "your prompt here" --device cuda
```

The full results from our Qwen2.5-7B run — including the HTML heatmap report — are in the `results/` directory of the repo.

Source code, tests, and documentation: **[github.com/sahilmalik27/sinkhole](https://github.com/sahilmalik27/sinkhole)**

---

## Reference

> Sun, S., Canziani, A., & LeCun, Y. (2026). *The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks.* arXiv:2603.05498. ICML 2026.
