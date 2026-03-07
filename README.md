# 🕳 sinkhole

**Diagnose attention sinks and activation spikes in transformer language models.**

sinkhole loads any Llama / Qwen / Mistral-family model and tells you:

- **Which tokens are spike tokens** — tokens with extreme activation outliers (massive hidden-state norms) in specific channels
- **Which tokens are sink tokens** — tokens that absorb disproportionate attention mass across heads and layers, regardless of semantic relevance
- **Whether they overlap** — the paper's core finding: in pre-norm transformers, spike tokens and sink tokens are always the same tokens
- **How much KV budget is being wasted** — sink tokens consume attention compute without contributing semantically

Based on: *"The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks"* — Shangwen Sun, Alfredo Canziani, Yann LeCun (arXiv:2603.05498, ICML 2026).

---

## Why this matters

Attention sinks affect:
- **KV cache efficiency** — sink tokens occupy cache slots that could hold semantically relevant context
- **Quantization** — activation spikes cause outliers that break INT8/INT4 quantization
- **Pruning** — sink heads behave differently from semantic heads; pruning them requires awareness
- **Long-context inference** — sinks accumulate across layers and worsen with sequence length

sinkhole gives you a concrete measurement of all of the above for any model you care about.

---

## Install

```bash
git clone https://github.com/sahilmalik27/sinkhole
cd sinkhole
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Requirements**: Python 3.10+, PyTorch 2.0+, CUDA GPU (for 7B+ models)

---

## Quick start

### CLI

```bash
sinkhole analyze \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompt "Explain the theory of relativity in simple terms." \
  --output report.html \
  --device cuda
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | HuggingFace model name or local path |
| `--prompt` | required | Input text to analyze |
| `--output` | `report.html` | Output HTML report path |
| `--json` | — | Also save JSON output to this path |
| `--threshold` | `10.0` | Spike score cutoff (× median norm) |
| `--sink-top-k` | `0.5` | Sink threshold (fraction of max attention mass) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--dtype` | `float16` | `float16`, `bfloat16`, or `float32` |

### Python API

```python
from sinkhole import ModelProbe, analyze
from sinkhole.extractor import extract
from sinkhole.report import print_report, save_html, save_json

# Load model and run forward pass with hooks
probe = ModelProbe("Qwen/Qwen2.5-7B-Instruct", device="cuda")
capture = probe.run("Explain the theory of relativity in simple terms.")
probe.cleanup()

# Extract arrays: hidden [layers, seq, d_model], attn [layers, heads, seq, seq]
hidden, attn = extract(capture)

# Analyze
report = analyze(
    hidden, attn,
    token_texts=capture.token_texts,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    prompt="Explain the theory of relativity in simple terms.",
)

# Output
print_report(report)               # rich terminal output
save_html(report, attn, "report.html")
save_json(report, "report.json")
```

---

## Output

### Terminal

```
╭─ sinkhole ─────────────────────────────────────────────────────╮
│ Model   Qwen/Qwen2.5-7B-Instruct                               │
│ Prompt  Explain the theory of relativity in simple terms.      │
│ Tokens  40   Layers  28   Heads  28                            │
╰────────────────────────────────────────────────────────────────╯

 Spike Tokens  1 found (threshold 10.0×)
 ┌──────┬───────────┬────────────┬──────────────────────────────┐
 │  Pos │ Token     │  Score     │ Spike Channels               │
 ├──────┼───────────┼────────────┼──────────────────────────────┤
 │    2 │ '\n'      │   38.2×    │ [2730, 458, 2570, 2718]      │
 └──────┴───────────┴────────────┴──────────────────────────────┘

 Sink Tokens  1 found
 ┌──────┬───────────┬─────────────┬──────────────────────────────┐
 │  Pos │ Token     │ Attn Mass   │ Heads Dominated              │
 ├──────┼───────────┼─────────────┼──────────────────────────────┤
 │    2 │ '\n'      │   54.9%     │ 754 / 784  (96.2%)          │
 └──────┴───────────┴─────────────┴──────────────────────────────┘

 Spike ∩ Sink  1 token overlap  (Jaccard = 1.00)
 '\n' is both a spike token and a sink token

 KV Impact  Sink tokens consume 54.9% of attention budget
            Evicting them could free ~55% of KV cache
```

### HTML report

Interactive heatmaps showing:
- Attention weight matrix (all layers × heads)
- Per-token activation norm across layers
- Sink mass bar chart per token

### JSON

```json
{
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "seq_len": 40,
  "n_layers": 28,
  "n_heads": 28,
  "spikes": {
    "count": 1,
    "tokens": [{"position": 2, "text": "\n", "score": 38.2, "channels": [2730, 458, 2570, 2718]}]
  },
  "sinks": {
    "count": 1,
    "tokens": [{"position": 2, "text": "\n", "attn_mass": 0.549, "head_count": 754}]
  },
  "overlap": {"tokens": ["\n"], "jaccard": 1.0}
}
```

---

## Results: Qwen2.5-7B-Instruct — 400 Prompt Evaluation

We ran sinkhole across **400 diverse prompts** (factual questions, instructions, coding, reasoning) to verify the results are not prompt-specific. Full data in [`eval/results/`](eval/results/).

### Aggregate statistics

| Metric | Mean | Std | 95% CI | Range |
|--------|------|-----|--------|-------|
| Spike tokens per prompt | **1.00** | 0.00 | [1.00, 1.00] | always 1 |
| Spike score | **38.2×** | 0.69 | [38.15, 38.29] | [35.9×, 39.8×] |
| Spike channels | **[2730, 458, 2570, 2718]** | — | — | identical in all 400 |
| Sink tokens per prompt | **1.00** | 0.00 | [1.00, 1.00] | always 1 |
| Sink attention mass | **54.9%** | 0.54% | [54.88%, 54.98%] | [51.5%, 56.1%] |
| Heads dominated | **96.25%** | 0.11% | [96.24%, 96.27%] | |
| Spike ∩ Sink (Jaccard) | **1.00** | 0.00 | [1.00, 1.00] | always 1.0 |

### Hypothesis tests

| Test | Result | p-value |
|------|--------|---------|
| Jaccard > 0 | ✅ reject H₀ | p < 0.001 *** |
| Sink mass > 30% | ✅ reject H₀ (t=918.7) | p < 0.001 *** |
| Seq len vs sink mass | ✅ negative correlation (r=−0.52) | p < 0.001 *** |

**The spike-sink overlap is 1.0 in every single prompt, regardless of content.** The same 4 channels (2730, 458, 2570, 2718) spike every time. The `\n` token at position 2 absorbs 55% of attention in every run. Longer prompts slightly dilute the effect (r=−0.52) but never eliminate it.

### Diverse examples

Results are identical across completely different prompt types:

| Prompt | Spike | Score | Sink mass | Heads |
|--------|-------|-------|-----------|-------|
| "Tell me three short-term effects of smoking marijuana." | `\n` | 37.7× | 55.3% | 96.3% |
| "Generate a website design for a house cleaning company" | `\n` | 38.7× | 55.1% | 96.3% |
| "How many continents are there on Earth?" | `\n` | 38.5× | 55.1% | 96.2% |
| "Arrange the following musical notes" | `\n` | 38.6× | 55.5% | 96.2% |
| "Explain the concept of socio-economic privilege." | `\n` | 38.1× | 55.4% | 96.2% |
| "Classify these five animals into two groups." | `\n` | 38.3× | 55.0% | 96.3% |
| "List five benefits of going for a walk" | `\n` | 38.7× | 55.9% | 96.2% |
| "Pick the best response based on the given situation." | `\n` | 39.7× | 55.2% | 96.2% |
| "Explain the theory of relativity in simple terms." | `\n` | 38.6× | 55.1% | 96.3% |

The prompt content is completely irrelevant. The sink token is always `\n`, its position is always 2, and the spike channels are always the same 4 dimensions. This is a structural property of the model, not a semantic one.

---

## Supported architectures

| Architecture | Models |
|-------------|--------|
| `LlamaForCausalLM` | Llama 2, Llama 3, Llama 3.1 |
| `Qwen2ForCausalLM` | Qwen2, Qwen2.5 (all sizes) |
| `MistralForCausalLM` | Mistral 7B, Mixtral |
| `Phi3ForCausalLM` | Phi-3 |

Any model using pre-norm + RMSNorm will show these phenomena. Other architectures with a compatible attention interface are likely to work too.

---

## Development

```bash
pip install -e '.[dev]'
pytest tests/ -v
```

Tests use a small random GPT2-architecture model — no GPU required.

---

## Paper

```bibtex
@article{sun2026spike,
  title     = {The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks},
  author    = {Sun, Shangwen and Canziani, Alfredo and LeCun, Yann},
  journal   = {arXiv:2603.05498},
  year      = {2026}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
