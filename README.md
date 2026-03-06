# 🕳 sinkhole

**Diagnose attention sinks and activation spikes in transformer language models.**

sinkhole loads any Llama / Qwen / Mistral-family model and tells you:

- **Which tokens are spike tokens** — tokens with extreme activation outliers (massive hidden-state norms) in specific channels
- **Which tokens are sink tokens** — tokens that absorb disproportionate attention mass across heads and layers, regardless of semantic relevance
- **Whether they overlap** — the paper's core finding: in pre-norm transformers, spike tokens and sink tokens are almost always the same tokens
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
 ┌──────┬───────────┬────────────┬──────────────────────────┐
 │  Pos │ Token     │  Score     │ Spike Channels           │
 ├──────┼───────────┼────────────┼──────────────────────────┤
 │    2 │ '\n'      │   38.6×    │ [458, 2570, 2718, 2730]  │
 └──────┴───────────┴────────────┴──────────────────────────┘

 Sink Tokens  1 found
 ┌──────┬───────────┬─────────────┬─────────────────────────┐
 │  Pos │ Token     │ Attn Mass   │ Heads Dominated          │
 ├──────┼───────────┼─────────────┼─────────────────────────┤
 │    2 │ '\n'      │   55.1%     │ 754 / 784  (96%)        │
 └──────┴───────────┴─────────────┴─────────────────────────┘

 Spike ∩ Sink  1 token overlap  (Jaccard = 1.00)
 '\n' is both a spike token and a sink token

 KV Impact  Sink tokens consume 55.1% of attention budget
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
  "prompt": "...",
  "seq_len": 40,
  "n_layers": 28,
  "n_heads": 28,
  "spikes": {
    "count": 1,
    "tokens": [{"position": 2, "text": "\n", "score": 38.55, "channels": [458, 2570, 2718, 2730]}]
  },
  "sinks": {
    "count": 1,
    "tokens": [{"position": 2, "text": "\n", "attn_mass": 0.55, "head_count": 754}]
  },
  "overlap": {"tokens": ["\n"], "jaccard": 1.0}
}
```

---

## Results: Qwen2.5-7B-Instruct

Prompt: *"Explain the theory of relativity in simple terms."* (40 tokens)

| Finding | Value |
|---------|-------|
| Spike token | `\n` at position 2 — **38.6×** above median norm |
| Spike channels | 458, 2570, 2718, 2730 |
| Sink token | `\n` at position 2 — absorbs **55.1%** of all attention |
| Heads dominated | 754 out of 784 (96%) |
| Spike ∩ Sink | **Jaccard = 1.0** (perfect overlap) |
| KV waste | ~55% of attention budget on 1 token |

This confirms the paper's central claim: in pre-norm transformers, the same tokens that exhibit massive activations also become attention sinks. The `\n` token following the system prompt acts as an implicit anchor that every attention head gravitates toward.

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
