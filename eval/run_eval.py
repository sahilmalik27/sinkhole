"""Main evaluation loop: run sinkhole on 1000 prompts."""
from __future__ import annotations

import gc
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sinkhole.probe import ModelProbe
from sinkhole.extractor import extract
from sinkhole.analyzer import analyze
from eval.dataset import load_prompts

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_RESULTS_PATH = RESULTS_DIR / "raw_results.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def _load_completed_ids() -> set[int]:
    """Load prompt_ids already in raw_results.jsonl for resume support."""
    completed = set()
    if RAW_RESULTS_PATH.exists():
        with open(RAW_RESULTS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    completed.add(row["prompt_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def _classify_token(text: str) -> str:
    """Classify a token as structural type."""
    text = text.strip()
    if not text:
        return "whitespace"
    if text in ("<|im_start|>", "<|im_end|>", "<s>", "</s>", "<|endoftext|>", "<bos>", "<eos>"):
        return "special"
    if text == "\n":
        return "newline"
    if all(c in ".,;:!?-—()[]{}\"'" for c in text):
        return "punctuation"
    return "content"


def _process_prompt(probe: ModelProbe, prompt: dict) -> dict:
    """Run sinkhole analysis on a single prompt and return metrics dict."""
    prompt_text = prompt["prompt_text"]

    capture = probe.run(prompt_text, max_new_tokens=1)
    hidden, attn = extract(capture)
    report = analyze(
        hidden, attn,
        token_texts=capture.token_texts,
        model_name=MODEL_NAME,
        prompt=prompt_text,
    )

    # Spike metrics
    spike_count = len(report.spikes.tokens)
    spike_score_max = max((t.score for t in report.spikes.tokens), default=0.0)
    spike_score_mean = (
        sum(t.score for t in report.spikes.tokens) / spike_count
        if spike_count > 0 else 0.0
    )
    spike_positions = [t.position for t in report.spikes.tokens]
    spike_token_texts = [t.text for t in report.spikes.tokens]
    spike_channels = []
    for t in report.spikes.tokens:
        spike_channels.extend(t.channels)
    spike_channels = list(set(spike_channels))

    # Classify spike positions
    spike_position_types = [_classify_token(t.text) for t in report.spikes.tokens]

    # Relative spike positions (0-1)
    seq_len = report.seq_len
    spike_rel_positions = [p / max(seq_len - 1, 1) for p in spike_positions]

    # Sink metrics
    sink_count = len(report.sinks.tokens)
    sink_attn_mass_max = max((t.attn_mass for t in report.sinks.tokens), default=0.0)
    sink_attn_mass_total = sum(t.attn_mass for t in report.sinks.tokens)
    sink_head_coverage = (
        max((t.head_count / t.total_heads for t in report.sinks.tokens), default=0.0)
    )
    sink_positions = [t.position for t in report.sinks.tokens]
    sink_token_texts = [t.text for t in report.sinks.tokens]

    # Overlap
    jaccard = report.overlap_jaccard
    overlap_count = len(report.overlap_tokens)

    return {
        "prompt_id": prompt["prompt_id"],
        "category": prompt["category"],
        "seq_len": seq_len,
        "spike_count": spike_count,
        "spike_score_max": spike_score_max,
        "spike_score_mean": spike_score_mean,
        "spike_positions": spike_positions,
        "spike_rel_positions": spike_rel_positions,
        "spike_token_texts": spike_token_texts,
        "spike_channels": spike_channels,
        "spike_position_types": spike_position_types,
        "sink_count": sink_count,
        "sink_attn_mass_max": sink_attn_mass_max,
        "sink_attn_mass_total": sink_attn_mass_total,
        "sink_head_coverage": sink_head_coverage,
        "sink_positions": sink_positions,
        "sink_token_texts": sink_token_texts,
        "jaccard": jaccard,
        "overlap_count": overlap_count,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading prompts...")
    prompts = load_prompts(
        model_name=MODEL_NAME,
        cache_path=str(RESULTS_DIR / "prompts_cache.json"),
    )

    completed_ids = _load_completed_ids()
    remaining = [p for p in prompts if p["prompt_id"] not in completed_ids]

    print(f"Total prompts: {len(prompts)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All prompts already processed!")
        _print_summary(len(prompts))
        return

    print(f"Loading model {MODEL_NAME}...")
    probe = ModelProbe(MODEL_NAME, device="cuda")
    print("Model loaded.")

    succeeded = len(completed_ids)
    failed = 0
    errors: list[str] = []

    with open(RAW_RESULTS_PATH, "a") as f:
        for prompt in tqdm(remaining, desc="Evaluating", unit="prompt"):
            try:
                result = _process_prompt(probe, prompt)
                f.write(json.dumps(result) + "\n")
                f.flush()
                succeeded += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                error_msg = f"OOM on prompt_id={prompt['prompt_id']} (seq_len={prompt['seq_len']})"
                errors.append(error_msg)
                tqdm.write(f"  SKIP: {error_msg}")
                failed += 1
            except Exception as e:
                error_msg = f"Error on prompt_id={prompt['prompt_id']}: {type(e).__name__}: {e}"
                errors.append(error_msg)
                tqdm.write(f"  SKIP: {error_msg}")
                failed += 1

    probe.cleanup()

    print(f"\nDone!")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed: {failed}")
    print(f"  Results saved to: {RAW_RESULTS_PATH}")

    if errors:
        error_log = RESULTS_DIR / "errors.log"
        with open(error_log, "w") as f:
            f.write("\n".join(errors))
        print(f"  Error log: {error_log}")


def _print_summary(total: int):
    completed = _load_completed_ids()
    print(f"  Completed: {len(completed)} / {total}")


if __name__ == "__main__":
    main()
