"""Load 1000 diverse prompts from public HuggingFace datasets."""
from __future__ import annotations

import json
import random
from pathlib import Path

from transformers import AutoTokenizer


SEED = 42
TARGET_TOTAL = 1000

SOURCES = [
    {
        "name": "alpaca",
        "dataset_id": "tatsu-lab/alpaca",
        "subset": None,
        "split": "train",
        "field": "instruction",
        "count": 200,
        "category": "instruction",
    },
    {
        "name": "triviaqa",
        "dataset_id": "trivia_qa",
        "subset": "unfiltered",
        "split": "train",
        "field": "question",
        "count": 200,
        "category": "factual",
    },
    {
        "name": "humaneval",
        "dataset_id": "openai/openai_humaneval",
        "subset": None,
        "split": "test",
        "field": "prompt",
        "count": 100,
        "category": "code",
    },
    {
        "name": "mmlu",
        "dataset_id": "cais/mmlu",
        "subset": "all",
        "split": "test",
        "field": "question",
        "count": 200,
        "category": "reasoning",
    },
    {
        "name": "sharegpt",
        "dataset_id": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "subset": None,
        "split": "train",
        "field": "__sharegpt_first_human__",
        "count": 300,
        "category": "conversation",
    },
]


def _extract_sharegpt_first_human(row: dict) -> str | None:
    """Extract the first human turn from a ShareGPT conversation."""
    convos = row.get("conversations") or row.get("conversation") or []
    for turn in convos:
        if turn.get("from") in ("human", "user"):
            value = turn.get("value", "").strip()
            if value:
                return value
    return None


def _load_source(source: dict, rng: random.Random) -> list[dict]:
    """Load and sample prompts from a single HuggingFace dataset source."""
    from datasets import load_dataset

    name = source["name"]
    count = source["count"]
    print(f"  Loading {name} ({source['dataset_id']})...")

    try:
        kwargs = {"trust_remote_code": True}
        if source["subset"]:
            ds = load_dataset(source["dataset_id"], source["subset"], split=source["split"], **kwargs)
        else:
            ds = load_dataset(source["dataset_id"], split=source["split"], **kwargs)
    except Exception as e:
        print(f"  SKIP {name}: {e}")
        return []

    prompts = []
    if source["field"] == "__sharegpt_first_human__":
        for row in ds:
            text = _extract_sharegpt_first_human(row)
            if text and len(text) > 10:
                prompts.append(text)
    else:
        for row in ds:
            text = str(row.get(source["field"], "")).strip()
            if text and len(text) > 10:
                prompts.append(text)

    if not prompts:
        print(f"  SKIP {name}: no valid prompts found")
        return []

    rng.shuffle(prompts)
    sampled = prompts[:count]
    print(f"  Got {len(sampled)} prompts from {name}")
    return [{"text": t, "category": source["category"]} for t in sampled]


def load_prompts(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    cache_path: str | None = None,
) -> list[dict]:
    """Load 1000 diverse prompts, apply chat template, return list of dicts.

    Each dict: {prompt_id, prompt_text, category, seq_len}
    """
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached prompts from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    rng = random.Random(SEED)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    all_prompts: list[dict] = []
    remaining_sources = list(SOURCES)

    for source in SOURCES:
        batch = _load_source(source, rng)
        all_prompts.extend(batch)

    # If we're short of TARGET_TOTAL, try to fill from sources that had extras
    if len(all_prompts) < TARGET_TOTAL:
        deficit = TARGET_TOTAL - len(all_prompts)
        print(f"  Short by {deficit} prompts, attempting to fill...")
        for source in SOURCES:
            if deficit <= 0:
                break
            extra = _load_source(source, random.Random(SEED + 1))
            # Only take ones we don't already have
            existing_texts = {p["text"] for p in all_prompts}
            for p in extra:
                if p["text"] not in existing_texts and deficit > 0:
                    all_prompts.append(p)
                    existing_texts.add(p["text"])
                    deficit -= 1

    # Truncate to TARGET_TOTAL
    all_prompts = all_prompts[:TARGET_TOTAL]

    # Apply chat template and compute seq_len
    results = []
    for i, p in enumerate(all_prompts):
        prompt_text = p["text"]
        # Apply chat template for seq_len computation
        messages = [{"role": "user", "content": prompt_text}]
        try:
            templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            token_ids = tokenizer.encode(templated, add_special_tokens=False)
            seq_len = len(token_ids)
        except Exception:
            seq_len = len(tokenizer.encode(prompt_text))

        results.append({
            "prompt_id": i,
            "prompt_text": prompt_text,
            "category": p["category"],
            "seq_len": seq_len,
        })

    print(f"Total: {len(results)} prompts loaded")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(results, f)
        print(f"Cached to {cache_path}")

    return results


if __name__ == "__main__":
    prompts = load_prompts(cache_path="eval/results/prompts_cache.json")
    cats = {}
    for p in prompts:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    print(f"Categories: {cats}")
    print(f"Seq len range: {min(p['seq_len'] for p in prompts)} - {max(p['seq_len'] for p in prompts)}")
