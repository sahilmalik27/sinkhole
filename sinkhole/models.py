from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpikeToken:
    position: int
    text: str
    score: float
    channels: list[int] = field(default_factory=list)


@dataclass
class SinkToken:
    position: int
    text: str
    attn_mass: float
    head_count: int
    total_heads: int


@dataclass
class SpikeResult:
    tokens: list[SpikeToken]
    norms_per_layer: np.ndarray  # [n_layers, seq_len]
    threshold: float


@dataclass
class SinkResult:
    tokens: list[SinkToken]
    sink_heads: list[tuple[int, int]]  # (layer, head) pairs
    attn_mass_per_token: np.ndarray  # [seq_len]


@dataclass
class RawCapture:
    hidden_states: list[np.ndarray]  # per-layer: [seq_len, d_model]
    attn_weights: list[np.ndarray]  # per-layer: [n_heads, seq_len, seq_len]
    token_ids: list[int]
    token_texts: list[str]


@dataclass
class AnalysisReport:
    model_name: str
    prompt: str
    token_texts: list[str]
    spikes: SpikeResult
    sinks: SinkResult
    overlap_tokens: list[str]
    overlap_jaccard: float
    kv_waste_fraction: float
    kv_waste_attn_mass: float
    n_layers: int
    n_heads: int
    seq_len: int
