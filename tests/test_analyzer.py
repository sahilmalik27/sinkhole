import numpy as np
import pytest

from sinkhole.analyzer import analyze, find_sinks, find_spikes


def test_spike_detection():
    """Spike at token 0 should be detected."""
    n_layers, seq_len, d_model = 4, 8, 64
    hidden = np.random.randn(n_layers, seq_len, d_model).astype(np.float32)
    # Inject massive spike at token 0
    hidden[:, 0, 10] = 500.0
    hidden[:, 0, 20] = 400.0

    tokens = [f"tok{i}" for i in range(seq_len)]
    result = find_spikes(hidden, tokens, threshold=10.0)

    assert len(result.tokens) > 0
    assert result.tokens[0].position == 0
    assert result.tokens[0].score > 10.0


def test_no_spikes():
    """Uniform hidden states should produce no spikes."""
    n_layers, seq_len, d_model = 4, 8, 64
    hidden = np.ones((n_layers, seq_len, d_model), dtype=np.float32)

    tokens = [f"tok{i}" for i in range(seq_len)]
    result = find_spikes(hidden, tokens, threshold=10.0)

    assert len(result.tokens) == 0


def test_sink_detection():
    """Token 0 receiving all attention should be detected as sink."""
    n_layers, n_heads, seq_len = 2, 4, 8

    # Create attention where all tokens attend to token 0
    attn = np.zeros((n_layers, n_heads, seq_len, seq_len), dtype=np.float32)
    attn[:, :, :, 0] = 0.9
    # Distribute remaining attention
    for j in range(1, seq_len):
        attn[:, :, :, j] = 0.1 / (seq_len - 1)

    tokens = [f"tok{i}" for i in range(seq_len)]
    result = find_sinks(attn, tokens, top_k=0.5)

    assert len(result.tokens) > 0
    assert result.tokens[0].position == 0
    assert result.tokens[0].attn_mass > 0


def test_sink_heads():
    """Heads where top-2 tokens get >50% attention should be sink heads."""
    n_layers, n_heads, seq_len = 2, 4, 8

    attn = np.zeros((n_layers, n_heads, seq_len, seq_len), dtype=np.float32)
    attn[:, :, :, 0] = 0.6
    attn[:, :, :, 1] = 0.3
    for j in range(2, seq_len):
        attn[:, :, :, j] = 0.1 / (seq_len - 2)

    tokens = [f"tok{i}" for i in range(seq_len)]
    result = find_sinks(attn, tokens)

    assert len(result.sink_heads) > 0


def test_overlap():
    """Spike and sink at same position should produce overlap."""
    n_layers, n_heads, seq_len, d_model = 2, 4, 8, 64

    hidden = np.random.randn(n_layers, seq_len, d_model).astype(np.float32) * 0.1
    hidden[:, 0, 10] = 500.0

    attn = np.zeros((n_layers, n_heads, seq_len, seq_len), dtype=np.float32)
    attn[:, :, :, 0] = 0.9
    for j in range(1, seq_len):
        attn[:, :, :, j] = 0.1 / (seq_len - 1)

    tokens = [f"tok{i}" for i in range(seq_len)]
    report = analyze(hidden, attn, tokens, "test-model", "test prompt")

    assert report.overlap_jaccard > 0
    assert "tok0" in report.overlap_tokens


def test_empty_input():
    """Edge case: single token."""
    hidden = np.random.randn(2, 1, 64).astype(np.float32)
    attn = np.ones((2, 4, 1, 1), dtype=np.float32)

    tokens = ["<s>"]
    report = analyze(hidden, attn, tokens, "test", "test")

    assert report.seq_len == 1
