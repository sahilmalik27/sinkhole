import pytest
import numpy as np

from sinkhole.extractor import extract
from sinkhole.models import RawCapture


def test_extract_shapes():
    """Test that extraction produces correct shapes."""
    n_layers, seq_len, d_model, n_heads = 4, 8, 64, 4

    capture = RawCapture(
        hidden_states=[np.random.randn(seq_len, d_model).astype(np.float32) for _ in range(n_layers)],
        attn_weights=[np.random.randn(n_heads, seq_len, seq_len).astype(np.float32) for _ in range(n_layers)],
        token_ids=list(range(seq_len)),
        token_texts=[f"tok{i}" for i in range(seq_len)],
    )

    hidden, attn = extract(capture)

    assert hidden.shape == (n_layers, seq_len, d_model)
    assert attn.shape == (n_layers, n_heads, seq_len, seq_len)
    assert hidden.dtype == np.float32
    assert attn.dtype == np.float32


def test_extract_dtypes():
    """Test that float64 inputs are cast to float32."""
    capture = RawCapture(
        hidden_states=[np.random.randn(4, 32).astype(np.float64)],
        attn_weights=[np.random.randn(2, 4, 4).astype(np.float64)],
        token_ids=[0, 1, 2, 3],
        token_texts=["a", "b", "c", "d"],
    )

    hidden, attn = extract(capture)
    assert hidden.dtype == np.float32
    assert attn.dtype == np.float32
