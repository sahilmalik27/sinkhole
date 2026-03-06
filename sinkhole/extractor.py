from __future__ import annotations

import numpy as np

from sinkhole.models import RawCapture


def extract(capture: RawCapture) -> tuple[np.ndarray, np.ndarray]:
    """Extract clean tensors from raw capture.

    Returns:
        hidden: [n_layers, seq_len, d_model] float32
        attn:   [n_layers, n_heads, seq_len, seq_len] float32
    """
    hidden = np.stack(capture.hidden_states, axis=0)  # [n_layers, seq_len, d_model]
    attn = np.stack(capture.attn_weights, axis=0)     # [n_layers, n_heads, seq_len, seq_len]
    # Replace NaN with 0 (can occur in final layer with float16 attention)
    np.nan_to_num(attn, copy=False, nan=0.0)
    np.nan_to_num(hidden, copy=False, nan=0.0)
    return hidden.astype(np.float32), attn.astype(np.float32)
