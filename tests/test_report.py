import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sinkhole.analyzer import analyze
from sinkhole.report import save_html, save_json


def _make_report():
    n_layers, n_heads, seq_len, d_model = 2, 4, 8, 64

    hidden = np.random.randn(n_layers, seq_len, d_model).astype(np.float32) * 0.1
    hidden[:, 0, 10] = 500.0

    attn = np.zeros((n_layers, n_heads, seq_len, seq_len), dtype=np.float32)
    attn[:, :, :, 0] = 0.8
    for j in range(1, seq_len):
        attn[:, :, :, j] = 0.2 / (seq_len - 1)

    tokens = [f"tok{i}" for i in range(seq_len)]
    report = analyze(hidden, attn, tokens, "test-model", "test prompt")
    return report, attn


def test_json_export():
    report, attn = _make_report()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "report.json"
        save_json(report, path)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["model_name"] == "test-model"
        assert data["spikes"]["count"] > 0
        assert data["sinks"]["count"] > 0
        assert "jaccard" in data["overlap"]


def test_html_export():
    report, attn = _make_report()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "report.html"
        save_html(report, attn, path)

        assert path.exists()
        html = path.read_text()
        assert "sinkhole report" in html
        assert "test-model" in html
        assert "data:image/png;base64," in html
