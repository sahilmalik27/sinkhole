from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from jinja2 import Template
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sinkhole.models import AnalysisReport
from sinkhole.viz import plot_attention_heatmap, plot_sink_mass, plot_spike_norms


def print_report(report: AnalysisReport, console: Console | None = None):
    """Print rich terminal report."""
    if console is None:
        console = Console()

    # Header
    console.print()
    console.print(Panel(
        f"[bold]Model:[/bold] {report.model_name}  |  "
        f"[bold]Prompt:[/bold] {report.seq_len} tokens  |  "
        f"[bold]Layers:[/bold] {report.n_layers}  |  "
        f"[bold]Heads:[/bold] {report.n_heads}",
        title="[bold cyan]sinkhole report[/bold cyan]",
        border_style="cyan",
    ))

    # Spike tokens
    spike_table = Table(title=f"SPIKE TOKENS ({len(report.spikes.tokens)} found)", border_style="yellow")
    spike_table.add_column("Pos", style="dim")
    spike_table.add_column("Token", style="bold")
    spike_table.add_column("Score", style="yellow")
    spike_table.add_column("Channels", style="dim")

    for t in report.spikes.tokens[:10]:
        channels_str = str(t.channels[:5])
        if len(t.channels) > 5:
            channels_str = channels_str[:-1] + f", ... +{len(t.channels)-5} more]"
        spike_table.add_row(str(t.position), repr(t.text), f"{t.score:.1f}x", channels_str)
    console.print(spike_table)

    # Sink tokens
    sink_table = Table(title=f"SINK TOKENS ({len(report.sinks.tokens)} found)", border_style="magenta")
    sink_table.add_column("Pos", style="dim")
    sink_table.add_column("Token", style="bold")
    sink_table.add_column("Attn Mass", style="magenta")
    sink_table.add_column("Heads", style="dim")

    for t in report.sinks.tokens[:10]:
        sink_table.add_row(
            str(t.position),
            repr(t.text),
            f"{t.attn_mass:.4f}",
            f"{t.head_count}/{t.total_heads}",
        )
    console.print(sink_table)

    # Overlap & KV waste
    console.print()
    if report.overlap_tokens:
        console.print(f"  [bold green]OVERLAP:[/bold green] {len(report.overlap_tokens)} tokens are both spike and sink: {report.overlap_tokens}")
    else:
        console.print("  [bold green]OVERLAP:[/bold green] No overlap between spike and sink tokens")
    console.print(f"  [bold green]Jaccard:[/bold green] {report.overlap_jaccard:.2%}")
    console.print(f"  [bold red]KV WASTE:[/bold red] {len(report.sinks.tokens)} sink tokens absorb {report.kv_waste_attn_mass:.1%} of attention mass")
    console.print(f"  [bold red]KV BUDGET:[/bold red] {report.kv_waste_fraction:.1%} of KV cache spent on sink tokens")
    console.print()


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>sinkhole report — {{ model_name }}</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }
h1 { color: #58a6ff; }
h2 { color: #79c0ff; border-bottom: 1px solid #21262d; padding-bottom: 8px; }
.meta { background: #161b22; padding: 16px; border-radius: 8px; margin-bottom: 20px; }
.meta span { margin-right: 24px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #21262d; }
th { background: #161b22; color: #58a6ff; }
tr:hover { background: #161b22; }
.score { color: #f0883e; font-weight: bold; }
.mass { color: #bc8cff; font-weight: bold; }
.stat { background: #161b22; padding: 12px 20px; border-radius: 8px; margin: 8px 0; display: inline-block; margin-right: 12px; }
.stat-label { color: #8b949e; font-size: 0.9em; }
.stat-value { font-size: 1.3em; font-weight: bold; }
.overlap { color: #3fb950; }
.waste { color: #f85149; }
img { max-width: 100%; border-radius: 8px; margin: 12px 0; }
</style>
</head>
<body>
<h1>sinkhole report</h1>
<div class="meta">
    <span><strong>Model:</strong> {{ model_name }}</span>
    <span><strong>Tokens:</strong> {{ seq_len }}</span>
    <span><strong>Layers:</strong> {{ n_layers }}</span>
    <span><strong>Heads:</strong> {{ n_heads }}</span>
</div>
<div class="meta">
    <strong>Prompt:</strong> {{ prompt }}
</div>

<h2>Spike Tokens ({{ spikes|length }} found)</h2>
<table>
<tr><th>Pos</th><th>Token</th><th>Score</th><th>Channels</th></tr>
{% for t in spikes %}
<tr><td>{{ t.position }}</td><td><code>{{ t.text }}</code></td><td class="score">{{ "%.1f"|format(t.score) }}x</td><td>{{ t.channels[:8] }}</td></tr>
{% endfor %}
</table>

<h2>Sink Tokens ({{ sinks|length }} found)</h2>
<table>
<tr><th>Pos</th><th>Token</th><th>Attn Mass</th><th>Heads</th></tr>
{% for t in sinks %}
<tr><td>{{ t.position }}</td><td><code>{{ t.text }}</code></td><td class="mass">{{ "%.4f"|format(t.attn_mass) }}</td><td>{{ t.head_count }}/{{ t.total_heads }}</td></tr>
{% endfor %}
</table>

<div>
    <div class="stat"><span class="stat-label">Overlap (Jaccard)</span><br><span class="stat-value overlap">{{ "%.1f"|format(overlap_jaccard * 100) }}%</span></div>
    <div class="stat"><span class="stat-label">Overlap Tokens</span><br><span class="stat-value overlap">{{ overlap_tokens }}</span></div>
    <div class="stat"><span class="stat-label">KV Waste (attn mass)</span><br><span class="stat-value waste">{{ "%.1f"|format(kv_waste_attn_mass * 100) }}%</span></div>
    <div class="stat"><span class="stat-label">KV Budget on Sinks</span><br><span class="stat-value waste">{{ "%.1f"|format(kv_waste_fraction * 100) }}%</span></div>
</div>

<h2>Attention to Sink Tokens (per layer/head)</h2>
<img src="data:image/png;base64,{{ attn_heatmap }}" alt="Attention heatmap">

<h2>Activation Norms per Token</h2>
<img src="data:image/png;base64,{{ spike_norms }}" alt="Spike norms">

<h2>Attention Mass per Token</h2>
<img src="data:image/png;base64,{{ sink_mass }}" alt="Sink mass">

</body>
</html>
"""


def save_html(report: AnalysisReport, attn: np.ndarray, path: str | Path):
    """Generate and save HTML report with embedded visualizations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    attn_heatmap = plot_attention_heatmap(report, attn)
    spike_norms = plot_spike_norms(report)
    sink_mass = plot_sink_mass(report)

    template = Template(HTML_TEMPLATE)
    html = template.render(
        model_name=report.model_name,
        prompt=report.prompt,
        seq_len=report.seq_len,
        n_layers=report.n_layers,
        n_heads=report.n_heads,
        spikes=[{
            "position": t.position,
            "text": t.text,
            "score": t.score,
            "channels": t.channels,
        } for t in report.spikes.tokens],
        sinks=[{
            "position": t.position,
            "text": t.text,
            "attn_mass": t.attn_mass,
            "head_count": t.head_count,
            "total_heads": t.total_heads,
        } for t in report.sinks.tokens],
        overlap_tokens=report.overlap_tokens,
        overlap_jaccard=report.overlap_jaccard,
        kv_waste_fraction=report.kv_waste_fraction,
        kv_waste_attn_mass=report.kv_waste_attn_mass,
        attn_heatmap=attn_heatmap,
        spike_norms=spike_norms,
        sink_mass=sink_mass,
    )
    path.write_text(html)


def save_json(report: AnalysisReport, path: str | Path):
    """Export report as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model_name": report.model_name,
        "prompt": report.prompt,
        "seq_len": report.seq_len,
        "n_layers": report.n_layers,
        "n_heads": report.n_heads,
        "token_texts": report.token_texts,
        "spikes": {
            "threshold": report.spikes.threshold,
            "count": len(report.spikes.tokens),
            "tokens": [{
                "position": t.position,
                "text": t.text,
                "score": t.score,
                "channels": t.channels,
            } for t in report.spikes.tokens],
        },
        "sinks": {
            "count": len(report.sinks.tokens),
            "sink_heads_count": len(report.sinks.sink_heads),
            "tokens": [{
                "position": t.position,
                "text": t.text,
                "attn_mass": t.attn_mass,
                "head_count": t.head_count,
                "total_heads": t.total_heads,
            } for t in report.sinks.tokens],
        },
        "overlap": {
            "tokens": report.overlap_tokens,
            "jaccard": report.overlap_jaccard,
        },
        "kv_waste": {
            "fraction": report.kv_waste_fraction,
            "attn_mass": report.kv_waste_attn_mass,
        },
    }
    path.write_text(json.dumps(data, indent=2))
