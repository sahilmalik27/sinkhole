from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from sinkhole.analyzer import analyze
from sinkhole.extractor import extract
from sinkhole.probe import ModelProbe
from sinkhole.report import print_report, save_html, save_json


@click.group()
def cli():
    """sinkhole — attention sink and activation spike analyzer for transformer models."""
    pass


@cli.command()
@click.option("--model", required=True, help="HuggingFace model name (e.g. Qwen/Qwen2.5-7B-Instruct)")
@click.option("--prompt", required=True, help="Input prompt to analyze")
@click.option("--output", default=None, help="Output path for HTML report")
@click.option("--threshold", default=10.0, type=float, help="Spike detection threshold (default: 10.0)")
@click.option("--device", default="cuda", help="Device to run on (default: cuda)")
def analyze_cmd(model: str, prompt: str, output: str | None, threshold: float, device: str):
    """Analyze a model for attention sinks and activation spikes."""
    console = Console()

    console.print(f"[bold cyan]Loading model:[/bold cyan] {model}")
    probe = ModelProbe(model, device=device)

    console.print(f"[bold cyan]Running forward pass...[/bold cyan]")
    capture = probe.run(prompt)
    probe.cleanup()

    console.print(f"[bold cyan]Extracting tensors...[/bold cyan]")
    hidden, attn = extract(capture)

    console.print(f"[bold cyan]Analyzing...[/bold cyan]")
    report = analyze(hidden, attn, capture.token_texts, model, prompt, threshold=threshold)

    print_report(report, console)

    if output:
        output_path = Path(output)
        save_html(report, attn, output_path)
        console.print(f"[bold green]HTML report saved:[/bold green] {output_path}")

        json_path = output_path.with_suffix(".json")
        save_json(report, json_path)
        console.print(f"[bold green]JSON report saved:[/bold green] {json_path}")


if __name__ == "__main__":
    cli()
