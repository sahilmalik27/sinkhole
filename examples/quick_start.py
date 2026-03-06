"""Quick start example for sinkhole."""

from sinkhole import ModelProbe, analyze
from sinkhole.extractor import extract
from sinkhole.report import print_report, save_html, save_json

# Load model and run analysis
probe = ModelProbe("Qwen/Qwen2.5-7B-Instruct", device="cuda")
capture = probe.run("Explain the theory of relativity in simple terms.")
probe.cleanup()

# Extract and analyze
hidden, attn = extract(capture)
report = analyze(hidden, attn, capture.token_texts, "Qwen/Qwen2.5-7B-Instruct", capture.token_texts[0])

# Output
print_report(report)
save_html(report, attn, "results/quick_start.html")
save_json(report, "results/quick_start.json")
