# Speculative Decoding for Agent Workloads engineering report

Open [index.html](index.html) for the interactive report or
[speculative-agent-workloads-report.pdf](speculative-agent-workloads-report.pdf) for the printable edition.

The report covers the system, decoding algorithm, held-out measurements,
training curves, corrected partial sweep, judge rerun and six fixes from the
commit history. It also identifies an open issue in DPO pair construction.

This is an October 6, 2026 snapshot. The corrected 1.5B sweep, controlled
speed benchmark and GPU measurements were not available when it was built.
The report marks that evidence as pending.

The HTML embeds its styles, scripts and aggregate data and works offline.
Its download buttons export the same evidence as `data.json` and `metrics.csv`.
No transcript text, prompts, token sequences or run identifiers are exported.

Rebuild from the project root, using a Python environment with PyArrow:

```bash
python tools/build_presentation.py
python tools/build_presentation.py --publish ../../reports/speculative-agent-workloads-report.html
```

The builder reads stored artifacts without running experiments. It includes
saved sidecar rows without modifying them, retains missing/partial status,
and records the source revision and SHA-256 hashes. If the missing benchmark
files become available, it requires a narrative update before rebuilding.

The interactive edition was reviewed in Chromium at desktop and phone widths.
Checks covered all ten sections, metric and temperature controls, partial
estimates, decoder stages, calculator endpoints, keyboard interaction,
commit links, downloads, offline rendering and page overflow. The recorded
verification and screenshots are in the local `review/` folder.
