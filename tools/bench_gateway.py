#!/usr/bin/env python3
"""Load bench through the gateway: streamed chat completions at a fixed concurrency.

Measures per-request TTFT/latency, total completion tokens, throughput, and the
gateway /metrics delta (speculative_agent_workloads_draft_tokens_total / _accepted_tokens_total for
the model). Writes a JSON report to --out.

Usage:
  python tools/bench_gateway.py --url http://127.0.0.1:8200 \
      --key sk-speculative-agent-workloads-dev \
      --model Qwen/Qwen2.5-7B-Instruct --concurrency 4 --requests 24 --max-tokens 128 \
      --out results/gpu/bench_ngram_c4.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# Fixed agent-like prompts (one per content kind), cycled across requests.
PROMPTS = [
    "Write a Python function that reads a JSONL file of agent turns and counts "
    "tokens per message kind. Include docstring and type hints.",
    "Summarize this bug report in two sentences: the test fails only when the KV "
    "cache is padded, because the attention mask is off by one in the last row.",
    "You are a coding agent. Plan the next tool call to verify the fix, then call "
    "Bash with the pytest command.",
]


def sse_request(url: str, key: str, model: str, prompt: str, max_tokens: int) -> dict:
    """One streamed chat completion; returns ttft, latency, completion tokens."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    ttft = None
    n_tok = 0
    with requests.post(
        url + "/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {key}"},
        stream=True,
        timeout=600,
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if ttft is None:
                ttft = time.perf_counter() - t0
            choice = (chunk.get("choices") or [{}])[0]
            if (choice.get("delta") or {}).get("content"):
                n_tok += 1
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens"):
                n_tok = usage["completion_tokens"]
    return {"ttft_s": ttft, "latency_s": time.perf_counter() - t0, "completion_tokens": n_tok}


def metrics_delta(url: str, key: str, model: str, before: dict, after: dict) -> dict:
    def val(m: dict, name: str) -> float:
        return m.get(name, {}).get(model, 0.0)

    draft, acc = (
        "speculative_agent_workloads_draft_tokens_total",
        "speculative_agent_workloads_accepted_tokens_total",
    )
    proposed = val(after, draft) - val(before, draft)
    accepted = val(after, acc) - val(before, acc)
    return {
        "draft_tokens_proposed": proposed,
        "draft_tokens_accepted": accepted,
        "accept_rate": (accepted / proposed) if proposed else None,
    }


def fetch_metrics(url: str) -> dict:
    """Parse /metrics text into {metric{name="model"}: value} for our counters."""
    out: dict[str, float] = {}
    r = requests.get(url + "/metrics", timeout=30)
    if not r.ok:
        return out
    for line in r.text.splitlines():
        if line.startswith("#") or "{" not in line:
            continue
        head, rest = line.split("{", 1)
        labels, _, value = rest.rpartition(" ")
        try:
            v = float(value)
        except ValueError:
            continue
        if 'model="' in labels:
            model = labels.split('model="')[1].split('"')[0]
            out.setdefault(head, {})[model] = v
    return out


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"p50": None, "p95": None, "mean": None}
    xs = sorted(xs)
    return {
        "p50": statistics.median(xs),
        "p95": xs[max(0, int(0.95 * len(xs)) - 1)],
        "mean": statistics.fmean(xs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    ap.add_argument("--key", default="sk-speculative-agent-workloads-dev")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    before = fetch_metrics(args.url)
    t0 = time.perf_counter()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [
            ex.submit(sse_request, args.url, args.key, args.model, PROMPTS[i % 3], args.max_tokens)
            for i in range(args.requests)
        ]
        for f in futs:
            try:
                rows.append(f.result())
            except Exception as e:  # keep benching; record failure
                rows.append({"error": str(e)})
    wall = time.perf_counter() - t0
    after = fetch_metrics(args.url)

    ok = [r for r in rows if "error" not in r and r.get("ttft_s") is not None]
    n_tok = sum(r["completion_tokens"] for r in ok)
    rep = {
        "model": args.model,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "ok": len(ok),
        "errors": [r["error"] for r in rows if "error" in r],
        "wall_s": wall,
        "completion_tokens": n_tok,
        "throughput_tok_s": n_tok / wall if wall else None,
        "ttft_s": _stats([r["ttft_s"] for r in ok]),
        "latency_s": _stats([r["latency_s"] for r in ok]),
        "gateway_metrics_delta": metrics_delta(args.url, args.key, args.model, before, after),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n")
    keys = ("ok", "wall_s", "completion_tokens", "throughput_tok_s", "gateway_metrics_delta")
    print(json.dumps({k: rep[k] for k in keys}))


if __name__ == "__main__":
    main()
