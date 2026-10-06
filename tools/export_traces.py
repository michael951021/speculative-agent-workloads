"""Host-side: export the agent loop's transcripts (../../logs/*.jsonl, Claude Code stream-json)
into data/raw/agent_turns.jsonl — one record per content block the model produced or received.
Run from the host (the sandbox cannot see ../../logs):
    python3 tools/export_traces.py [--max-mb 400]
Temporal hold-out (runs started on/after a date, some projects skipped):
    python3 tools/export_traces.py --since 20260930 --skip-projects apply \
        --out data/raw/heldout_turns.jsonl
"""

import argparse
import glob
import json
import os
import re
import sys

SECRET = re.compile(r"(ghp_|gho_|github_pat_|sk-|xox[abp]-|AKIA)[A-Za-z0-9_\-]{12,}")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def blocks(line):
    ev = json.loads(line)
    t = ev.get("type")
    if t not in ("assistant", "user"):
        return
    for b in (ev.get("message") or {}).get("content") or []:
        if not isinstance(b, dict):
            continue
        k = b.get("type")
        if k == "text":
            yield t, "text", b.get("text", "")
        elif k == "thinking":
            yield t, "thinking", b.get("thinking", "")
        elif k == "tool_use":
            yield (
                t,
                "tool_use",
                json.dumps({"name": b.get("name"), "input": b.get("input")}, ensure_ascii=False),
            )
        elif k == "tool_result":
            c = b.get("content")
            if isinstance(c, list):
                c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
            yield t, "tool_result", str(c or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mb", type=float, default=400)
    ap.add_argument("--since", default="", help="only runs started on/after YYYYMMDD")
    ap.add_argument("--skip-projects", default="", help="comma-separated project names")
    ap.add_argument("--out", default="data/raw/agent_turns.jsonl")
    a = ap.parse_args()
    skip = {x for x in a.skip_projects.split(",") if x}
    out = os.path.join(ROOT, a.out)
    n, size = 0, 0
    with open(out, "w") as f:
        paths = glob.glob(os.path.join(ROOT, "../../logs/*.jsonl"))
        for path in sorted(paths, key=os.path.getmtime, reverse=True):
            run = os.path.basename(path)[:-6]
            project = run.rsplit("-", 2)[0]
            started = run.rsplit("-", 2)[-2] if run.count("-") >= 2 else ""
            if (a.since and started < a.since) or re.sub(r"-w\d+$", "", project) in skip:
                continue
            for i, line in enumerate(open(path, errors="replace")):
                try:
                    for role, kind, text in blocks(line):
                        if not text.strip():
                            continue
                        rec = json.dumps(
                            {
                                "run": run,
                                "project": project,
                                "seq": i,
                                "role": role,
                                "kind": kind,
                                "text": SECRET.sub("<REDACTED>", text),
                            },
                            ensure_ascii=False,
                        )
                        f.write(rec + "\n")
                        n += 1
                        size += len(rec)
                except (json.JSONDecodeError, AttributeError):
                    continue
            if size > a.max_mb * 1e6:
                break
    print(f"{n} blocks, {size / 1e6:.0f} MB -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
