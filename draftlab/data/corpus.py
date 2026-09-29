"""Corpus build: data/raw/agent_turns.jsonl -> data/corpus/{train,val,test}.parquet.

One row per content block (fenced code split out of `text` blocks). Columns:
run, project, seq, role, kind, tool, text, n_tokens (Qwen tokenizer).
kind: thinking | text | code | tool_use (tool = tool name) | tool_result.
Blocks < MIN_TOKENS dropped, exact (kind, text) repeats deduped, text truncated
at MAX_TOKENS. Splits are whole runs: runs ordered by sha1(run), first 80% train,
next 10% val, rest test — so no run ever lands in two splits.
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/agent_turns.jsonl"
OUT_DIR = ROOT / "data/corpus"
TOKENIZER_DIR = ROOT / "models/qwen2.5-0.5b-instruct"

KINDS = ("thinking", "text", "code", "tool_use", "tool_result")
MIN_TOKENS = 8
MAX_TOKENS = 4096

_FENCE = re.compile(r"^(\s*)(```+|~~~+)")


def split_code(text: str) -> list[tuple[str, str]]:
    """Split a text block into (kind, text) parts: fenced segments -> code, rest -> text.

    An unclosed fence at end of input still closes as code. Parts are stripped;
    empty parts are dropped.
    """
    parts: list[tuple[str, str]] = []
    buf: list[str] = []
    kind = "text"

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            parts.append((kind, body))
        buf.clear()

    for line in text.splitlines():
        m = _FENCE.match(line)
        if m:
            flush()
            buf.clear()
            kind = "code" if kind == "text" else "text"
        else:
            buf.append(line)
    flush()
    # drop fence markers from code parts
    out = []
    for k, body in parts:
        if k == "code":
            lines = body.splitlines()
            if lines and _FENCE.match(lines[0]):
                lines = lines[1:]
            if lines and _FENCE.match(lines[-1]):
                lines = lines[:-1]
            body = "\n".join(lines).strip()
        if body:
            out.append((k, body))
    return out


def _tool_name(text: str) -> str | None:
    if text.startswith("{"):
        try:
            name = json.loads(text).get("name")
            if isinstance(name, str):
                return name
        except json.JSONDecodeError:
            pass
    return None


def load_raw(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r.get("text"), str) or not r["text"].strip():
                continue
            rows.append(
                {
                    "run": str(r.get("run", "")),
                    "project": str(r.get("project", "")),
                    "seq": int(r.get("seq", 0)),
                    "role": str(r.get("role", "")),
                    "kind": r.get("kind", ""),
                    "text": r["text"],
                }
            )
    return rows


def expand(rows: list[dict]) -> list[dict]:
    """Split fenced code out of text blocks; set tool name on tool_use."""
    out = []
    for r in rows:
        k = r["kind"]
        if k == "text":
            pieces = split_code(r["text"])
        elif k in KINDS:
            pieces = [(k, r["text"])]
        else:
            continue
        for kind, text in pieces:
            out.append(
                {
                    **{c: r[c] for c in ("run", "project", "seq", "role")},
                    "kind": kind,
                    "tool": _tool_name(text) if kind == "tool_use" else None,
                    "text": text,
                }
            )
    return out


def assign_splits(runs: list[str]) -> dict[str, str]:
    """Map run -> split. Runs ordered by sha1(run): 80% train, 10% val, 10% test."""
    ordered = sorted(set(runs), key=lambda r: hashlib.sha1(r.encode()).hexdigest())
    n = len(ordered)
    n_train, n_val = round(n * 0.8), round(n * 0.1)
    cuts = {"train": n_train, "val": n_train + n_val, "test": n}
    splits = {}
    for i, run in enumerate(ordered):
        for name in ("train", "val", "test"):
            if i < cuts[name]:
                splits[run] = name
                break
    assert splits[ordered[0]] == "train" and splits[ordered[-1]] == "test"
    return splits


def build(raw_path: Path = RAW_PATH, out_dir: Path = OUT_DIR, tok=None) -> dict[str, int]:
    """Build the parquet splits. Returns {split: n_rows}. tok: any obj with
    encode(str)->list[int] and decode(list[int])->str (defaults to Qwen)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    rows = expand(load_raw(raw_path))
    splits = assign_splits([r["run"] for r in rows])

    seen: set[tuple[str, str]] = set()
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        text = r["text"]
        n = len(tok.encode(text))
        if n < MIN_TOKENS:
            continue
        if n > MAX_TOKENS:
            text = tok.decode(tok.encode(text)[:MAX_TOKENS])
            n = MAX_TOKENS
        key = (r["kind"], text)
        if key in seen:
            continue
        seen.add(key)
        r = {**r, "text": text, "n_tokens": n}
        by_split[splits[r["run"]]].append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rs in by_split.items():
        if not rs:
            raise RuntimeError(f"empty split: {name}")
        table = pa.table(
            {
                "run": [r["run"] for r in rs],
                "project": [r["project"] for r in rs],
                "seq": [r["seq"] for r in rs],
                "role": [r["role"] for r in rs],
                "kind": [r["kind"] for r in rs],
                "tool": [r["tool"] for r in rs],
                "text": [r["text"] for r in rs],
                "n_tokens": [r["n_tokens"] for r in rs],
            }
        )
        pq.write_table(table, out_dir / f"{name}.parquet")
        counts[name] = len(rs)
        print(f"{name}: {len(rs)} blocks -> {out_dir / (name + '.parquet')}")
    return counts


STATS_SQL = Path(__file__).resolve().parent / "stats.sql"
RESULTS_DIR = ROOT / "results"


def _named_queries(sql: str) -> dict[str, str]:
    """Split a stats.sql script into {name: query} at `-- @query NAME` markers."""
    names: dict[str, list[str]] = {}
    order: list[str] = []
    name = None
    for line in sql.splitlines():
        if line.startswith("-- @query "):
            name = line.split()[2]
            names[name] = []
            order.append(name)
        elif name and not line.startswith("--"):
            names[name].append(line)
    return {n: "\n".join(names[n]).strip().rstrip(";") for n in order if names[n]}


def stats(out_dir: Path = OUT_DIR, out_path: Path | None = None) -> dict:
    """Run data/stats.sql against the parquet splits, write results/corpus_stats.json."""
    import json

    import duckdb

    con = duckdb.connect()
    for name in ("train", "val", "test"):
        p = out_dir / f"{name}.parquet"
        if not p.exists():
            raise RuntimeError(f"missing {p}; run `python -m draftlab.data build` first")
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{p}')")
    result = {"files": {n: str(out_dir / f"{n}.parquet") for n in ("train", "val", "test")}}
    for qname, sql in _named_queries(STATS_SQL.read_text()).items():
        res = con.execute(sql)
        cols = [d[0] for d in res.description]
        result[qname] = [dict(zip(cols, row, strict=True)) for row in res.fetchall()]
    out_path = out_path or (RESULTS_DIR / "corpus_stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(f"stats -> {out_path}")
    return result
