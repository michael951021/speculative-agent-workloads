"""Fixture tests for the corpus build: kinds, code split, split disjointness, filters."""

import json

import pyarrow.parquet as pq

from draftlab.data.corpus import MAX_TOKENS, MIN_TOKENS, assign_splits, build, split_code


class WordTok:
    """1 token per whitespace word."""

    def encode(self, text):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def _row(run, kind, text, seq=0, role="assistant", project="p"):
    return {"run": run, "project": project, "seq": seq, "role": role, "kind": kind, "text": text}


def _fixture(tmp_path):
    lines = [
        _row("rich", "thinking", "alpha beta gamma delta epsilon zeta eta theta iota"),
        _row(
            "rich",
            "text",
            "Intro words here and there now ok yes.\n```python\n"
            "print('hello world example sentence of code here now')\n```\n"
            "Closing prose words follow here and now too.",
            seq=1,
        ),
        _row(
            "rich",
            "tool_use",
            json.dumps({"name": "Bash", "input": {"command": "echo hello world again words"}}),
            seq=2,
        ),
        _row("rich", "tool_result", "ok fine output words here today now yes", seq=3, role="user"),
        # exact repeat of the thinking block + one too-short block (7 words)
        _row("rich", "thinking", "alpha beta gamma delta epsilon zeta eta theta iota", seq=4),
        _row("rich", "text", "only seven words here", seq=5),
        # over-long block: MAX_TOKENS + 50 words
        _row("rich", "text", " ".join(f"w{i}" for i in range(MAX_TOKENS + 50)), seq=6),
    ]
    # 9 filler runs so the 80/10/10 split is non-empty on all three sides
    for i in range(9):
        lines.append(_row(f"run{i}", "text", f"filler block words for run number {i} here now"))
    p = tmp_path / "raw.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    return p


def _read_all(out_dir):
    out = {}
    for name in ("train", "val", "test"):
        out[name] = pq.read_table(out_dir / f"{name}.parquet").to_pydict()
    return out


def _rows(tables):
    rows = []
    for t in tables.values():
        n = len(t["run"])
        rows += [{c: t[c][i] for c in t} for i in range(n)]
    return rows


def test_split_code():
    assert split_code("a b\nc d") == [("text", "a b\nc d")]
    assert split_code("x\n```py\ncode line one two\n```\ny") == [
        ("text", "x"),
        ("code", "code line one two"),
        ("text", "y"),
    ]
    # unclosed fence closes as code; tildes work
    assert split_code("```\nunfenced tail") == [("code", "unfenced tail")]
    assert split_code("~~~\ntilde body\n~~~") == [("code", "tilde body")]
    # empty segments dropped
    assert split_code("```\n\n```\nplain words only here") == [("text", "plain words only here")]


def test_build(tmp_path):
    out_dir = tmp_path / "corpus"
    counts = build(_fixture(tmp_path), out_dir, tok=WordTok())
    assert sum(counts.values()) == 16  # 7 rich blocks + 9 fillers
    tables = _read_all(out_dir)
    rows = _rows(tables)
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    # kinds: code split out of the text block; tool_use carries the tool name
    assert set(by_kind) == {"thinking", "text", "code", "tool_use", "tool_result"}
    code = by_kind["code"][0]
    assert "print('hello world example sentence of code here now')" in code["text"]
    assert "```" not in code["text"]
    assert [r["tool"] for r in by_kind["tool_use"]] == ["Bash"]
    # dedupe: the repeated thinking block survives once
    assert len(by_kind["thinking"]) == 1
    # min-length filter drops the 7-word block; long block truncated at MAX_TOKENS
    assert all(r["n_tokens"] >= MIN_TOKENS for r in rows)
    assert not any(r["text"] == "only seven words here" for r in rows)
    long_ = [r for r in rows if r["n_tokens"] == MAX_TOKENS]
    assert len(long_) == 1 and len(long_[0]["text"].split()) == MAX_TOKENS
    # no run in two splits; 10 runs -> 8/1/1
    runs = {name: set(t["run"]) for name, t in tables.items()}
    assert runs["train"].isdisjoint(runs["val"])
    assert runs["train"].isdisjoint(runs["test"])
    assert runs["val"].isdisjoint(runs["test"])
    assert sorted(map(len, runs.values()), reverse=True) == [8, 1, 1]


def test_assign_splits_deterministic():
    s1 = assign_splits([f"r{i}" for i in range(20)])
    s2 = assign_splits(list(reversed([f"r{i}" for i in range(20)])))
    assert s1 == s2
    from collections import Counter

    c = Counter(s1.values())
    assert c["train"] == 16 and c["val"] == 2 and c["test"] == 2
