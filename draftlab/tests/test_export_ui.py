"""Schema + consistency checks for the export contract web/public/data/*.json
(schema v1). Fast tests are pure reads; the one model-based test
(test_replays_match_alpha_pipeline) re-scores the 30 stored blocks with
draftlab.eval.alpha.evaluate (the exact pipeline behind results/alpha/dpo.json)
and requires per-position agreement. Run: .venv/bin/python -m pytest
draftlab/tests/test_export_ui.py -q"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "web" / "public" / "data"
CORPUS = ROOT / "data" / "corpus"
REPLAY_TARGET = ROOT / "models" / "qwen2.5-1.5b-instruct"

# Model-based replay check reads gitignored inputs (see test_alpha_known.py style).
needs_replay_inputs = pytest.mark.skipif(
    not (CORPUS / "val.parquet").exists()
    or not (REPLAY_TARGET / "config.json").exists()
    or not (ROOT / "ckpt" / "dpo").is_dir(),
    reason="data/corpus, models/ or ckpt/dpo not fetched",
)

DRAFTERS = {"ngram", "lora", "dpo", "tiny"}
KS = {1, 2, 4, 8}
TEMPS = {0.0, 0.7}
GRID_DRAFTERS = {"ngram", "tiny", "base", "lora", "lora_dpo"}


def load(name: str):
    p = DATA / name
    if not p.exists():
        pytest.fail(f"{p} missing — run `python -m draftlab.export_ui`")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def files() -> dict:
    return {n: load(f"{n}.json") for n in ("summary", "alpha", "sweep", "replays")}


def cell_key(drafter: str, k: int, temp: float) -> str:
    return f"{drafter} k={k} T={temp}"


def expected_cells() -> set[str]:
    return {cell_key(d, k, t) for d in GRID_DRAFTERS for k in KS for t in TEMPS} | {
        "none k=0 T=0.0",
        "none k=0 T=0.7",
    }


def test_summary_schema(files):
    s = files["summary"]
    assert s["schema_version"] == 1
    assert s["generated_at"]
    assert any(p.endswith(".parquet") for p in s["inputs"])
    assert s["corpus"]["splits"] and s["corpus"]["kinds"]
    al = s["alpha"]
    assert DRAFTERS <= set(al) - {"best"} and isinstance(al["best"], str)
    best = max((d for d in al if d != "best"), key=lambda d: al[d]["alpha"])
    assert al["best"] == best  # best drafter = argmax overall alpha
    sw = s["sweep"]
    cells = sw["cells"]
    assert expected_cells() <= set(cells), "grid must list every expected cell"
    for key, c in cells.items():
        assert c["status"] in ("done", "pending"), key
        if c["status"] == "done":
            assert c["alpha"] is None or 0 <= c["alpha"] <= 1
            assert c["tpc"] >= 1.0 and c["tok_s"] > 0 and c["speedup_vs_none"] > 0
    assert sw["n_cells_done"] == sum(c["status"] == "done" for c in cells.values())
    assert sw["n_cells_expected"] == len(expected_cells())
    r = s["replays"]
    assert r["n_blocks"] == 30 and len(r["kinds"]) == 6 and sum(r["kinds"].values()) == 30


def test_alpha_schema(files):
    al = files["alpha"]
    assert DRAFTERS <= set(al["drafters"]) and al["best"] in al["drafters"]
    for d, v in al["drafters"].items():
        assert v["n_blocks"] > 0 and v["n_tokens"] > 0
        for scope, s in {**v["by_kind"], "overall": v["overall"]}.items():
            for m in ("alpha", "greedy", "greedy_corpus"):
                assert 0 <= s[m] <= 1, (d, scope, m)
                assert s[f"{m}_ci95"][0] <= s[f"{m}_ci95"][1]
            tpc = [s["tokens_per_call"][str(k)] for k in (1, 2, 4, 8)]
            # tpc non-decreasing in k
            assert all(a <= b + 1e-9 for a, b in zip(tpc, tpc[1:], strict=False)), (d, scope)


def test_sweep_schema(files):
    sw = files["sweep"]
    assert sw["none_tok_s"] and all(v > 0 for v in sw["none_tok_s"].values())
    cells = sw["cells"]
    done = {c for c, v in cells.items() if v["status"] == "done"}
    assert expected_cells() & set(cells) and done, "batch 1/2 cells must be present"
    for key, c in cells.items():
        if c["status"] != "done":
            continue
        d, k, t = key.split()
        assert d in GRID_DRAFTERS | {"none"} and int(k.split("=")[1]) in KS | {0}
        assert c["n_emit"] >= c["n_acc"] >= 0 and c["n_emit"] >= c["n_steps"]
        assert abs(c["tpc"] - c["n_emit"] / c["n_steps"]) < 1e-9
        assert abs(c["tok_s"] - c["n_emit"] / c["wall_s"]) < 1e-9
        # speedup baseline = none tok_s at the same temperature
        want = 1.0 if d == "none" else c["tok_s"] / sw["none_tok_s"][t.split("=")[1]]
        assert abs(c["speedup_vs_none"] - want) < 1e-9
        assert set(c["by_kind"]), key


def test_replays_schema(files):
    r = files["replays"]
    assert r["schema_version"] == 1 and r["target"] == "qwen2.5-1.5b-instruct"
    assert set(r["drafters"]) == {"dpo", "ngram"}
    assert len(r["blocks"]) == 30
    kinds: dict[str, int] = {}
    for b in r["blocks"]:
        assert b["run"] and b["seq"] is not None and b["kind"]
        n = len(b["tokens"])
        assert n > 0 and b["n_tokens"] == n
        for t in b["tokens"]:
            assert isinstance(t["id"], int) and isinstance(t["text"], str)
        for d in r["drafters"]:
            v = b[d]
            for arr in ("p", "q", "accept", "alpha"):
                assert len(v[arr]) == n, (d, arr)
            for i in range(n):
                p, q, a, al = v["p"][i], v["q"][i], v["accept"][i], v["alpha"][i]
                assert p is None or 0.0 <= p <= 1.0
                assert q is None or 0.0 <= q <= 1.0
                assert al is None or 0.0 <= al <= 1.0
                assert isinstance(a, bool)
                if q is None:
                    assert a is False
                if p is None:
                    assert q is None and a is False
        kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    assert len(kinds) == 6 and all(v == 5 for v in kinds.values()), kinds


def test_summary_matches_other_files(files):
    s, al, sw, r = files["summary"], files["alpha"], files["sweep"], files["replays"]
    for d, v in al["drafters"].items():
        assert abs(s["alpha"][d]["alpha"] - v["overall"]["alpha"]) < 1e-9
    for key, c in sw["cells"].items():
        if c["status"] == "done":
            sc = s["sweep"]["cells"][key]
            assert sc["status"] == "done"
            assert abs(sc["tpc"] - c["tpc"]) < 1e-9
            assert abs(sc["tok_s"] - c["tok_s"]) < 1e-9
            assert (sc["alpha"] is None) == (c["alpha"] is None), key
            if c["alpha"] is not None:
                assert abs(sc["alpha"] - c["alpha"]) < 1e-9
    from collections import Counter

    assert s["replays"]["kinds"] == dict(Counter(b["kind"] for b in r["blocks"]))
    assert s["replays"]["n_blocks"] == len(r["blocks"])


def test_replays_near_alpha_reference(files):
    """Fast sanity (pure read): per-kind mean of the stored per-position alpha
    (and mean accept = greedy) must be within ~3 SE of the reference's
    per-kind values. 5 blocks per kind => SE ~0.03-0.05, so 0.15 catches
    pipeline-level drift; the exact per-position check is
    test_replays_match_alpha_pipeline below."""
    ref = json.loads((ROOT / "results" / "alpha" / "dpo.json").read_text())
    r = files["replays"]
    tol = 0.15
    # dpo.json predates the code kind split: compare only kinds the reference has.
    kinds = {b["kind"] for b in r["blocks"]} & set(ref["by_kind"])
    assert kinds
    for kind in sorted(kinds):
        a, g = [], []
        for b in r["blocks"]:
            if b["kind"] != kind:
                continue
            v = b["dpo"]
            a.extend(x for x in v["alpha"] if x is not None)
            g.extend(v["accept"])
        assert a, kind
        assert abs(sum(a) / len(a) - ref["by_kind"][kind]["alpha"]) < tol, (
            kind,
            sum(a) / len(a),
            ref["by_kind"][kind]["alpha"],
        )
        assert abs(sum(g) / len(g) - ref["by_kind"][kind]["greedy"]) < tol, (
            kind,
            sum(g) / len(g),
            ref["by_kind"][kind]["greedy"],
        )


@needs_replay_inputs
def test_replays_match_alpha_pipeline(files):
    """Known answer (model-based): rescore the 30 stored blocks with
    draftlab.eval.alpha.evaluate — the exact pipeline that produced
    results/alpha/dpo.json (same pool, target, ctx, tokenizer, alignment) —
    and require per-position agreement of the stored alpha values and greedy
    accept flags. Blocks are matched to pool rows by (run, seq) AND identical
    token ids (also pins the tokenizer)."""
    r = files["replays"]
    best = next(d for d in r["drafters"] if d != "ngram")
    from transformers import AutoTokenizer

    from draftlab.eval.alpha import (
        TARGET_DIR,
        HFDrafter,
        JaxModel,
        NGramDrafter,
        evaluate,
        load_blocks,
    )

    tok = AutoTokenizer.from_pretrained(TARGET_DIR)

    def ids_of(t: str):
        return tok.encode(t, add_special_tokens=False)

    pool = load_blocks(r["split"], 256, 400)
    pool_by = {}
    for pb in pool:
        pool_by.setdefault((pb["run"], pb["seq"]), []).append(pb)
    blocks = []
    for b in r["blocks"]:
        ids = [t["id"] for t in b["tokens"]]
        cands = [
            pb
            for pb in pool_by.get((b["run"], b["seq"]), [])
            if ids_of(pb["text"]) == ids
        ]
        assert cands, f"replay block not in {r['split']} pool: {b['run']} seq={b['seq']}"
        blocks.append(cands[0])

    target = JaxModel.from_path(TARGET_DIR, r["ctx_budget"] + 256)
    drafters = {
        best: HFDrafter(ROOT / "ckpt" / best),
        "ngram": NGramDrafter(target.cfg.vocab_size),
    }
    for name, d in drafters.items():
        records, _ = evaluate(blocks, target, d, ids_of, r["ctx_budget"])
        for i, b in enumerate(r["blocks"]):
            _, _, a, g, _, _ = records[i]
            v = b[name]
            for j in range(len(v["alpha"])):
                stored = v["alpha"][j]
                if stored is None:
                    assert v["accept"][j] is False, (name, b["run"], b["seq"], j)
                else:
                    assert abs(float(a[j]) - stored) < 1e-5, (
                        name,
                        b["run"],
                        b["seq"],
                        j,
                        float(a[j]),
                        stored,
                    )
                if v["q"][j] is not None:
                    assert v["accept"][j] is bool(g[j]), (name, b["run"], b["seq"], j)


def test_ngram_replay_nulls_are_no_proposals(files):
    """ngram coverage on the 400 pool is ~0.76: expect a substantial fraction of
    null q's (proposals missing), not 0 or 100%."""
    r = files["replays"]
    tot = null = 0
    for b in r["blocks"]:
        q = b["ngram"]["q"]
        tot += len(q)
        null += sum(x is None for x in q)
    frac = null / tot
    assert 0.1 < frac < 0.95, frac
