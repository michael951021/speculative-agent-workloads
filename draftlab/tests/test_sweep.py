import json

import pandas as pd

from draftlab import sweep


def _args(out, **over):
    a = {
        "drafters": ["ngram", "base"],
        "ks": [1],
        "temps": [0.0],
        "n_per_kind": 2,
        "n_gen": 8,
        "max_tokens": 64,
        "num_cpus": 4,
        "cpus_per_cell": 2,
        "out": str(out),
        "target": str(sweep.MODELS / "qwen2.5-0.5b-instruct"),
    }
    a.update(over)
    return sweep.to_argv(a)


def test_two_cell_ray_sweep(tmp_path):
    out = tmp_path / "sweep.parquet"
    res = sweep.main(_args(out))
    assert len(res["ran"]) == 3  # ngram, base + no-drafter baseline
    df = pd.read_parquet(out)
    cells = set(zip(df["drafter"], df["k"], df["temp"], strict=True))
    assert ("ngram", 1, 0.0) in cells and ("base", 1, 0.0) in cells
    assert ("none", 0, 0.0) in cells
    # Known answer: base drafter == target at T=0, greedy must always match.
    b = df[df["drafter"] == "base"]
    assert b["n_draft"].sum() > 0 and (b["n_acc"] == b["n_draft"]).all()
    # ngram can't match the target's greedy everywhere on these prompts.
    n = df[df["drafter"] == "ngram"]
    assert n["n_acc"].sum() / n["n_draft"].sum() < 1.0
    # Emitted tokens are exactly n_gen (last step may truncate the bonus).
    assert (df["n_emit"] == 8).all()
    # Resumable: second run skips every cell.
    res2 = sweep.main(_args(out))
    assert res2["ran"] == []
    assert len(pd.read_parquet(out)) == len(df)


def test_prompt_level_resume_and_sidecar_recovery(tmp_path):
    out = tmp_path / "sweep.parquet"
    res = sweep.main(_args(out, drafters=["ngram"]))
    assert set(res["ran"]) == {"ngram/1/0.0", "none/0/0.0"}  # baseline always added
    df = pd.read_parquet(out)
    key = ["drafter", "k", "temp", "i"]
    assert not df.duplicated(key).any()
    # Simulate a kill mid-cell: one ngram prompt's row is missing from the parquet.
    i = df.index[df["drafter"] == "ngram"][0]
    df.drop(index=i).to_parquet(out, index=False)
    res2 = sweep.main(_args(out, drafters=["ngram"]))
    assert res2["ran"] == ["ngram/1/0.0"]  # only the partial cell reruns
    df2 = pd.read_parquet(out)
    assert len(df2) == len(df)
    assert not df2.duplicated(key).any()
    # A killed run leaves its finished prompts in sidecars; startup folds them in,
    # and fully covered cells are skipped without running Ray.
    out2 = tmp_path / "s2.parquet"
    prompts = sweep.load_prompts("val", 2, 128)
    rows = []
    for d, k, nd in (("ngram", 1, 1), ("none", 0, 0)):
        rows += [{"drafter": d, "k": k, "temp": 0.0, "kind": p["kind"], "run": p["run"],
                  "seq": p["seq"], "i": p["i"], "prompt_len": len(p["ids"]), "n_steps": 8,
                  "n_draft": nd, "n_acc": nd, "n_emit": 8, "wall_s": 1.0, "n_gen": 8,
                  "max_tokens": 128, "target": "qwen2.5-0.5b-instruct"} for p in prompts]
        part = sweep._sidecar_path(out2, d, k, 0.0)
        part.write_text("".join(json.dumps(r) + "\n" for r in rows[-len(prompts):]))
    res3 = sweep.main(_args(out2, drafters=["ngram"]))
    assert res3["ran"] == []
    assert len(pd.read_parquet(out2)) == len(rows)
    assert not list(out2.parent.glob(out2.name + ".part-*"))


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_two_cell_ray_sweep(Path(tempfile.mkdtemp()))
    print("ok")
