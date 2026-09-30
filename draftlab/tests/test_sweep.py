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


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_two_cell_ray_sweep(Path(tempfile.mkdtemp()))
    print("ok")
