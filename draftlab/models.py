"""Shared model downloads: Qwen2.5 instruct/base checkpoints + a tiny test config.

`fetch` pulls safetensors + tokenizer (+ config.json) into models/<key>/; `info` prints
params/dtype/vocab per model and asserts all three tokenizers share one vocab (what makes
cross-model speculative decoding legal). `tiny_config()` is a random-init 2-layer d=64
vocab-512 Qwen2 config for fast tests.
"""

import argparse
import json
import math
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# key -> HuggingFace repo
MODELS = {
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
}

# safetensors + tokenizer only (config.json needed to load either)
ALLOW_PATTERNS = ["*.safetensors", "tokenizer*", "vocab*", "merges*", "config.json"]


def fetch(key: str | None = None, dest: Path = MODEL_DIR) -> list[Path]:
    """Download models (all, or one by key) into dest/<key>/. Idempotent."""
    import huggingface_hub

    keys = list(MODELS) if key is None else [key]
    out = []
    for k in keys:
        d = dest / k
        d.mkdir(parents=True, exist_ok=True)
        huggingface_hub.snapshot_download(
            MODELS[k], local_dir=str(d), allow_patterns=ALLOW_PATTERNS
        )
        out.append(d)
        print(f"fetched {MODELS[k]} -> {d}")
    return out


def _model_info(d: Path) -> dict:
    from safetensors import safe_open
    from transformers import AutoTokenizer

    n_params = 0
    dtypes = set()
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for name in sf.keys():
                n_params += math.prod(sf.get_slice(name).get_shape())
                dtypes.add(sf.get_slice(name).get_dtype())
    cfg = json.loads((d / "config.json").read_text())
    tok = AutoTokenizer.from_pretrained(d)
    return {
        "n_params": n_params,
        "dtype": "/".join(sorted(dtypes)),
        "vocab_size": cfg["vocab_size"],
        "n_tokens": len(tok),
        "vocab": tok.get_vocab(),
    }


def info(dest: Path = MODEL_DIR) -> list[dict]:
    """Print params/dtype/vocab per model; assert all tokenizers share one vocab."""
    rows = []
    for k in MODELS:
        d = dest / k
        if not (d / "config.json").exists():
            raise SystemExit(f"missing {d} — run `python -m draftlab.models fetch`")
        r = _model_info(d)
        rows.append(r)
        print(f"{k:24s} params={r['n_params'] / 1e9:.3f}B dtype={r['dtype']} "
              f"vocab={r['vocab_size']} (tokenizer n={r['n_tokens']})")
    ref = rows[0]["vocab"]
    assert all(r["vocab"] == ref for r in rows[1:]), "tokenizers do not share one vocab"
    print("ok: all three tokenizers share the same vocab")
    return rows


def tiny_config():
    """2-layer, d=64, vocab-512 Qwen2 config with random init weights (fast tests)."""
    from transformers import Qwen2Config

    return Qwen2Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m draftlab.models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_fetch = sub.add_parser("fetch", help="download safetensors + tokenizer into models/")
    p_fetch.add_argument("--name", choices=sorted(MODELS), help="fetch one model only")
    sub.add_parser("info", help="print params/dtype/vocab and check shared vocab")
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        fetch(args.name)
    else:
        info()


if __name__ == "__main__":
    main()
