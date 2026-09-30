"""LLM-as-judge lossless check.

Blind pairwise judging (both orders) of outputs generated at T=0.7 on
test-split prompts: target-only vs speculative (merged LoRA drafter) vs
drafter-alone. Judge = OpenAI-compatible endpoint at $OLLAMA_BASE_URL
(openai SDK), JSON verdicts, rubric in the system prompt. API calls capped
(--max-calls, default 150) and cached in data/judge_cache/. Writes
results/judge.json with win/tie/loss rates, Wilson 95% CIs, and the
order-bias (cross-order disagreement) rate per comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
CKPT = ROOT / "ckpt"
RESULTS = ROOT / "results"
DATA = ROOT / "data"

RUBRIC_VERSION = "v1"
SYSTEM = (
    "You are an impartial judge comparing two model responses, A and B, "
    "produced for the same prompt. Decide which response is the better "
    "continuation of the prompt on relevance, factual correctness, coherence "
    "and usefulness. Do not favor length, style or formatting. If both are "
    "equally good, answer tie. Respond with only a JSON object: "
    '{"winner": "A" or "B" or "tie", "reason": "one short sentence"}.'
)
Z95 = 1.959964


def wilson(p: int, n: int) -> list[float]:
    """95% Wilson score interval for p successes in n trials."""
    if n == 0:
        return [0.0, 0.0]
    ph = p / n
    z2 = Z95 * Z95
    denom = 1 + z2 / n
    center = (ph + z2 / (2 * n)) / denom
    half = Z95 * math.sqrt(ph * (1 - ph) / n + z2 / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def parse_verdict(content: str) -> dict | None:
    """Parse the judge's JSON verdict; None if not parseable/valid."""
    c = content.strip()
    if c.startswith("```"):
        c = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", c, flags=re.S).strip()
    try:
        d = json.loads(c)
    except ValueError:
        m = re.search(r"\{.*\}", c, re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
        except ValueError:
            return None
    w = str(d.get("winner", "")).strip().lower()
    if w not in ("a", "b", "tie"):
        return None
    return {"winner": w.upper(), "reason": str(d.get("reason", ""))[:200]}


class JudgeClient:
    """OpenAI-chat judge with a per-(pair, order) on-disk cache and a hard cap."""

    def __init__(self, base_url: str, model: str, cache_dir: str | Path,
                 max_calls: int = 150, workers: int = 3, timeout: float = 180.0):
        from openai import OpenAI

        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls
        self.workers = workers
        self.calls = 0
        self.cache_hits = 0
        self._lock = threading.Lock()
        self._client = OpenAI(base_url=base_url, api_key="sk-draftlab",
                              timeout=timeout)

    def _ask_once(self, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}],
            temperature=0, max_tokens=700)
        return resp.choices[0].message.content or ""

    def ask(self, key: str, user: str) -> dict:
        """One judged (pair, order). Returns a verdict record (cached or fresh)."""
        cfile = self.cache_dir / (key + ".json")
        if cfile.exists():
            with self._lock:
                self.cache_hits += 1
            return json.loads(cfile.read_text())
        with self._lock:
            if self.calls >= self.max_calls:
                return {"winner": None, "reason": "call cap reached", "capped": True}
            self.calls += 1
        content = self._ask_once(user)
        verdict = parse_verdict(content)
        if verdict is None:  # one retry; counts against the cap
            with self._lock:
                if self.calls < self.max_calls:
                    self.calls += 1
                    content = self._ask_once(user)
                    verdict = parse_verdict(content)
        rec = {"winner": verdict["winner"] if verdict else None,
               "reason": (verdict or {}).get("reason", content[:200]) if verdict
               else content[:200],
               "content": content[:500]}
        if verdict is None:
            rec["invalid"] = True
        cfile.write_text(json.dumps(rec, indent=1))
        return rec

    def ask_all(self, jobs: list[tuple[str, str]]) -> list[dict]:
        """(cache_key, user) pairs -> verdict records in the same order."""
        out: list[dict] = [None] * len(jobs)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self.ask, k, u): i for i, (k, u) in enumerate(jobs)}
            for f in as_completed(futs):
                out[futs[f]] = f.result()
        return out


def judge_user(prompt_text: str, a: str, b: str) -> str:
    return f"PROMPT:\n{prompt_text}\n\nRESPONSE A:\n{a}\n\nRESPONSE B:\n{b}"


def cache_key(model: str, comparison: str, run, seq, order: str,
              a: str, b: str) -> str:
    h = hashlib.sha1("|".join([model, RUBRIC_VERSION, comparison,
                               str(run), str(seq), order, a, b]).encode())
    return h.hexdigest()[:24]


def build_judge_jobs(prompts: list[dict], n_drafter: int = 25) -> list[dict]:
    """Both orders of spec-vs-target for every prompt, spec-vs-drafter_alone for
    the first n_drafter. a_mode/b_mode name the side shown to the judge as A/B."""
    jobs = []
    for p in prompts:
        jobs.append({"comparison": "spec_vs_target", "run": p["run"], "seq": p["seq"],
                     "kind": p["kind"], "order": "ab", "a_mode": "spec",
                     "b_mode": "target"})
        jobs.append({"comparison": "spec_vs_target", "run": p["run"], "seq": p["seq"],
                     "kind": p["kind"], "order": "ba", "a_mode": "target",
                     "b_mode": "spec"})
    for p in prompts[: n_drafter]:
        jobs.append({"comparison": "spec_vs_drafter_alone", "run": p["run"],
                     "seq": p["seq"], "kind": p["kind"], "order": "ab",
                     "a_mode": "spec", "b_mode": "drafter_alone"})
        jobs.append({"comparison": "spec_vs_drafter_alone", "run": p["run"],
                     "seq": p["seq"], "kind": p["kind"], "order": "ba",
                     "a_mode": "drafter_alone", "b_mode": "spec"})
    return jobs


def spec_outcome(verdict: dict, a_mode: str) -> str | None:
    """Judge winner reframed from the speculative side's point of view."""
    w = verdict.get("winner")
    if w is None or verdict.get("invalid") or verdict.get("capped"):
        return None
    if w == "TIE":
        return "tie"
    return "win" if (w == "A") == (a_mode == "spec") else "loss"


def judge_all(client: JudgeClient, jobs: list[dict], prompts: list[dict],
              outputs: dict[str, str]) -> list[dict]:
    """Judge every job. outputs is keyed 'run:seq:mode'. Returns verdict records
    aligned with jobs (adds spec_outcome)."""
    text = {f"{p['run']}:{p['seq']}": p["text"] for p in prompts}
    pairs = []
    for j in jobs:
        key = f"{j['run']}:{j['seq']}"
        a = outputs[f"{key}:{j['a_mode']}"]
        b = outputs[f"{key}:{j['b_mode']}"]
        ck = cache_key(client.model, j["comparison"], j["run"], j["seq"],
                       j["order"], a, b)
        pairs.append((ck, judge_user(text[key], a, b)))
    verdicts = client.ask_all(pairs)
    out = []
    for j, v in zip(jobs, verdicts, strict=True):
        out.append({**{k: j[k] for k in ("comparison", "run", "seq", "kind", "order")},
                    "winner": v["winner"], "invalid": bool(v.get("invalid")),
                    "capped": bool(v.get("capped")),
                    "cached": bool(v.get("cached")),
                    "spec_outcome": spec_outcome(v, j["a_mode"])})
    return out


def summarize(jobs: list[dict], verdicts: list[dict]) -> dict:
    """Win/tie/loss + Wilson CIs + cross-order disagreement, per comparison."""
    rep: dict = {}
    for comp in sorted({j["comparison"] for j in jobs}):
        vs = [v for j, v in zip(jobs, verdicts, strict=True)
              if j["comparison"] == comp]
        oc = [v["spec_outcome"] for v in vs if v["spec_outcome"] is not None]
        wins = oc.count("win")
        ties = oc.count("tie")
        losses = oc.count("loss")
        n = len(oc)
        per: dict[str, dict] = {}
        for v in vs:
            per.setdefault(f"{v['run']}:{v['seq']}", {"kind": v["kind"]})[v["order"]] = \
                v["spec_outcome"]
        both = [p for p in per.values() if "ab" in p and "ba" in p]
        dis = [p for p in both if p["ab"] is not None and p["ba"] is not None
               and p["ab"] != p["ba"]]
        rep[comp] = {
            "n_prompts": len(per),
            "n_verdicts": n,
            "wins": wins, "ties": ties, "losses": losses,
            "win_rate": wins / n if n else 0.0,
            "win_ci95": wilson(wins, n),
            "loss_rate": losses / n if n else 0.0,
            "loss_ci95": wilson(losses, n),
            "order_bias_disagreement": len(dis) / len(both) if both else 0.0,
            "per_prompt": [
                {"run_seq": rs, "kind": p["kind"],
                 "ab": p.get("ab"), "ba": p.get("ba")} for rs, p in per.items()
            ],
        }
    return rep


# ---------------------------------------------------------------- generation


class _NullDrafter:
    def propose(self, ctx, k):
        return [], []

    def rollback(self, n_kept):
        pass


def drafter_alone(dcfg, dparams, ids: list[int], n: int, max_len: int,
                  temperature: float, seed: int) -> list[int]:
    """Plain autoregressive decoding with the drafter model alone (no target)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from draftlab.jaxlm import model as M
    from draftlab.jaxlm import spec as S

    lg, cache = M.prefill(dcfg, dparams, ids, max_len)
    prev = lg[-1]
    rng = jax.random.PRNGKey(seed)
    out = []
    for _ in range(n):
        if temperature == 0:
            t = int(np.argmax(np.asarray(prev)))
        else:
            p = np.asarray(jax.nn.softmax(prev / temperature), jnp.float32)
            t = S._categorical(rng, p)
        rng = jax.random.fold_in(rng, len(out) + 1)
        out.append(t)
        lg, cache = M.decode(dcfg, dparams, cache, [t])
        prev = lg[-1]
    return out


def generate_outputs(prompts: list[dict], target_dir: Path, drafter_ckpt: Path,
                     k: int, n_gen: int, temperature: float, seed: int,
                     out_file: Path) -> dict[str, str]:
    """Generate {run:seq:mode} -> text for mode in target/spec/drafter_alone.
    Incremental + resumable via out_file; wiped if the generation meta changed."""
    import jax  # noqa: F401  (warms the runtime after env is set)
    from transformers import AutoTokenizer

    from draftlab.jaxlm import model as M
    from draftlab.jaxlm import spec as S
    from draftlab.sweep import merge_adapter

    max_len = max(len(p["ids"]) for p in prompts) + n_gen + 8
    tcfg, tparams = M.load_safetensors(target_dir)
    tok = AutoTokenizer.from_pretrained(target_dir)
    dcfg, dparams = M.load_safetensors(MODELS / "qwen2.5-0.5b")
    dparams = merge_adapter(dparams, dcfg, drafter_ckpt)

    meta = {"seed": seed, "n_gen": n_gen, "k": k, "temperature": temperature,
            "target": str(target_dir), "drafter": str(drafter_ckpt),
            "prompts": hashlib.sha1(
                "|".join(f"{p['run']}:{p['seq']}:{p['kind']}" for p in prompts)
                .encode()).hexdigest()}
    outputs: dict[str, str] = {}
    if out_file.exists():
        old = json.loads(out_file.read_text())
        if old.get("meta") == meta:
            outputs = old["outputs"]
            print(f"judge: resuming {len(outputs)} cached outputs from {out_file}")
        else:
            print(f"judge: generation meta changed; regenerating ({out_file})")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    mode_off = {"target": 0, "spec": 1, "drafter_alone": 2}
    n0 = len(outputs)
    for p in prompts:
        for mode in ("target", "spec", "drafter_alone"):
            key = f"{p['run']}:{p['seq']}:{mode}"
            if key in outputs:
                continue
            s = seed + zlib.crc32(f"{p['run']}:{p['seq']}".encode()) + mode_off[mode]
            t0 = time.perf_counter()
            if mode == "target":
                dec = S.SpecDecoder(tcfg, tparams, _NullDrafter(), max_len, k=0,
                                    temperature=temperature, seed=s)
                out, _ = dec.generate(p["ids"], n_gen)
            elif mode == "spec":
                dec = S.SpecDecoder(tcfg, tparams,
                                    S.ModelDrafter(dcfg, dparams, max_len), max_len,
                                    k=k, temperature=temperature, seed=s)
                out, _ = dec.generate(p["ids"], n_gen)
            else:
                out = drafter_alone(dcfg, dparams, p["ids"], n_gen, max_len,
                                    temperature, s)
            outputs[key] = tok.decode(out)
            dt = time.perf_counter() - t0
            out_file.write_text(json.dumps(
                {"meta": meta, "outputs": outputs}, indent=1))
            print(f"  gen [{len(outputs) - n0}/{len(prompts) * 3}] {key}: "
                  f"{len(out)} tok in {dt:.1f}s", flush=True)
    return outputs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m draftlab.eval.judge")
    ap.add_argument("--n-per-kind", type=int, default=7,
                    help="test-split prompts per kind (default -> ~49 prompts, "
                         "148 judge calls <= 150 cap)")
    ap.add_argument("--n-gen", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--target", default=str(MODELS / "qwen2.5-0.5b-instruct"))
    ap.add_argument("--drafter", default="dpo", choices=["lora", "dpo"],
                    help="best drafter; its PEFT adapter is merged into 0.5b base")
    ap.add_argument("--n-drafter", type=int, default=25,
                    help="prompts also judged spec-vs-drafter_alone")
    ap.add_argument("--judge-model", default="qwen-agent")
    ap.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", ""),
                    help="OpenAI-compatible base URL (default $OLLAMA_BASE_URL)")
    ap.add_argument("--max-calls", type=int, default=150)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "judge_cache"))
    ap.add_argument("--outputs", default=str(ROOT / "data" / "judge_outputs.json"))
    ap.add_argument("--out", default=str(RESULTS / "judge.json"))
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--judge-only", action="store_true")
    args = ap.parse_args(argv)

    os.environ.setdefault("OMP_NUM_THREADS", "16")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")

    from draftlab.sweep import load_prompts

    prompts = load_prompts("test", args.n_per_kind, args.max_tokens,
                           args.target, include_text=True)
    outputs_file = Path(args.outputs)
    meta = {}
    if not args.judge_only:
        outputs = generate_outputs(
            prompts, Path(args.target), CKPT / args.drafter, args.k, args.n_gen,
            args.temperature, args.seed, outputs_file)
        meta = {"n_prompts": len(prompts), "n_gen": args.n_gen, "k": args.k,
                "temperature": args.temperature, "seed": args.seed,
                "target": str(Path(args.target).name),
                "drafter": f"{args.drafter} adapter merged into qwen2.5-0.5b"}
    else:
        blob = json.loads(outputs_file.read_text())
        outputs, meta = blob["outputs"], blob["meta"]

    if args.generate_only:
        return

    if not args.base_url:
        raise SystemExit("--base-url or $OLLAMA_BASE_URL is required for judging")

    jobs = build_judge_jobs(prompts, args.n_drafter)
    if len(jobs) > args.max_calls:
        raise SystemExit(f"{len(jobs)} judge calls needed > cap {args.max_calls}; "
                         "lower --n-per-kind/--n-drafter")
    client = JudgeClient(args.base_url, args.judge_model, args.cache_dir,
                         max_calls=args.max_calls, workers=args.workers)
    t0 = time.perf_counter()
    verdicts = judge_all(client, jobs, prompts, outputs)
    dt = time.perf_counter() - t0
    report = {
        "meta": {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "judge_model": args.judge_model,
            "rubric_version": RUBRIC_VERSION,
            "calls_used": client.calls,
            "cache_hits": client.cache_hits,
            "invalid": sum(v["invalid"] for v in verdicts),
            "wall_s": round(dt, 1),
            **meta,
        },
        **summarize(jobs, verdicts),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    for comp, r in report.items():
        if comp == "meta":
            continue
        print(f"judge {comp}: n={r['n_verdicts']} W/T/L="
              f"{r['wins']}/{r['ties']}/{r['losses']} "
              f"win {r['win_rate']:.2f} {r['win_ci95']} "
              f"loss {r['loss_rate']:.2f} {r['loss_ci95']} "
              f"order-bias {r['order_bias_disagreement']:.2f}")
    print(f"judge: {client.calls} calls, {client.cache_hits} cache hits, "
          f"{dt:.0f}s -> {out}")


if __name__ == "__main__":
    main()
