# speculative-agent-workloads: how predictable is a coding agent?

## Question

How many of the tokens this machine's own coding agents produce could a small draft model,
or a plain n-gram lookup, have guessed, per kind of content? And what wall-clock speedup does
speculative decoding actually buy on that traffic?

## Short answer

- A LoRA drafter matches the Qwen2.5-1.5B-Instruct target on **0.7120** of the probability
  mass per position on held-out agent runs (95% CI [0.7061, 0.7187]). An n-gram lookup gets
  **0.3228** [0.3074, 0.3389]. A distilled tiny model gets **0.0114**.
- Tool calls are the most predictable content, and thinking and prose the least, for the
  n-gram and LoRA drafters.
- Measured head to head on CPU against the same 1.5B target, speculation is faster than
  plain decoding: **1.58×** with the LoRA drafter at k=2, and **1.50×** with n-grams at k=8.
  Longer LoRA drafts get slower (**1.25×** at k=4) because each draft token costs a drafter
  forward.
- DPO on top of LoRA is not distinguishable from LoRA alone.

All numbers below are recomputed from `results/` by `tools/check_report.py`, which also fails
on any number in this file it cannot trace.

## Corpus

`data/raw/agent_turns.jsonl` holds content blocks (thinking, text, code, tool calls, tool
results) from local Qwen 27B agent sessions. `python -m speculative_agent_workloads.data build`
writes `data/corpus/*.parquet`. Splits are by run, so no run leaks between them. Training uses
**45,257** blocks, **15.7**M tokens from **26** runs.

The headline evaluation uses a temporal hold-out: **14** runs recorded after the training data,
**700** blocks, **56,990** scored positions. Confidence intervals resample whole runs
(`speculative_agent_workloads/eval/alpha.py`), so tokens from one session stay together.

## Method

- **Alpha.** At each position, α = Σ_x min(p(x), q(x)) = 1 − TV(p, q), with p the target and q
  the drafter, teacher-forced on the recorded text. Greedy acceptance is argmax q = argmax p.
- **Target.** Qwen2 in JAX (`speculative_agent_workloads/jaxlm/`): grouped-query attention,
  rotary positions, RMSNorm, fixed-size KV cache. Qwen2.5-1.5B-Instruct is the main target;
  Qwen2.5-0.5B-Instruct is the fast one for the full sweep.
- **Drafters.** n-gram prompt lookup; a distilled JAX drafter (`drafters/distill.py`);
  Qwen2.5-0.5B-Instruct + LoRA (`drafters/lora.py`); the LoRA drafter + DPO on
  speculative-decoding preference pairs (`drafters/dpo.py`).
- **Engine** (`jaxlm/spec.py`). One target call scores the pending token plus k drafts.
  Acceptance is u·q(x) < p(x), with replacement sampling from normalized max(0, p − q), so
  sampling at T > 0 preserves the target distribution. T = 0 reduces to greedy matching.
- **Speed.** `tools/bench_speedup.py` runs each configuration and plain decoding in one process
  on the same held-out prompts, after a compile warm-up, interleaved over **3** repetitions so
  drift hits every configuration alike. Speedup = plain-decoding wall time / configuration
  wall time on the same prompts.

## Results

### Held-out alpha, vs Qwen2.5-1.5B-Instruct (14 runs, 56,990 positions)

| drafter | alpha | 95% CI | greedy acceptance |
|---|---|---|---|
| n-gram | 0.3228 | [0.3074, 0.3389] | 0.3586 |
| LoRA (0.5B) | 0.7120 | [0.7061, 0.7187] | 0.6965 |
| LoRA + DPO | 0.7135 | [0.7078, 0.7201] | 0.6984 |
| distilled tiny | 0.0114 | [0.0109, 0.0120] | 0.1383 |

The n-gram drafter proposes on 73.9% of positions; the model drafters propose at nearly
every position.

### Alpha by kind of content

| kind | n-gram | LoRA | LoRA + DPO | tiny |
|---|---|---|---|---|
| text | 0.1983 | 0.6276 | 0.6311 | 0.0191 |
| thinking | 0.1632 | 0.6433 | 0.6472 | 0.0162 |
| tool_result | 0.2733 | 0.7198 | 0.7206 | 0.0122 |
| tool_use:Bash | 0.4769 | 0.7644 | 0.7644 | 0.0068 |
| tool_use:Edit | 0.5691 | 0.8166 | 0.8165 | 0.0063 |
| tool_use:Read | 0.5601 | 0.7739 | 0.7744 | 0.0046 |

Figure: `results/fig/alpha_by_kind.png`.

### Measured speedup (CPU, head to head, 34 held-out prompts, 32 tokens each)

Plain decoding runs at **6.47** tok/s on the 0.5B target and **1.90** tok/s (T=0) /
**1.91** tok/s (T=0.7) on the 1.5B target.

| target | drafter | k | T | tok/s | speedup | rep range | acceptance | tokens / step |
|---|---|---|---|---|---|---|---|---|
| 0.5B | n-gram | 8 | 0 | 8.92 | **1.38×** | 1.35–1.41 | 0.1319 | 1.26 |
| 1.5B | n-gram | 8 | 0 | 2.85 | **1.50×** | 1.48–1.53 | 0.2020 | 1.36 |
| 1.5B | LoRA | 2 | 0 | 2.99 | **1.58×** | 1.52–1.65 | 0.4548 | 1.86 |
| 1.5B | LoRA | 4 | 0 | 2.38 | **1.25×** | 1.22–1.27 | 0.3092 | 2.14 |
| 1.5B | LoRA | 4 | 0.7 | 2.62 | **1.38×** | 1.36–1.41 | 0.3740 | 2.39 |

Figure: `results/fig/speedup.png`. These are CPU numbers. On a GPU the target forward is
memory-bound and the draft/verify cost ratio differs, so they say nothing about GPU speedups.

### Tokens per target call across the grid (1.5B target, 115 held-out prompts)

| drafter | k | acceptance T=0 | tokens/call T=0 | acceptance T=0.7 | tokens/call T=0.7 |
|---|---|---|---|---|---|
| n-gram | 1 | 0.5706 | 1.2324 | 0.4452 | 1.1496 |
| n-gram | 2 | 0.4594 | 1.3290 | 0.3011 | 1.1686 |
| n-gram | 4 | 0.3290 | 1.4089 | 0.2123 | 1.2194 |
| n-gram | 8 | 0.2152 | 1.4586 | 0.1434 | 1.2624 |
| LoRA | 1 | 0.6283 | 1.6091 | 0.6141 | 1.5938 |
| LoRA | 2 | 0.5083 | 1.9669 | 0.4820 | 1.9187 |
| LoRA | 4 | 0.3575 | 2.3247 | 0.3579 | 2.3276 |
| LoRA | 8 | 0.2208 | 2.5698 | 0.2301 | 2.6323 |
| LoRA + DPO | 1 | 0.6331 | 1.6133 | 0.6035 | 1.5842 |
| LoRA + DPO | 2 | 0.5138 | 1.9774 | 0.5203 | 1.9881 |
| LoRA + DPO | 4 | 0.3627 | 2.3439 | 0.3687 | 2.3681 |
| LoRA + DPO | 8 | 0.2247 | 2.5970 | 0.2052 | 2.4436 |

Acceptance here is accepted / drafted, which counts drafts after the first rejection, so it
falls with k even when tokens per call rise. Sweep wall times are not used for speed: cells
ran concurrently under Ray. Figure: `results/fig/tpc_vs_k.png`.

**Known answer.** With the target as its own drafter (0.5B grid), acceptance is 1.0000 and
tokens per call are 2.0000 / 2.9091 / 4.5714 / 8.0000 at k = 1 / 2 / 4 / 8. That is the
32-token budget divided by the number of steps the engine needs, as expected.

### Engine vs metric

The engine's measured greedy acceptance on real prompts (0.5B target, k=1) falls inside the
alpha metric's prediction for the same trajectories: LoRA 0.6263 in [0.6141, 0.6657], LoRA +
DPO 0.6310 in [0.6189, 0.6703] (`results/engine_traj_metric.json`).

### DPO vs LoRA

Held-out alpha is 0.7135 with DPO and 0.7120 without, and the 95% intervals overlap
([0.7078, 0.7201] vs [0.7061, 0.7187]). Per kind and per grid cell the two are within noise.
No measurable effect.

### Judge

A blind A/B judge (`eval/judge.py`, local model via an OpenAI-compatible endpoint, both
orders, 150 calls, 18 invalid verdicts dropped) on the 0.5B target with the DPO drafter at k=4,
T=0.7. Speculative output vs the drafter alone wins **0.655** [0.473, 0.801] over 29 verdicts.
Speculative output vs the target alone wins **0.430** [0.327, 0.540] over 79 verdicts. That
interval covers 0.5, so there is no evidence that lossless speculation changed quality, which
is what the acceptance rule guarantees.

## Corrections

The previous version of this report had five errors. The code fixes are commits with
regression tests; the DPO claim fell with the bootstrap fix.

| what | before | after | fix |
|---|---|---|---|
| Run-bootstrap CIs | n-gram α CI [0.3336, 0.3343] | [0.249, 0.362] on the 3 val runs | `0158d68`: the bootstrap drew the wrong number of replicates, so CIs were about 100× too narrow |
| Lossless sampling at T=0.7 | identity drafter: 2.3 distinct outputs per 4 seeds | 4.0 | `1e9f5d3`: drafts are sampled from softmax(logits/T) and q is that distribution |
| LoRA k=8 T=0.7 acceptance (0.5B) | 0.3042 | 0.2596 | `1e9f5d3`: the old q inflated T > 0 acceptance |
| DPO vs LoRA | "+0.0013, CIs disjoint" (0.7319 vs 0.7306, 0.7316 > 0.7308) | CIs overlap, no effect | follows from the bootstrap fix |
| Serving | generation ran past EOS, finish_reason always "stop" | stops at EOS, reports "length" at the cap | `48cf15b` |

The task note for this rewrite expected LoRA k=8 T=0.7 acceptance to fall to about 0.10. The
fixed-engine sweep measures 0.2596, and that is what is reported.

## What failed, and why

- **The distilled tiny drafter is useless** (α 0.0114). It shares no weights or embeddings
  with the target and is far below the capacity where imitation works.
- **Recorded-text alpha overstates generation acceptance.** LoRA's α is 0.7120 on recorded
  prefixes, but its k=1 acceptance while generating is 0.6283. Once generation leaves the
  recorded text, the drafter sees contexts it was not trained on.
- **Long drafts don't pay on CPU.** LoRA at k=4 emits more tokens per call (2.14) than at k=2
  (1.86), but runs slower (1.25× vs 1.58×). Each extra draft costs a 0.5B forward.

## Not done

GPU measurements (a 7B target, vLLM speculative decoding, and gateway load tests) are scripted
in `scripts/gpu_run.sh` but have not been run. Everything above is CPU.

## Reproduce

```bash
make bootstrap
python -m speculative_agent_workloads.data build
python -m speculative_agent_workloads.eval.alpha --split heldout --blocks 700 --drafter ngram
python -m speculative_agent_workloads.eval.alpha --split heldout --blocks 700 --drafter hf --draft-model ckpt/lora
python -m speculative_agent_workloads.eval.alpha --split heldout --blocks 700 --drafter hf --draft-model ckpt/dpo
python -m speculative_agent_workloads.eval.alpha --split heldout --blocks 700 --drafter hf --draft-model ckpt/tiny/export
python -m speculative_agent_workloads.sweep --split heldout
python -m speculative_agent_workloads.sweep --split heldout --target models/qwen2.5-1.5b-instruct --drafters ngram lora lora_dpo
python tools/bench_speedup.py
python tools/score_engine_trajectory.py
python -m speculative_agent_workloads.eval.judge --split heldout
python tools/make_figures.py
python tools/check_report.py
```

Other figure: `results/fig/distill_kl.png` (distillation loss).
