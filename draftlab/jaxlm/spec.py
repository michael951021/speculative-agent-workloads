"""Speculative sampling (Leviathan 2023 / Chen 2023) on the JAX Qwen2 target.

Drafter protocol: ``propose(ctx, k) -> (tokens, probs)`` — up to k draft
tokens plus a (V,) float32 draft distribution for each; ``rollback(n_kept)``
undoes draft-side state (KV cache) added for the drafted suffix beyond the
n_kept drafts that were committed. The engine scores all m drafted tokens plus
the bonus position with one target call; accepts while u < p/q, then either
resamples from max(0, p - q) at the first rejection or draws the bonus from p.
Target-cache rollback is truncating the positional cache's `len`; a
fully-accepted step decodes the bonus in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from draftlab.jaxlm import model as M


class Drafter(Protocol):
    """Proposes draft tokens for `ctx` (all committed tokens so far)."""

    def propose(self, ctx: list[int], k: int) -> tuple[list[int], list[np.ndarray]]:
        """Up to k draft tokens and a (V,) float32 draft distribution for each."""

    def rollback(self, n_kept: int) -> None:
        """Undo draft-side state for the drafted suffix beyond the n_kept committed drafts."""


class NGramDrafter:
    """Incremental prompt-lookup drafter (vLLM-style, first occurrence, no self-match).

    State is the n-gram dict, extended only from committed tokens, so
    `rollback` is a no-op. Online proposals match `NGramMatcher.proposals`
    computed offline over the same sequence.
    """

    def __init__(self, vocab_size: int, max_n: int = 4) -> None:
        self.vocab_size = vocab_size
        self.max_n = max_n
        self._d: list[dict[tuple[int, ...], int]] = [dict() for _ in range(max_n + 1)]
        self._last = 0

    def _extend(self, ctx: list[int]) -> None:
        d = self._d
        mn = self.max_n
        for j in range(self._last, len(ctx)):
            for n in range(1, mn + 1):
                if j < n:
                    break
                g = tuple(ctx[j - n : j])
                if g not in d[n]:
                    d[n][g] = ctx[j]
        self._last = len(ctx)

    def propose(self, ctx: list[int], k: int) -> tuple[list[int], list[np.ndarray]]:
        self._extend(ctx)
        toks: list[int] = []
        probs: list[np.ndarray] = []
        for _ in range(k):
            t: int | None = None
            for n in range(self.max_n, 0, -1):
                if len(ctx) >= n:
                    t = self._d[n].get(tuple(ctx[-n:]))
                    if t is not None:
                        break
            if t is None:
                break
            toks.append(t)
            q = np.zeros(self.vocab_size, np.float32)
            q[t] = 1.0
            probs.append(q)
            ctx = ctx + [t]  # drafts extend the lookup context; registered when committed
        return toks, probs

    def rollback(self, n_kept: int) -> None:
        pass


class ModelDrafter:
    """A second JAX Qwen2 proposes its own greedy continuation.

    Keeps its own KV cache, extended from committed tokens; a partial
    acceptance restores the pre-draft snapshot so the committed suffix is
    re-decoded at the next propose.
    """

    def __init__(self, dcfg: M.Qwen2Cfg, dparams: dict, max_len: int) -> None:
        self.dcfg = dcfg
        self.dparams = dparams
        self.cache = M.empty_cache(dcfg, max_len)
        self._last = 0
        self._undo: tuple[dict, int] | None = None

    def propose(self, ctx: list[int], k: int) -> tuple[list[int], list[np.ndarray]]:
        if len(ctx) > self._last:
            _, self.cache = M.decode(self.dcfg, self.dparams, self.cache, ctx[self._last :])
            self._last = len(ctx)
        self._undo = (self.cache, self._last, k)
        toks: list[int] = []
        probs: list[np.ndarray] = []
        nxt = ctx[-1]
        for _ in range(k):
            lg, self.cache = M.decode(self.dcfg, self.dparams, self.cache, [nxt])
            p = np.asarray(jax.nn.softmax(lg[-1]), np.float32)
            toks.append(int(np.argmax(p)))
            probs.append(p)
            nxt = toks[-1]
        return toks, probs

    def rollback(self, n_kept: int) -> None:
        if self._undo is None:
            return
        cache, last, m = self._undo
        self._undo = None
        if n_kept == m:  # all drafts committed; the bonus extends at next propose
            return
        self.cache, self._last = cache, last


def _categorical(rng, p) -> int:
    """Sample from a probability vector."""
    return int(jax.random.categorical(rng, jnp.log(jnp.asarray(p, jnp.float32) + 1e-30)))


@dataclass
class StepStats:
    drafted: int  # m: draft tokens scored this step
    accepted: int  # consecutive drafts accepted (bonus excluded)
    per_position: list[bool]  # accept per draft position, length m
    wall_ms: float


class SpecDecoder:
    """Speculative sampling over one jitted target model."""

    def __init__(
        self,
        cfg: M.Qwen2Cfg,
        params: dict,
        drafter: Drafter,
        max_len: int,
        k: int = 5,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.params = params
        self.drafter = drafter
        self.max_len = max_len
        self.k = k
        self.temperature = temperature
        self.rng = jax.random.PRNGKey(seed)

    def generate(self, prompt: list[int], n_tokens: int) -> tuple[list[int], list[StepStats]]:
        """Sample `n_tokens` tokens after `prompt`. Returns (tokens, per-step stats)."""
        cfg, params, drafter = self.cfg, self.params, self.drafter
        T0 = self.temperature == 0.0
        rng = self.rng
        ctx = list(prompt)
        pre_lg, cache = M.prefill(cfg, params, ctx, self.max_len)
        prev_lg = pre_lg[-1]  # dist over position L (first generated token)
        out: list[int] = []
        stats: list[StepStats] = []
        while len(out) < n_tokens:
            rem = n_tokens - len(out)
            t0 = time.perf_counter()
            step_rng = jax.random.fold_in(rng, len(out))
            k = min(self.k, rem)
            dtoks: list[int] = []
            dprobs: list[np.ndarray] = []
            if k:
                dtoks, dprobs = drafter.propose(ctx, k)
            m = len(dtoks)
            if m == 0:  # no draft: draw from the held last row, then one decode
                if T0:
                    tok = int(np.argmax(np.asarray(prev_lg)))
                else:
                    tok = _categorical(step_rng, jax.nn.softmax(prev_lg))
                lg, cache = M.decode(cfg, params, cache, [tok])
                prev_lg = lg[-1]
                emitted, A = [tok], 0
            else:
                # one target call over the m drafts: row r is the dist over the
                # token at position L+r+1, i.e. draft r+1... so draft j's dist is
                # prev (j=0) or row j-1, and row m-1 is the bonus dist
                scored, cache2 = M.decode(cfg, params, cache, dtoks)
                Pd = [prev_lg if j == 0 else scored[j - 1] for j in range(m)]
                if T0:
                    A = 0
                    while A < m and dtoks[A] == int(np.argmax(np.asarray(Pd[A]))):
                        A += 1
                    bonus = int(np.argmax(np.asarray(Pd[A] if A < m else scored[m - 1])))
                else:
                    P = [np.asarray(jax.nn.softmax(row), np.float32) for row in Pd]
                    A = 0
                    for i, q in enumerate(dprobs):
                        u = float(jax.random.uniform(step_rng))
                        step_rng = jax.random.fold_in(step_rng, i + 1)
                        if u * q[dtoks[i]] < P[i][dtoks[i]]:
                            A += 1
                        else:
                            break
                    if A == m:
                        bonus = _categorical(step_rng, jax.nn.softmax(scored[m - 1]))
                    else:
                        resid = np.maximum(0.0, P[A] - dprobs[A])
                        bonus = _categorical(step_rng, resid / resid.sum())
                emitted = dtoks[:A] + [bonus]
                E = len(emitted) if A < m else min(rem, m + 1)
                emitted = emitted[:E]
                # Roll back the target cache to the A committed drafts, then
                # commit the bonus (its key/value only exists after this decode;
                # it was not part of the scoring call).
                trunc = {"k": cache2["k"], "v": cache2["v"], "len": jnp.int32(len(ctx) + A)}
                prev2, cache_b = M.decode(cfg, params, trunc, [bonus])
                if A == m and rem >= m + 1:
                    cache, prev_lg = cache_b, prev2[-1]
                else:
                    cache = {"k": cache_b["k"], "v": cache_b["v"], "len": jnp.int32(len(ctx) + E)}
                    prev_lg = prev2[-1] if len(out) + E < n_tokens else None
            drafter.rollback(A)
            ctx.extend(emitted)
            out.extend(emitted)
            dt_ms = (time.perf_counter() - t0) * 1e3
            stats.append(StepStats(m, A, [x < A for x in range(m)], dt_ms))
        self.rng = rng
        self.last_ctx = ctx  # prompt + generated
        self.last_cache = cache
        return out, stats
