"""Speculative sampling (Leviathan 2023 / Chen 2023) on the JAX Qwen2 target.

Drafter protocol: ``propose(ctx, k) -> (tokens, probs)`` — up to k draft
tokens plus a (V,) float32 draft distribution for each; ``rollback(n_kept)``
undoes draft-side state (KV cache) added for the drafted suffix beyond the
n_kept drafts that were committed. The engine scores the pending token plus all m drafted
tokens with one target call; accepts while u < p/q, then either resamples
from max(0, p - q) at the first rejection or draws the bonus from p.
Target-cache rollback is truncating the positional cache's `len`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from speculative_agent_workloads.jaxlm import model as M


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


class NullDrafter:
    """No drafting: plain autoregressive target decoding."""

    def propose(self, ctx: list[int], k: int) -> tuple[list[int], list[np.ndarray]]:
        return [], []

    def rollback(self, n_kept: int) -> None:
        pass


class ModelDrafter:
    """A second JAX Qwen2 proposes its own continuation.

    T=0: argmax drafts with one-hot q. T>0: drafts sampled from
    softmax(logits/T) and q is exactly that distribution, so accept/reject is
    lossless (SpecDecoder sets T and the seed via `configure`). Keeps its own
    KV cache; rollback truncates it to the committed drafts (no re-decode).
    The last draft is never decoded here: on full acceptance the next propose
    decodes it together with the bonus token.
    """

    def __init__(
        self, dcfg: M.Qwen2Cfg, dparams: dict, max_len: int, temperature: float = 0.0, seed: int = 0
    ) -> None:
        self.dcfg = dcfg
        self.dparams = dparams
        self.cache = M.empty_cache(dcfg, max_len)
        self._last = 0
        self._undo: tuple[int, int] | None = None
        self.configure(temperature, seed)

    def configure(self, temperature: float, seed: int) -> None:
        self.temperature = temperature
        self._rng = np.random.default_rng(seed)

    def _pick(self, row) -> tuple[int, np.ndarray]:
        if self.temperature == 0.0:
            t = int(np.argmax(np.asarray(row)))
            q = np.zeros(row.shape[-1], np.float32)
            q[t] = 1.0
            return t, q
        q = np.asarray(jax.nn.softmax(row.astype(jnp.float32) / self.temperature), np.float64)
        q /= q.sum()
        t = int(self._rng.choice(q.size, p=q))
        return t, q.astype(np.float32)

    def propose(self, ctx: list[int], k: int) -> tuple[list[int], list[np.ndarray]]:
        # extending the cache to len(ctx) gives the dist over the FIRST draft on
        # its last row; drafts 1..k-1 each decode the previous draft
        if len(ctx) <= self._last:
            raise RuntimeError("propose called without new committed tokens")
        lg, self.cache = M.decode(self.dcfg, self.dparams, self.cache, ctx[self._last :])
        self._last = len(ctx)
        toks: list[int] = []
        probs: list[np.ndarray] = []
        for j in range(k):
            t, q = self._pick(lg[-1])
            toks.append(t)
            probs.append(q)
            if j < k - 1:
                lg, self.cache = M.decode(self.dcfg, self.dparams, self.cache, [t])
        self._undo = (self._last, len(toks))
        return toks, probs

    def rollback(self, n_kept: int) -> None:
        if self._undo is None:
            return
        last, m = self._undo
        self._undo = None
        keep = min(n_kept, m - 1)  # cached drafts that were committed
        self._last = last + keep
        self.cache = {"k": self.cache["k"], "v": self.cache["v"], "len": jnp.int32(self._last)}


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
    """Speculative sampling over one jitted target model.

    temperature > 0 samples the target's softmax(lg/T); the drafter is
    configured to draft at the same T and report its true q, so accept/reject
    + residual max(0, p - q) is lossless. One target forward per step: the
    cache holds every committed token but the newest (`pend`), and each step
    decodes [pend] + drafts, so row j scores draft j and row m is the bonus.
    """

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
        if hasattr(drafter, "configure"):
            drafter.configure(temperature, seed + 1)

    def _steps(self, prompt: list[int], n_tokens: int):
        """Core loop; yields (emitted_tokens, StepStats) per target-call step."""
        cfg, params, drafter = self.cfg, self.params, self.drafter
        T0 = self.temperature == 0.0
        T = self.temperature if not T0 else 1.0
        rng = self.rng
        ctx = list(prompt)
        _, cache = M.prefill(cfg, params, ctx, self.max_len)
        cache = {"k": cache["k"], "v": cache["v"], "len": jnp.int32(len(ctx) - 1)}
        out: list[int] = []
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
            rows, cache2 = M.decode(cfg, params, cache, [ctx[-1]] + dtoks)
            if T0:
                A = 0
                while A < m and dtoks[A] == int(np.argmax(np.asarray(rows[A]))):
                    A += 1
                bonus = int(np.argmax(np.asarray(rows[A])))
            else:
                A = 0
                for i, q in enumerate(dprobs):
                    p_i = np.asarray(jax.nn.softmax(rows[i] / T), np.float32)
                    u = float(jax.random.uniform(step_rng))
                    step_rng = jax.random.fold_in(step_rng, i + 1)
                    if u * q[dtoks[i]] < p_i[dtoks[i]]:
                        A += 1
                    else:
                        resid = np.maximum(0.0, p_i - q)
                        break
                if A == m:
                    bonus = _categorical(step_rng, jax.nn.softmax(rows[m] / T))
                elif resid.sum() > 0:
                    bonus = _categorical(step_rng, resid / resid.sum())
                else:  # p == q on the draft's support: residual is p itself
                    bonus = _categorical(step_rng, p_i)
            emitted = (dtoks[:A] + [bonus])[:rem]
            # keep [pend] + the A accepted drafts; the bonus becomes the new pend
            cache = {"k": cache2["k"], "v": cache2["v"], "len": jnp.int32(len(ctx) + A)}
            drafter.rollback(A)
            ctx.extend(emitted)
            out.extend(emitted)
            dt_ms = (time.perf_counter() - t0) * 1e3
            st = StepStats(m, A, [x < A for x in range(m)], dt_ms)
            yield emitted, st
        self.rng = rng
        self.last_ctx = ctx  # prompt + generated
        self.last_cache = cache  # holds last_ctx[:-1] (the newest token is pending)

    def generate(self, prompt: list[int], n_tokens: int) -> tuple[list[int], list[StepStats]]:
        """Sample `n_tokens` tokens after `prompt`. Returns (tokens, per-step stats)."""
        out: list[int] = []
        stats: list[StepStats] = []
        for emitted, st in self._steps(prompt, n_tokens):
            out.extend(emitted)
            stats.append(st)
        return out, stats

    def generate_stream(self, prompt: list[int], n_tokens: int):
        """Like generate, but yields (token, step_stats) as each token is emitted."""
        for emitted, st in self._steps(prompt, n_tokens):
            for tok in emitted:
                yield tok, st
