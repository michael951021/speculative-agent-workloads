"""Matcher tests on hand-made token sequences (no model, no data files)."""

import pytest

from draftlab.drafters.ngram import NGramMatcher


def test_basic_lookup_and_first_occurrence():
    # "1" first followed by 2, "2" first by 3, "1,2" first by 3.
    # At t=8 the suffix (8,) has no earlier occurrence (its only occurrence is
    # the current suffix itself) -> no proposal.
    ctx = [5, 1, 2, 3, 7, 1, 2, 8, 1, 2, 3]
    P = NGramMatcher(4).proposals(ctx)
    assert P == [None, None, None, None, None, None, 2, 3, None, 2, 3]
    accept = [p is not None and p == ctx[t] for t, p in enumerate(P)]
    assert accept == [False] * 6 + [True, False, False, True, True]
    assert NGramMatcher.accept_runs(accept) == [0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 1]


def test_longest_n_wins_over_shorter():
    # At t=7 the n=2 suffix (3,7) first continued with 5, while n=1 (7,)
    # first continued with 2. Longest match must win -> 5.
    ctx = [7, 2, 3, 7, 5, 3, 7, 9]
    P = NGramMatcher(2).proposals(ctx)
    assert P[4] == 2  # n=1: (7,) -> 2
    assert P[7] == 5  # n=2 beats n=1's 2


def test_first_occurrence_continuation():
    ctx = [4, 1, 4, 2, 4]
    P = NGramMatcher(1).proposals(ctx)
    # (4,) first continued with 1 (at t=3); the later 4->2 must not win.
    assert P == [None, None, None, 1, None]


def test_no_self_match():
    # (1,2) appears only as the current prefix; continuation not yet in context.
    ctx = [1, 2, 9]
    assert NGramMatcher(4).proposals(ctx) == [None, None, None]


def test_accept_runs():
    assert NGramMatcher.accept_runs([True, True, False, True]) == [2, 1, 0, 1]
    assert NGramMatcher.accept_runs([]) == []
    assert NGramMatcher.accept_runs([False, False]) == [0, 0]


@pytest.mark.parametrize("k,expected", [(1, 3 / 11), (2, 4 / 11), (3, 4 / 11), (8, 4 / 11)])
def test_mean_accepted_by_k(k, expected):
    ctx = [5, 1, 2, 3, 7, 1, 2, 8, 1, 2, 3]
    P = NGramMatcher(4).proposals(ctx)
    accept = [p is not None and p == ctx[t] for t, p in enumerate(P)]
    runs = NGramMatcher.accept_runs(accept)
    mean_l = sum(min(r, k) for r in runs) / len(runs)
    assert mean_l == pytest.approx(expected)
