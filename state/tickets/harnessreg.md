# harnessreg — Diagnose harness RESULT registration

## Issue
Runs end with clean `RESULT: done` final messages, but no TODO line gets checked and no
TASKLOG.md entry appears. Consequence: finished work burns re-verify attempts (alphafix a3–a23,
distill), the next agent misreads state, and every "registration" in this project turned out to
be agent self-checking — which the loop contract forbids.

## Question
Does the host registration path ever fire at run end, and if not, at which gate (line text
match, line number, run-end signal)? Refuting observation: any run whose final message is a
bare RESULT block yet left the line unchecked proves the path doesn't run at all, not a
matching bug.

## Evidence
- Transcript scan, 501 sessions (/tmp/scan.py → .agent-evidence/harnessreg-scan.log): every run
  09-30 03:28+ (distill db45b037; alphafix 19ddc6ef…7aa7ce3f, 1086b5c4; tiny-rerun e3d21f7e,
  61d4ca0f, b0fd5866, f2542893) ends with `RESULT: done` in the last assistant message; 4
  shapes (bare, fenced, prose-prefixed).
- git log (verified this run): all 9 checked lines and all 8 TASKLOG entries live inside agent
  commits (b7c6e4f, e6895d1, a0fe7fd self-checks) — agrees with a19 forensics: harness
  registered 0 blocks project-wide.
- No parser code in sandbox: /opt/agent/hooks = 7 files, none parse RESULT; root is tmpfs,
  host runner not mounted; no host run-state files found.
- Observed fact: host registration has never fired here → total gap, not "later blocks
  skipped". Guess: host runner lacks/never runs the registration step; sandbox fix = Stop hook.
- a3: Stop hook WAS wired and fired (audit lines 07:36/07:37 for sids 5b940b1d, 9dcf5bf0)
  but verdict=none: the final assistant message is flushed to the transcript at stop time
  (both transcript mtimes == hook fire time; both transcripts contain `RESULT: done` now).
  Fix: hook polls up to 20 s (settle 4 s, HARNESSREG_* env) until the block lands or the
  file stops growing; regression test appends the block 2.5 s after hook start.

## Done when
`pytest draftlab/tests/test_register_result.py` passes: hook run against REAL transcripts
(db45b037 bare block, f2542893 prose+fenced) + fixture TODO/TASKLOG in a tmp git repo flips the
line, appends the TASKLOG entry, commits; second run no-op; `not done` and missing block no-op;
title-edited line still matched via `(id:)`; late-flushed block (appended 2.5 s after hook
start) still registers. Known answers: expected parses hand-verified from the transcripts
(RESULT=done; SUMMARY contains "30.6%" / "ebd0e83"). End-to-end: at run end, TODO line 13
flips and a `## harnessreg —` TASKLOG entry appear without any agent edit (a3's stop is the
live test; verified by the next run).

## Stop when
Host runner code is out of scope (unreachable here; Kevin fixes host side; hook is idempotent so
a later host fix double-registers nothing). If project-level Stop hooks don't load in this
build (hook file untouched at stop), report not-done with that evidence. Two attempts, no new
evidence → stop.

## Plan
1. This ticket. 2. Failing pytest driving the not-yet-existing hook. 3. Implement
   `.claude/hooks/register_result.py` + project `.claude/settings.json` Stop hook. 4. Tests +
   ruff green. 5. NOTES, commit.
