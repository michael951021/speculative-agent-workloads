# Fix dl-check: non-zero exit (exit-code)

## Issue
Compute job `dl-check` exited non-zero (`exit-code`, exit 2) with no Python traceback — a failing command, not a crash in the training code. Blocks its dependents.

## Evidence
Exit: `exit-code`, code 2; memory cap 16.0 GB; peak 16.0 GB (systemd MemoryPeak, host-side).
Failing command: `PATH=/work/.tools/go/bin:/work/.tools/bin:/work/.venv/bin:$PATH make check`
Last console output (the UI does not show this):
== shell: shellcheck ==
== promtool ==
Checking deploy/prometheus/prometheus.yml
  SUCCESS: 1 rule files found
 SUCCESS: deploy/prometheus/prometheus.yml is valid prometheus config file syntax
Checking deploy/prometheus/rules.yml
  SUCCESS: 6 rules found
Checking deploy/prometheus/rules.yml
  SUCCESS: 6 rules found
Unit Testing:  deploy/prometheus/rules_test.yml
  SUCCESS
== ci.yml: yaml parse ==
make check: FAILURES
make: *** [Makefile:11: check] Error 1
Full log: /home/kevin/Documents/coders/loop/logs/jobs/speculative-agent-workloads-dl-check.log

## Done when
The command exits 0 on the real inputs, with a fast check that covers the fixed path.

## Stop when
After two attempts with no new evidence, record the blocker and stop — the signature then escalates to a human
card. Reproduce cheaply first (tiny config / small input), not by re-running the whole job.

## Plan
1. Reproduce the failure with the smallest/fastest invocation.
2. Fix the cause (trim memory / fix the command); add a fast check.
3. Make sure the fix reaches the job's branch so its retry picks it up.
