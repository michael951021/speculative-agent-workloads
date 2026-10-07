# Corpus to Parquet

## Issue
No labeled corpus exists: raw JSONL (50,825 blocks, 32 runs) is untokenized, unsplit, and
code isn't separated from prose. Every drafter and metric needs one Parquet corpus,
kind-labeled and run-split, or all downstream numbers are untrustworthy.

## Question
Does `python -m speculative_agent_workloads.data build` produce train/val/test Parquet where (a) each content
block gets kind ∈ {thinking, text, code, tool_use, tool_result} with code split from fenced
text, (b) no run appears in more than one split, (c) blocks <8 tokens dropped, exact dups
removed, text truncated at 4096 Qwen tokens? Refutation: a run id present in two split files,
a fenced code block left labeled `text`, or n_tokens > 4096 / < 8 in any output.

## Evidence
- Observed: data/raw/agent_turns.jsonl rows = {run, project, seq, role, kind, text}; kinds
  tool_use 15,705 / tool_result 15,702 / thinking 13,278 / text 6,140; tool_use text is
  JSON {"name","input"}; 32 runs; tokenizer at models/qwen2.5-0.5b-instruct (shared vocab).
- Check: tests run build() on a fixture jsonl with a stub tokenizer; output saved under
  .agent-evidence/corpus-build.log (full run on real data).

## Done when
- `.venv/bin/python -m pytest speculative_agent_workloads/tests/test_corpus.py -q` passes (kinds, code split,
  split disjointness on fixtures).
- Real build: 3 parquet files in data/corpus/; `python -m speculative_agent_workloads.data stats` writes
  results/corpus_stats.json with block/token counts per kind, tool, project and n_tokens
  percentiles. Exit 0.
- Regression: test_corpus.py covers the three invariants above.

## Stop when
Out of scope: sampling, token-level labels, n-gram features, UI export, GPU. If Qwen
tokenizer can't load from models/, or DuckDB queries can't express the stats, record the
blocker and stop after 2 attempts.

## Plan
1. speculative_agent_workloads/data/corpus.py: parse rows, code/text split, tool name, fake-or-real tokenizer,
   filter/dedupe/truncate, sha1(run)-ordered 80/10/10 split, write parquet.
2. speculative_agent_workloads/data/stats.sql: named DuckDB queries over the three files (kept in the source
   package — top-level data/ is gitignored; .gitignore pattern anchored to /data/).
3. speculative_agent_workloads/data/__main__-style CLI in speculative_agent_workloads/data/__init__.py: build | stats.
4. Tests on fixture; run real build; verify stats JSON.
