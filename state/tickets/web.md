# Ticket: React explorer (id: web)

## Issue
Findings (alpha, sweep, replays) live in JSON/Parquet only; a reader cannot see, token by token,
which parts of a real agent turn a drafter would have guessed. Need an interactive page served at
gateway /ui.

## Question
Does `web/` build a Vite+React+TS app with four views (Overview, Heatmap, Speedup, Replay),
light/dark, that renders values matching `web/public/data/*.json` exactly — such that a rendered
value disagreeing with the JSON refutes it?

## Evidence
- Data: `web/public/data/{summary,alpha,sweep,replays}.json` (schema v1, from `speculative_agent_workloads/export_ui.py`).
- Gateway already strips `/ui` prefix to a static dir (gateway/proxy.go:55,67); helm sets
  `ui_dir: /app/web/dist`.
- Acceptance: `cd web && npm run build && npx vitest run` — tests read the REAL JSON files and
  assert rendered values. Run before implementing: fails (no package.json/tests).
- Known answers / ranges (pre-registered): best α (summary.alpha.dpo) 0.72–0.75; expected best
  speedup 2.0–2.5x (dpo 2.295 @ k=5, alpha.json); measured best speedup (sweep T=0) 1.05–1.2x
  (ngram 1.123 @ k=4); heatmap lora×text α 0.5–0.8 (NOTES: 0.66); Leviathan formula unit check
  (α=0.5, k=3 → tpc 1.875 exactly = (1−0.5⁴)/0.5; α=1 → k+1; k=1 → 1+α); replay
  accepted-span count == sum(accept) from JSON.

## Done when
1. `cd web && npm run build` exits 0 → web/dist/ with index.html + data/ copied.
2. `cd web && npx vitest run` exits 0, ≥1 test per page, all on real JSON (values cross-checked
   against the files independently in-test).
3. `make check` runs the web build (add to tools/check.sh) and passes.
4. Gateway smoke: serve ui_dir=web/dist, GET /ui/ → 200 app HTML, GET /ui/data/summary.json → 200.
5. Light/dark toggle present; charts use validated palette (validator run, both modes).

## Stop when
No npm network / vitest+jsdom incompatible with Node 24 after 2 attempts → record blocker, ship
static-only fallback. Scope OUT: no react-router (tabs are state; avoids SPA 404 under static /ui),
no new export fields (re-export belongs to the report task), no GPU.

## Plan
1. Write failing acceptance (vitest tests) + `npm run build` check; confirm they fail.
2. Scaffold Vite app (base './' so assets resolve under /ui/), types, data loader, theme.
3. Pages: Overview (KPI tiles, corpus table), Heatmap (kind×drafter, hover panel with CI),
   Speedup (Recharts: expected per drafter w/ CI band; measured vs Leviathan-expected per k),
   Replay (block + drafter pick, token spans, hover p/q).
4. Wire `npm run build` into tools/check.sh; gateway /ui smoke; run validator; commit.
