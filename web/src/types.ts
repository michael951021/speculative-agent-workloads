// Shapes of web/public/data/*.json (schema v1, draftlab/export_ui.py).

export type CI = [number, number];

export interface SplitStats {
  split: string;
  blocks: number;
  tokens: number;
  runs: number;
}

export interface KindStats {
  split: string;
  kind: string;
  blocks: number;
  tokens: number;
}

export interface DrafterSummary {
  target: string;
  alpha: number;
  alpha_ci95: CI;
  greedy: number;
  greedy_ci95: CI;
  coverage: number;
  tokens_per_call: Record<string, number>;
  speedup: Record<string, number>;
}

export interface SweepCell {
  status: string;
  alpha?: number | null;
  tpc?: number;
  tok_s?: number;
  speedup_vs_none?: number;
  by_kind?: Record<string, { tok_s: number; speedup_vs_none: number }>;
}

export interface SummaryData {
  schema_version: number;
  generated_at: string;
  inputs: string[];
  corpus: { splits: SplitStats[]; kinds: KindStats[] };
  alpha: Record<string, DrafterSummary> & { best: string };
  sweep: {
    target: string;
    n_cells_done: number;
    n_cells_expected: number;
    cells: Record<string, SweepCell>;
  };
  replays: {
    status: string;
    n_blocks: number;
    kinds: Record<string, number>;
    drafters: string[];
    target: string;
  };
}

export interface KindAlpha {
  n_tokens: number;
  n_runs: number;
  alpha: number;
  alpha_ci95: CI;
  greedy: number;
  greedy_ci95: CI;
  greedy_corpus: number;
  greedy_corpus_ci95: CI;
  coverage: number;
  tokens_per_call: Record<string, number>;
  speedup: Record<string, number>;
  tokens_per_call_ci95: Record<string, CI>;
  speedup_ci95: Record<string, CI>;
}

export interface DrafterAlpha {
  type: string;
  n_blocks: number;
  n_tokens: number;
  overall: KindAlpha;
  by_kind: Record<string, KindAlpha>;
}

export interface AlphaData {
  schema_version: number;
  generated_at: string;
  target: string;
  split: string;
  best: string;
  drafters: Record<string, DrafterAlpha>;
}

export interface SweepData {
  schema_version: number;
  generated_at: string;
  metric: string;
  target: string;
  n_gen: number;
  max_tokens: number;
  none_tok_s: Record<string, number>;
  cells: Record<string, SweepCell>;
}

export interface ReplayToken {
  id: number;
  text: string;
}

export interface ReplayDrafter {
  // null = no proposal at this position (drafter silent)
  p: (number | null)[];
  q: (number | null)[];
  accept: boolean[];
}

export interface ReplayBlock {
  run: string;
  seq: number;
  kind: string;
  n_tokens: number;
  tokens: ReplayToken[];
  [drafter: string]: ReplayDrafter | ReplayBlock['run'] | number | ReplayToken[];
}

export interface ReplaysData {
  schema_version: number;
  generated_at: string;
  target: string;
  split: string;
  ctx_budget: number;
  accept_rule: string;
  metric: string;
  drafters: Record<string, { type: string; path?: string }>;
  blocks: ReplayBlock[];
}

export interface AppData {
  summary: SummaryData;
  alpha: AlphaData;
  sweep: SweepData;
  replays: ReplaysData;
}
