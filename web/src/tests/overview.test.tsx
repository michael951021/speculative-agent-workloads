import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import Overview from '../pages/Overview';
import { realData } from './helpers';

describe('Overview (real data)', () => {
  const data = realData();
  const bestName = data.summary.alpha.best;
  const best = data.summary.alpha[bestName];

  it('renders the best-drafter alpha KPI matching summary.json', () => {
    render(<Overview data={data} theme="light" />);
    expect(screen.getByTestId('tile-best-alpha').textContent).toContain(
      best.alpha.toFixed(3),
    );
    // corrected engine (REPORT.md): LoRA 0.712, DPO 0.713. The original 0.72-0.75
    // range was set on the off-by-one evaluator.
    expect(best.alpha).toBeGreaterThan(0.70);
    expect(best.alpha).toBeLessThan(0.73);
  });

  it('renders the expected best speedup KPI matching summary.json', () => {
    render(<Overview data={data} theme="light" />);
    let mxK = 1;
    let mx = best.speedup['1'];
    for (const [k, v] of Object.entries(best.speedup)) if (v > mx) { mx = v; mxK = Number(k); }
    expect(screen.getByTestId('tile-expected-speedup').textContent).toContain(
      `${mx.toFixed(2)}×`,
    );
    expect(screen.getByTestId('tile-expected-speedup').textContent).toContain(`k=${mxK}`);
    // with the drafter's cost counted (corrected engine), the expected best is ~1.45x
    expect(mx).toBeGreaterThan(1.3);
    expect(mx).toBeLessThan(1.6);
  });

  it('renders the measured best speedup KPI matching the head-to-head bench', () => {
    render(<Overview data={data} theme="light" />);
    const best = data.summary.bench!.best!;
    const top = Object.keys(best).sort().at(-1)!;
    const mx = best[top].speedup;
    expect(screen.getByTestId('tile-measured-speedup').textContent).toContain(
      `${mx.toFixed(2)}×`,
    );
    expect(screen.getByTestId('tile-measured-speedup').textContent).toContain(top);
    // REPORT.md: LoRA k=2 on the 1.5B target, 1.58x on CPU
    expect(mx).toBeGreaterThan(1.5);
    expect(mx).toBeLessThan(1.7);
  });

  it('renders corpus KPI and per-kind table rows from summary.json', () => {
    render(<Overview data={data} theme="light" />);
    const totalBlocks = data.summary.corpus.splits.reduce((a, s) => a + s.blocks, 0);
    const totalTokens = data.summary.corpus.splits.reduce((a, s) => a + s.tokens, 0);
    expect(screen.getByTestId('tile-corpus').textContent).toContain(
      totalTokens.toLocaleString('en-US'),
    );
    expect(screen.getByTestId('tile-corpus').textContent).toContain(
      totalBlocks.toLocaleString('en-US'),
    );
    const valText = data.summary.corpus.kinds.find(
      (k) => k.split === 'val' && k.kind === 'text',
    )!;
    expect(screen.getByTestId('kind-row-text').textContent).toContain(
      String(valText.tokens),
    );
  });

  it('renders one row per drafter with alpha matching summary.json', () => {
    render(<Overview data={data} theme="light" />);
    for (const d of ['ngram', 'tiny', 'lora', bestName]) {
      const row = data.summary.alpha[d];
      expect(screen.getByTestId(`drafter-row-${d}`).textContent).toContain(
        row.alpha.toFixed(4),
      );
    }
  });
});
