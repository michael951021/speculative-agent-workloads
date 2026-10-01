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
    // pre-registered plausible range for the best drafter on agent traffic
    expect(best.alpha).toBeGreaterThan(0.72);
    expect(best.alpha).toBeLessThan(0.75);
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
    // Leviathan anchors: ~100x smaller drafters give alpha 0.5-0.9 => speedup a few x, not 10x
    expect(mx).toBeGreaterThan(2.0);
    expect(mx).toBeLessThan(2.5);
  });

  it('renders the measured best speedup KPI matching sweep.json', () => {
    render(<Overview data={data} theme="light" />);
    let mx = 0;
    for (const c of Object.values(data.sweep.cells)) {
      if (c.status === 'done' && (c.speedup_vs_none ?? 0) > mx) mx = c.speedup_vs_none!;
    }
    expect(screen.getByTestId('tile-measured-speedup').textContent).toContain(
      `${mx.toFixed(2)}×`,
    );
    // CPU engine: measured speedup near/under 1x, ngram slightly above (pre-registered 1.05-1.2)
    expect(mx).toBeGreaterThan(1.05);
    expect(mx).toBeLessThan(1.2);
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
