import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import Speedup from '../pages/Speedup';
import { expectedTpc } from '../theme';
import { realData } from './helpers';

describe('Speedup (real data)', () => {
  const data = realData();

  it('renders one expected-speedup line per drafter (alpha.json)', () => {
    const { container } = render(<Speedup alpha={data.alpha} sweep={data.sweep} theme="light" />);
    const chart = screen.getByTestId('speedup-chart');
    const lines = chart.querySelectorAll('.recharts-line');
    expect(lines.length).toBe(Object.keys(data.alpha.drafters).length);
    expect(container.querySelectorAll('.recharts-area').length).toBeGreaterThanOrEqual(
      Object.keys(data.alpha.drafters).length,
    );
  });

  it('renders the measured-vs-expected chart with two lines', () => {
    render(<Speedup alpha={data.alpha} sweep={data.sweep} theme="light" />);
    const chart = screen.getByTestId('measured-chart');
    expect(chart.querySelectorAll('.recharts-line').length).toBe(2);
    // known answer: expected tpc at k=1 is 1 + alpha (Leviathan closed form)
    const cell = data.sweep.cells['ngram k=1 T=0.0'];
    expect(expectedTpc(cell.alpha!, 1)).toBeCloseTo(1 + cell.alpha!, 12);
    // measured sits at/below the theoretical line (draft+verify overhead)
    expect(cell.tpc!).toBeLessThanOrEqual(expectedTpc(cell.alpha!, 1) + 1e-9);
  });

  it('renders every done sweep cell in the table with its tpc', () => {
    render(<Speedup alpha={data.alpha} sweep={data.sweep} theme="light" />);
    const done = Object.entries(data.sweep.cells).filter(([, c]) => c.status === 'done');
    // cross-export consistency: sweep.json done cells == summary count
    expect(done.length).toBe(data.summary.sweep.n_cells_done);
    for (const [name, c] of done) {
      const row = screen.getByTestId(`sweep-row-${name}`);
      if (c.tpc != null) expect(row.textContent).toContain(c.tpc.toFixed(3));
    }
    // monotone tpc in k for the best drafter (pre-registered: batch 2 result)
    const tpc = [1, 2, 4, 8].map(
      (k) => data.sweep.cells[`lora_dpo k=${k} T=0.0`].tpc,
    );
    for (let i = 1; i < tpc.length; i++) expect(tpc[i]!).toBeGreaterThan(tpc[i - 1]!);
  });
});
