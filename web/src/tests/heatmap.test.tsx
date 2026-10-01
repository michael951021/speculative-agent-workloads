import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import Heatmap from '../pages/Heatmap';
import { realData } from './helpers';

describe('Heatmap (real data)', () => {
  const data = realData();

  it('renders every drafter x kind cell with the alpha from alpha.json', () => {
    render(<Heatmap alpha={data.alpha} theme="light" />);
    for (const [d, drafter] of Object.entries(data.alpha.drafters)) {
      for (const [k, cell] of Object.entries(drafter.by_kind)) {
        expect(screen.getByTestId(`cell-${d}-${k}`).textContent).toContain(
          cell.alpha.toFixed(3),
        );
      }
    }
  });

  it('shows the 95% CI in the hover panel on mouse enter', async () => {
    const user = userEvent.setup();
    render(<Heatmap alpha={data.alpha} theme="light" />);
    const cell = data.alpha.drafters.lora.by_kind.text;
    // pre-registered plausible range for lora on plain text
    expect(cell.alpha).toBeGreaterThan(0.5);
    expect(cell.alpha).toBeLessThan(0.8);
    await user.hover(screen.getByTestId('cell-lora-text'));
    const panel = screen.getByTestId('heat-hover');
    expect(panel).toHaveTextContent(cell.alpha.toFixed(4));
    expect(panel).toHaveTextContent(
      `95% CI ${cell.alpha_ci95[0].toFixed(3)}, ${cell.alpha_ci95[1].toFixed(3)}`,
    );
    expect(panel).toHaveTextContent(cell.n_tokens.toLocaleString('en-US'));
  });

  it('CI interval contains the point estimate for every cell', () => {
    for (const drafter of Object.values(data.alpha.drafters)) {
      for (const cell of [drafter.overall, ...Object.values(drafter.by_kind)]) {
        const [lo, hi] = cell.alpha_ci95;
        expect(lo).toBeLessThanOrEqual(cell.alpha);
        expect(cell.alpha).toBeLessThanOrEqual(hi);
        // width may be 0 (tiny n -> degenerate CI), never negative
        expect(hi - lo).toBeGreaterThanOrEqual(0);
      }
    }
  });
});
