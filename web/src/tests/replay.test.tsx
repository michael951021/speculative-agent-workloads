import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import Replay from '../pages/Replay';
import type { ReplayBlock } from '../types';
import { realData } from './helpers';

type Rd = { p: (number | null)[]; q: (number | null)[]; accept: boolean[] };
const rd = (b: ReplayBlock, d: string) => (b as Record<string, unknown>)[d] as Rd;

describe('Replay (real data)', () => {
  const data = realData();
  const block = data.replays.blocks[0];

  it('renders one span per token, coloured by outcome (default drafter)', () => {
    render(<Replay replays={data.replays} />);
    const d = data.replays.drafters ? Object.keys(data.replays.drafters)[0] : '';
    const r = rd(block, d);
    expect(screen.getAllByTestId(/^replay-token-/)).toHaveLength(block.n_tokens);
    expect(document.querySelectorAll('.tok.accept')).toHaveLength(
      r.accept.filter(Boolean).length,
    );
    expect(document.querySelectorAll('.tok.none')).toHaveLength(
      r.q.filter((x) => x == null).length,
    );
    expect(screen.getByTestId('replay-stats').textContent).toContain(
      `accepted ${r.accept.filter(Boolean).length}`,
    );
  });

  it('hover shows p and q matching replays.json for that token', async () => {
    const user = userEvent.setup();
    render(<Replay replays={data.replays} />);
    const d = Object.keys(data.replays.drafters)[0];
    const r = rd(block, d);
    const i = r.p.findIndex((x) => x != null);
    expect(i).toBeGreaterThanOrEqual(0);
    await user.hover(screen.getByTestId(`replay-token-${i}`));
    const panel = screen.getByTestId('replay-detail');
    expect(panel).toHaveTextContent(`p=${r.p[i]!.toFixed(4)}`);
    expect(panel).toHaveTextContent(`q=${r.q[i]!.toFixed(4)}`);
  });

  it('switching drafter re-colours tokens per that drafter (ngram has nulls)', async () => {
    const user = userEvent.setup();
    render(<Replay replays={data.replays} />);
    await user.selectOptions(screen.getByLabelText('drafter'), 'ngram');
    const r = rd(block, 'ngram');
    expect(document.querySelectorAll('.tok.none')).toHaveLength(
      r.q.filter((x) => x == null).length,
    );
    expect(document.querySelectorAll('.tok.accept')).toHaveLength(
      r.accept.filter(Boolean).length,
    );
    // accepted implies proposed; rejected = proposed - accepted
    const proposed = r.q.filter((x) => x != null).length;
    const accepted = r.accept.filter(Boolean).length;
    expect(accepted).toBeLessThanOrEqual(proposed);
    expect(document.querySelectorAll('.tok.reject')).toHaveLength(proposed - accepted);
  });
});
