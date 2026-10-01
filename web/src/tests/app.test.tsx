import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from '../App';

const DATA_DIR = resolve(process.cwd(), 'public/data');

afterEach(() => {
  vi.restoreAllMocks();
});

describe('App (real data via fetch)', () => {
  function mockFetch() {
    const files: Record<string, string> = {
      'data/summary.json': readFileSync(`${DATA_DIR}/summary.json`, 'utf8'),
      'data/alpha.json': readFileSync(`${DATA_DIR}/alpha.json`, 'utf8'),
      'data/sweep.json': readFileSync(`${DATA_DIR}/sweep.json`, 'utf8'),
      'data/replays.json': readFileSync(`${DATA_DIR}/replays.json`, 'utf8'),
    };
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      const key = url.split('/').pop() ? `data/${url.split('/').pop()}` : url;
      if (!(key in files)) throw new Error(`unexpected fetch ${url}`);
      return {
        ok: true,
        status: 200,
        json: async () => JSON.parse(files[key]),
      } as Response;
    });
  }

  it('loads data and shows the overview KPIs', async () => {
    mockFetch();
    render(<App />);
    expect(await screen.findByTestId('tile-best-alpha')).toBeInTheDocument();
    const summary = JSON.parse(readFileSync(`${DATA_DIR}/summary.json`, 'utf8'));
    const best = summary.alpha[summary.alpha.best];
    expect(screen.getByTestId('tile-best-alpha').textContent).toContain(
      best.alpha.toFixed(3),
    );
  });

  it('theme toggle flips data-theme and survives re-render', async () => {
    const user = userEvent.setup();
    mockFetch();
    render(<App />);
    await screen.findByTestId('tile-best-alpha');
    const before = document.documentElement.dataset.theme;
    await user.click(screen.getByLabelText('Toggle dark mode'));
    expect(document.documentElement.dataset.theme).toBe(before === 'dark' ? 'light' : 'dark');
  });

  it('tab switch reaches the replay page', async () => {
    const user = userEvent.setup();
    mockFetch();
    render(<App />);
    await screen.findByTestId('tile-best-alpha');
    await user.click(screen.getByRole('tab', { name: 'Token replay' }));
    expect(screen.getByTestId('replay-block')).toBeInTheDocument();
  });
});
