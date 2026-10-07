import { useEffect, useState } from 'react';
import { loadData } from './data';
import type { AppData } from './types';
import Overview from './pages/Overview';
import Heatmap from './pages/Heatmap';
import Speedup from './pages/Speedup';
import Replay from './pages/Replay';

type Tab = 'overview' | 'heatmap' | 'speedup' | 'replay';
const TABS: { id: Tab; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'heatmap', label: 'Acceptance' },
  { id: 'speedup', label: 'Speedup' },
  { id: 'replay', label: 'Token replay' },
];

function initialTheme(): 'light' | 'dark' {
  const saved = localStorage.getItem('speculative-agent-workloads-theme');
  if (saved === 'light' || saved === 'dark') return saved;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  const [tab, setTab] = useState<Tab>('overview');
  const [theme, setTheme] = useState<'light' | 'dark'>(initialTheme);
  const [data, setData] = useState<AppData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('speculative-agent-workloads-theme', theme);
  }, [theme]);

  useEffect(() => {
    loadData().then(setData).catch((e: unknown) => setError(String(e)));
  }, []);

  return (
    <div className="app">
      <header className="header">
        <h1>speculative-agent-workloads explorer</h1>
        <p className="sub">
          speculative decoding on this machine&rsquo;s own agent traffic — how many tokens a draft
          model could have guessed, per kind of content
        </p>
        <button
          className="theme-btn"
          aria-label="Toggle dark mode"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        >
          {theme === 'dark' ? '☀ light' : '☾ dark'}
        </button>
      </header>
      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            className={tab === t.id ? 'tab active' : 'tab'}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <main className="content">
        {error && (
          <div className="error" role="alert">
            failed to load data: {error}
          </div>
        )}
        {!data && !error && <div className="loading">loading…</div>}
        {data && tab === 'overview' && <Overview data={data} theme={theme} />}
        {data && tab === 'heatmap' && <Heatmap alpha={data.alpha} theme={theme} />}
        {data && tab === 'speedup' && (
          <Speedup alpha={data.alpha} sweep={data.sweep} theme={theme} />
        )}
        {data && tab === 'replay' && <Replay replays={data.replays} />}
      </main>
      {data && (
        <footer className="footer">
          generated {data.summary.generated_at} · targets: alpha {data.alpha.target}, sweep{' '}
          {data.sweep.target} · <code>python -m speculative_agent_workloads.export_ui</code>
        </footer>
      )}
    </div>
  );
}
