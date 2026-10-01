import { Bar, BarChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from 'recharts';
import type { AppData } from '../types';
import { DRAFTER_ORDER, SERIES, fmt } from '../theme';

interface Props {
  data: AppData;
  theme: 'light' | 'dark';
}

export default function Overview({ data, theme }: Props) {
  const { summary, alpha, sweep } = data;
  const best = summary.alpha[summary.alpha.best];
  if (!best) return <div className="error">summary.alpha missing best drafter</div>;

  // KPI: expected best speedup = max over k of the best drafter's speedup.
  let bestSpK = 1;
  let bestSp = best.speedup['1'] ?? 0;
  for (const [k, v] of Object.entries(best.speedup)) {
    if (v > bestSp) {
      bestSp = v;
      bestSpK = Number(k);
    }
  }
  // KPI: measured best speedup = max speedup_vs_none over done sweep cells.
  let measSp = 0;
  let measLabel = '–';
  for (const [name, cell] of Object.entries(sweep.cells)) {
    if (cell.status === 'done' && cell.speedup_vs_none != null && cell.speedup_vs_none > measSp) {
      measSp = cell.speedup_vs_none;
      measLabel = name;
    }
  }
  const splits = summary.corpus.splits;
  const totalBlocks = splits.reduce((a, s) => a + s.blocks, 0);
  const totalTokens = splits.reduce((a, s) => a + s.tokens, 0);
  const val = splits.find((s) => s.split === 'val');

  // Corpus bar chart: tokens per kind on the val split.
  const kinds = summary.corpus.kinds.filter((k) => k.split === 'val');
  const barColor = SERIES.blue[theme];

  const drafters = Object.keys(summary.alpha)
    .filter((d) => d !== 'best')
    .sort((a, b) => DRAFTER_ORDER.indexOf(a) - DRAFTER_ORDER.indexOf(b));

  return (
    <div>
      <div className="tiles">
        <div className="tile" data-testid="tile-best-alpha">
          <p className="k">best drafter α (expected acceptance)</p>
          <p className="v">{fmt(best.alpha, 3)}</p>
          <p className="d">
            {summary.alpha.best} · 95% CI [{fmt(best.alpha_ci95[0], 3)}, {fmt(best.alpha_ci95[1], 3)}] ·{' '}
            {alpha.drafters[summary.alpha.best]?.type} target {best.target}
          </p>
        </div>
        <div className="tile" data-testid="tile-expected-speedup">
          <p className="k">best expected speedup</p>
          <p className="v">{fmt(bestSp, 2)}×</p>
          <p className="d">
            {summary.alpha.best} @ k={bestSpK} · greedy α {fmt(best.greedy, 3)} · coverage{' '}
            {fmt(best.coverage, 3)}
          </p>
        </div>
        <div className="tile" data-testid="tile-measured-speedup">
          <p className="k">best measured speedup (engine, CPU)</p>
          <p className="v">{fmt(measSp, 2)}×</p>
          <p className="d">{measLabel} · vs none on {sweep.target}</p>
        </div>
        <div className="tile" data-testid="tile-corpus">
          <p className="k">corpus</p>
          <p className="v">{totalTokens.toLocaleString('en-US')} tok</p>
          <p className="d">
            {totalBlocks.toLocaleString('en-US')} blocks · val {val ? val.blocks : '–'} blocks /{' '}
            {splits.length} splits
          </p>
        </div>
      </div>

      <div className="card">
        <h2>Drafters</h2>
        <p className="note">
          α = expected acceptance Σ min(p, q) on {alpha.split} blocks · greedy = argmax agreement ·
          speedup assumes draft cost ≈ negligible
        </p>
        <table>
          <thead>
            <tr>
              <th>drafter</th>
              <th>type</th>
              <th>α [CI95]</th>
              <th>greedy</th>
              <th>coverage</th>
              <th>speedup @ k=8</th>
            </tr>
          </thead>
          <tbody>
            {drafters.map((d) => {
              const row = summary.alpha[d];
              if (!row) return null;
              return (
                <tr key={d} data-testid={`drafter-row-${d}`}>
                  <td>{d}</td>
                  <td>{alpha.drafters[d]?.type ?? '–'}</td>
                  <td>
                    {fmt(row.alpha, 4)} [{fmt(row.alpha_ci95[0], 4)}, {fmt(row.alpha_ci95[1], 4)}]
                  </td>
                  <td>{fmt(row.greedy, 4)}</td>
                  <td>{fmt(row.coverage, 3)}</td>
                  <td>{fmt(row.speedup['8'], 3)}×</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>Corpus — tokens per kind (val split)</h2>
        <p className="note">
          kind labels: thinking | text | code | tool_use:&lt;Tool&gt; | tool_result; splits by run,
          never by block
        </p>
        <div className="chart" data-testid="corpus-chart" style={{ height: 240 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={kinds} margin={{ top: 8, right: 8, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="kind" tickLine={false} />
              <YAxis tickLine={false} width={70} />
              <Bar dataKey="tokens" fill={barColor} radius={[2, 2, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <table style={{ marginTop: 12 }}>
          <thead>
            <tr>
              <th>kind</th>
              {splits.map((s) => (
                <th key={s.split}>{s.split} blocks</th>
              ))}
              {splits.map((s) => (
                <th key={`t-${s.split}`}>{s.split} tokens</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {kinds.map((k) => (
              <tr key={k.kind} data-testid={`kind-row-${k.kind}`}>
                <td>{k.kind}</td>
                {splits.map((s) => (
                  <td key={s.split}>
                    {summary.corpus.kinds.find((x) => x.split === s.split && x.kind === k.kind)?.blocks ?? '–'}
                  </td>
                ))}
                {splits.map((s) => (
                  <td key={`t-${s.split}`}>
                    {summary.corpus.kinds.find((x) => x.split === s.split && x.kind === k.kind)?.tokens ?? '–'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
