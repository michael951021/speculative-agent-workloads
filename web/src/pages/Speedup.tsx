import { useState } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { AlphaData, SweepData } from '../types';
import { DRAFTER_HUE, DRAFTER_ORDER, SERIES, expectedTpc, fmt } from '../theme';

interface Props {
  alpha: AlphaData;
  sweep: SweepData;
  theme: 'light' | 'dark';
}

const K_LIST = [1, 2, 3, 4, 5, 6, 7, 8];

export default function Speedup({ alpha, sweep, theme }: Props) {
  const drafters = Object.keys(alpha.drafters).sort(
    (a, b) => DRAFTER_ORDER.indexOf(a) - DRAFTER_ORDER.indexOf(b),
  );

  // Chart 1 data: expected speedup vs k with CI band, per drafter (alpha metric).
  const expRows = K_LIST.map((k) => {
    const row: Record<string, number | [number, number]> = { k };
    for (const d of drafters) {
      const o = alpha.drafters[d].overall;
      const v = o.speedup[String(k)];
      const ci = o.speedup_ci95[String(k)];
      if (v != null) row[d] = v;
      if (ci) row[`${d}_ci`] = ci;
    }
    return row;
  });

  // Chart 2 data: measured (engine) vs expected (Leviathan from the cell's own alpha), T=0.
  const sweepDrafters = DRAFTER_ORDER.filter((d) =>
    K_LIST.some((k) => cell(sweep, d, k, 0)?.status === 'done'),
  );
  const [sel, setSel] = useState(sweepDrafters[0] ?? '');
  const measRows = K_LIST.filter((k) => cell(sweep, sel, k, 0)?.status === 'done').map((k) => {
    const c = cell(sweep, sel, k, 0)!;
    return {
      k,
      measured: c.tpc ?? null,
      expected: c.alpha != null ? expectedTpc(c.alpha, k) : null,
    };
  });

  const colorOf = (d: string) => SERIES[DRAFTER_HUE[d] ?? 'blue'][theme];

  // All done cells as a table (table view; also the yellow-slot contrast relief).
  const rows = Object.entries(sweep.cells)
    .filter(([, c]) => c.status === 'done')
    .map(([name, c]) => {
      const m = name.match(/^(\S+) k=(\d+) T=([\d.]+)$/);
      return {
        name,
        drafter: m?.[1] ?? name,
        k: m ? Number(m[2]) : null,
        t: m?.[3] ?? '?',
        c,
      };
    })
    .sort((a, b) =>
      DRAFTER_ORDER.indexOf(a.drafter) - DRAFTER_ORDER.indexOf(b.drafter) || (a.k ?? 0) - (b.k ?? 0),
    );

  return (
    <div>
      <div className="card">
        <h2>Expected speedup vs k, per drafter</h2>
        <p className="note">
          alpha metric on {alpha.split} blocks, target {alpha.target}; band = 95% CI on speedup
          (Leviathan et al. 2023). Draft cost assumed negligible — a lower bound on the ratio of
          target-call counts.
        </p>
        <div className="legend">
          {drafters.map((d) => (
            <span className="item" key={d}>
              <span className="swatch" style={{ background: colorOf(d) }} />
              {d}
            </span>
          ))}
        </div>
        <div className="chart" data-testid="speedup-chart" style={{ height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={expRows} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="k" type="number" domain={[1, 8]} ticks={K_LIST} tickLine={false} />
              <YAxis tickLine={false} width={44} />
              <Tooltip />
              {drafters.map((d) => (
                <Area
                  key={`${d}-ci`}
                  dataKey={`${d}_ci`}
                  stroke="none"
                  fill={colorOf(d)}
                  fillOpacity={0.12}
                  isAnimationActive={false}
                  connectNulls
                />
              ))}
              {drafters.map((d) => (
                <Line
                  key={d}
                  dataKey={d}
                  name={d}
                  stroke={colorOf(d)}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  isAnimationActive={false}
                  connectNulls
                />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h2>Measured vs expected tokens per call (engine, T=0)</h2>
        <p className="note">
          measured = engine tpc from the Ray sweep ({sweep.target}, {sweep.n_gen} fresh tokens per
          prompt); expected = Leviathan (1−α^(k+1))/(1−α) from the same cell&rsquo;s measured α —
          measured sits below expected: draft+verify overhead on CPU.
        </p>
        <div className="replay-controls">
          <label>
            drafter{' '}
            <select aria-label="sweep drafter" value={sel} onChange={(e) => setSel(e.target.value)}>
              {sweepDrafters.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="legend">
          <span className="item">
            <span className="swatch" style={{ background: colorOf(sel) }} />
            measured tpc
          </span>
          <span className="item">
            <span
              className="swatch dashed"
              style={{ borderColor: colorOf(sel) }}
            />
            expected tpc (from cell α)
          </span>
        </div>
        <div className="chart" data-testid="measured-chart" style={{ height: 280 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={measRows} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="k" type="number" domain={[1, 8]} ticks={K_LIST} tickLine={false} />
              <YAxis tickLine={false} width={44} />
              <Tooltip />
              <Line
                dataKey="measured"
                name="measured"
                stroke={colorOf(sel)}
                strokeWidth={2}
                dot={{ r: 4 }}
                isAnimationActive={false}
              />
              <Line
                dataKey="expected"
                name="expected"
                stroke={colorOf(sel)}
                strokeWidth={2}
                strokeDasharray="6 4"
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h2>Sweep cells (done)</h2>
        <p className="note">
          {rows.length} done · {sweep.none_tok_s['0.0']?.toFixed(2)} tok/s baseline (none, T=0)
        </p>
        <table data-testid="sweep-table">
          <thead>
            <tr>
              <th>drafter</th>
              <th>k</th>
              <th>T</th>
              <th>α (n_acc/n_draft)</th>
              <th>tpc</th>
              <th>tok/s</th>
              <th>speedup vs none</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ name, drafter, k, t, c }) => (
              <tr key={name} data-testid={`sweep-row-${name}`}>
                <td>{drafter}</td>
                <td>{k ?? '–'}</td>
                <td>{t}</td>
                <td>{fmt(c.alpha, 4)}</td>
                <td>{fmt(c.tpc, 3)}</td>
                <td>{fmt(c.tok_s, 2)}</td>
                <td>{fmt(c.speedup_vs_none, 3)}×</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function cell(
  sweep: SweepData,
  drafter: string,
  k: number,
  temp: number,
): SweepData['cells'][string] | undefined {
  const t = temp === 0 ? '0.0' : String(temp);
  return sweep.cells[`${drafter} k=${k} T=${t}`];
}
