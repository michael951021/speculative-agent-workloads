import { Fragment, useState } from 'react';
import type { AlphaData } from '../types';
import { DRAFTER_HUE, DRAFTER_ORDER, SERIES, fmt, rampColor } from '../theme';

interface Props {
  alpha: AlphaData;
  theme: 'light' | 'dark';
}

interface Hover {
  drafter: string;
  kind: string;
  alpha: number;
  ci: [number, number];
  nTokens: number;
}

const KIND_ORDER = ['thinking', 'text', 'code', 'tool_use:Bash', 'tool_use:Edit', 'tool_use:Read', 'tool_use:WebFetch', 'tool_use:Write', 'tool_result'];

export default function Heatmap({ alpha, theme }: Props) {
  const [hover, setHover] = useState<Hover | null>(null);
  const drafters = Object.keys(alpha.drafters).sort(
    (a, b) => DRAFTER_ORDER.indexOf(a) - DRAFTER_ORDER.indexOf(b),
  );
  const kinds = KIND_ORDER.filter((k) =>
    drafters.some((d) => k in alpha.drafters[d].by_kind),
  );

  const color = (t: number) => rampColor(t, theme === 'dark');
  const textOn = (t: number) => (t > 0.45 ? '#ffffff' : '#0b0b0b');

  return (
    <div className="card">
      <h2>Acceptance α by kind × drafter</h2>
      <p className="note">
        α = expected acceptance per kind, {alpha.split} blocks, target {alpha.target}. Colour is a
        fixed 0→1 blue ramp; hover a cell for the 95% CI. Dashed cell = drafter has no proposals
        for that kind (or kind absent).
      </p>
      <div
        className="heat"
        style={{ gridTemplateColumns: `160px repeat(${drafters.length}, 1fr)` }}
      >
        <div className="corner" />
        {drafters.map((d) => (
          <div className="hcol" key={d} data-testid={`heat-col-${d}`}>
            {d}
          </div>
        ))}
        {kinds.map((k) => (
          <Fragment key={k}>
            <div className="hrow">{k}</div>
            {drafters.map((d) => {
              const cell = alpha.drafters[d].by_kind[k];
              if (!cell) {
                return (
                  <div className="hcell na" key={`${k}-${d}`} data-testid={`cell-${d}-${k}`}>
                    –
                  </div>
                );
              }
              const t = cell.alpha;
              return (
                <div
                  key={`${k}-${d}`}
                  className="hcell"
                  role="img"
                  tabIndex={0}
                  aria-label={`${d} ${k}: alpha ${fmt(cell.alpha, 3)}`}
                  data-testid={`cell-${d}-${k}`}
                  style={{ background: color(t), color: textOn(t) }}
                  onMouseEnter={() =>
                    setHover({
                      drafter: d,
                      kind: k,
                      alpha: cell.alpha,
                      ci: cell.alpha_ci95,
                      nTokens: cell.n_tokens,
                    })
                  }
                  onFocus={() =>
                    setHover({
                      drafter: d,
                      kind: k,
                      alpha: cell.alpha,
                      ci: cell.alpha_ci95,
                      nTokens: cell.n_tokens,
                    })
                  }
                >
                  {fmt(cell.alpha, 3)}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
      <div className="hover-panel" aria-live="polite" data-testid="heat-hover">
        {hover ? (
          <>
            <b>
              {hover.drafter} × {hover.kind}
            </b>{' '}
            — α {fmt(hover.alpha, 4)} [95% CI {fmt(hover.ci[0], 3)}, {fmt(hover.ci[1], 3)}] ·{' '}
            {hover.nTokens.toLocaleString('en-US')} tokens
          </>
        ) : (
          'hover or focus a cell to see its 95% CI'
        )}
      </div>
      <div className="legend" style={{ marginTop: 8 }}>
        {drafters.map((d) => (
          <span className="item" key={d}>
            <span
              className="swatch"
              style={{ background: SERIES[DRAFTER_HUE[d]][theme] }}
            />
            {d}
          </span>
        ))}
      </div>
    </div>
  );
}
