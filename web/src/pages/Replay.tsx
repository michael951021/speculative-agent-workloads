import { useMemo, useState } from 'react';
import type { ReplaysData } from '../types';
import { fmt } from '../theme';

interface Props {
  replays: ReplaysData;
}

interface TokenDetail {
  idx: number;
  text: string;
  p: number | null;
  q: number | null;
  accept: boolean;
  state: 'accept' | 'reject' | 'none';
}

function label(replays: ReplaysData, i: number): string {
  const b = replays.blocks[i];
  return `${b.run} #${b.seq} · ${b.kind} · ${b.n_tokens} tok`;
}

export default function Replay({ replays }: Props) {
  const drafters = Object.keys(replays.drafters);
  const [bi, setBi] = useState(0);
  const [drafter, setDrafter] = useState(drafters[0] ?? '');
  const [detail, setDetail] = useState<TokenDetail | null>(null);

  const block = replays.blocks[bi];
  const rd = (block as Record<string, unknown>)[drafter] as
    | { p: (number | null)[]; q: (number | null)[]; accept: boolean[] }
    | undefined;

  const stats = useMemo(() => {
    if (!rd) return null;
    const proposed = rd.p.filter((x) => x != null).length;
    const accepted = rd.accept.filter(Boolean).length;
    return { proposed, accepted, rate: proposed ? accepted / proposed : null };
  }, [rd]);

  const stateOf = (i: number): 'accept' | 'reject' | 'none' =>
    !rd || rd.q[i] == null ? 'none' : rd.accept[i] ? 'accept' : 'reject';

  return (
    <div className="card">
      <h2>Token replay</h2>
      <p className="note">
        {replays.accept_rule}. p = target probability of the next corpus token; q = drafter
        probability of that token (accept_rule: greedy spec-accept). Target {replays.target},{' '}
        {replays.split} split, ctx budget {replays.ctx_budget} tok.
      </p>
      <div className="replay-controls">
        <label>
          block{' '}
          <select
            aria-label="replay block"
            value={bi}
            onChange={(e) => {
              setBi(Number(e.target.value));
              setDetail(null);
            }}
          >
            {replays.blocks.map((_, i) => (
              <option key={i} value={i}>
                {label(replays, i)}
              </option>
            ))}
          </select>
        </label>
        <label>
          drafter{' '}
          <select
            aria-label="replay drafter"
            value={drafter}
            onChange={(e) => {
              setDrafter(e.target.value);
              setDetail(null);
            }}
          >
            {drafters.map((d) => (
              <option key={d} value={d}>
                {d} ({replays.drafters[d].type})
              </option>
            ))}
          </select>
        </label>
      </div>
      {stats && (
        <div className="block-meta" data-testid="replay-stats">
          <span>
            tokens <b>{block.n_tokens}</b>
          </span>
          <span>
            proposals <b>{stats.proposed}</b>
          </span>
          <span>
            accepted <b>{stats.accepted}</b>
          </span>
          <span>
            acceptance rate <b>{fmt(stats.rate, 3)}</b>
          </span>
        </div>
      )}
      <div className="legend">
        <span className="item">
          <span
            className="swatch"
            style={{ width: 12, height: 12, borderRadius: 3, background: 'var(--good-bg)', border: '1px solid var(--good)' }}
          />
          accepted
        </span>
        <span className="item">
          <span
            className="swatch"
            style={{ width: 12, height: 12, borderRadius: 3, background: 'var(--hover)', border: '1px solid var(--border)' }}
          />
          rejected
        </span>
        <span className="item">
          <span
            className="swatch"
            style={{ width: 12, height: 12, borderRadius: 3, border: '1px dashed var(--muted)' }}
          />
          no proposal
        </span>
      </div>
      <div className="block" data-testid="replay-block">
        {block.tokens.map((tok, i) => (
          <span
            key={i}
            data-idx={i}
            data-testid={`replay-token-${i}`}
            className={`tok ${stateOf(i)}`}
            title={`#${i} ${tok.text.replace(/\n/g, '\\n')} — ${hoverText(i)}`}
            onMouseEnter={() => setDetail(detailOf(i))}
            onFocus={() => setDetail(detailOf(i))}
          >
            {tok.text}
          </span>
        ))}
      </div>
      <div className="token-detail" aria-live="polite" data-testid="replay-detail">
        {detail
          ? `#${detail.idx} “${detail.text.replace(/\n/g, '\\n')}” · p=${fmt(detail.p, 4)} · q=${fmt(detail.q, 4)} · ${detail.state}`
          : 'hover a token to see p, q and its outcome'}
      </div>
    </div>
  );

  function hoverText(i: number): string {
    if (!rd || rd.q[i] == null) return 'no proposal';
    return `p=${rd.p[i]} q=${rd.q[i]} ${rd.accept[i] ? 'accepted' : 'rejected'}`;
  }

  function detailOf(i: number): TokenDetail {
    const tok = block.tokens[i];
    return {
      idx: i,
      text: tok.text,
      p: rd?.p[i] ?? null,
      q: rd?.q[i] ?? null,
      accept: rd?.accept[i] ?? false,
      state: stateOf(i),
    };
  }
}
