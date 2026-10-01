import { describe, expect, it } from 'vitest';
import { expectedTpc } from '../theme';

// Known answers for the Leviathan et al. 2023 closed form:
// (1 - alpha^(k+1)) / (1 - alpha), limit k+1 at alpha=1.
describe('expectedTpc (known answers)', () => {
  it('alpha=0.5, k=3 -> (1-0.5^4)/0.5 = 1.875 exactly', () => {
    expect(expectedTpc(0.5, 3)).toBeCloseTo(1.875, 12);
  });
  it('alpha=1 -> k+1 (all drafts accepted)', () => {
    for (const k of [1, 4, 8]) expect(expectedTpc(1, k)).toBeCloseTo(k + 1, 12);
  });
  it('alpha=0 -> 1 (only the bonus token)', () => {
    expect(expectedTpc(0, 8)).toBeCloseTo(1, 12);
  });
  it('k=1 -> 1 + alpha', () => {
    for (const a of [0.2, 0.5, 0.9]) expect(expectedTpc(a, 1)).toBeCloseTo(1 + a, 12);
  });
  it('increases in k for alpha in (0,1)', () => {
    let prev = 0;
    for (let k = 1; k <= 8; k++) {
      const v = expectedTpc(0.7, k);
      expect(v).toBeGreaterThan(prev);
      prev = v;
    }
  });
});
