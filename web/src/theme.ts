// Validated default palette (dataviz skill), light/dark steps.
export const SERIES: Record<string, { light: string; dark: string }> = {
  blue: { light: '#2a78d6', dark: '#3987e5' },
  orange: { light: '#eb6834', dark: '#d95926' },
  aqua: { light: '#1baf7a', dark: '#199e70' },
  yellow: { light: '#eda100', dark: '#c98500' },
  magenta: { light: '#e87ba4', dark: '#d55181' },
  violet: { light: '#4a3aa7', dark: '#9085e9' },
};

// Fixed drafter order = fixed hue assignment; never cycled.
export const DRAFTER_ORDER = ['ngram', 'tiny', 'base', 'lora', 'lora_dpo'];
export const DRAFTER_HUE: Record<string, keyof typeof SERIES> = {
  ngram: 'blue',
  tiny: 'orange',
  base: 'aqua',
  lora: 'yellow',
  lora_dpo: 'magenta',
  dpo: 'violet',
};

export const STATUS = {
  good: '#0ca30c', // accepted
  rejected: { light: '#898781', dark: '#898781' }, // muted ink, never a series color
};

// Sequential blue ramp (magnitude: heatmap). t in [0,1] -> hex.
const RAMP: [number, string][] = [
  [0, '#cde2fb'],
  [0.2, '#86b6ef'],
  [0.5, '#3987e5'],
  [0.75, '#184f95'],
  [1, '#104281'],
];

function hexToRgb(h: string): [number, number, number] {
  return [
    parseInt(h.slice(1, 3), 16),
    parseInt(h.slice(3, 5), 16),
    parseInt(h.slice(5, 7), 16),
  ];
}

export function rampColor(t: number, dark: boolean): string {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < RAMP.length; i++) {
    if (x <= RAMP[i][0]) {
      const [t0, c0] = RAMP[i - 1];
      const [t1, c1] = RAMP[i];
      const f = (x - t0) / (t1 - t0);
      const a = hexToRgb(c0);
      const b = hexToRgb(c1);
      const mix = a.map((v, j) => Math.round(v + (b[j] - v) * f));
      return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
    }
  }
  return dark ? '#104281' : RAMP[RAMP.length - 1][1];
}

// Leviathan et al. 2023: expected tokens per target call for k drafts at
// acceptance rate alpha. Closed form (1 - a^(k+1)) / (1 - a); limit k+1 at a=1.
export function expectedTpc(alpha: number, k: number): number {
  if (alpha >= 1 - 1e-12) return k + 1;
  return (1 - Math.pow(alpha, k + 1)) / (1 - alpha);
}

export function fmt(x: number | null | undefined, digits = 3): string {
  return x == null || Number.isNaN(x) ? '–' : x.toFixed(digits);
}
