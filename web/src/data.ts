import type { AlphaData, AppData, ReplaysData, SummaryData, SweepData } from './types';

// Files live in web/public/data/, copied to dist/data/ by the build.
// base './' keeps these relative so the gateway can mount the app at /ui/.
const DATA_URL = (name: string) => `${import.meta.env.BASE_URL}data/${name}.json`;

async function load<T>(name: string): Promise<T> {
  const res = await fetch(DATA_URL(name));
  if (!res.ok) throw new Error(`failed to load data/${name}.json: ${res.status}`);
  return (await res.json()) as T;
}

export async function loadData(): Promise<AppData> {
  const [summary, alpha, sweep, replays] = await Promise.all([
    load<SummaryData>('summary'),
    load<AlphaData>('alpha'),
    load<SweepData>('sweep'),
    load<ReplaysData>('replays'),
  ]);
  return { summary, alpha, sweep, replays };
}
