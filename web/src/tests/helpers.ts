import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { AppData } from '../types';

// The acceptance tests run on the REAL export, web/public/data/*.json
// (written by `python -m draftlab.export_ui`), not fixtures.
// cwd is web/ when vitest runs via npm scripts.
const DATA_DIR = resolve(process.cwd(), 'public/data');

export function realData(): AppData {
  const j = (f: string) => JSON.parse(readFileSync(`${DATA_DIR}/${f}`, 'utf8'));
  return {
    summary: j('summary.json'),
    alpha: j('alpha.json'),
    sweep: j('sweep.json'),
    replays: j('replays.json'),
  };
}
