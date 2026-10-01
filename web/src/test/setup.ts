import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';
import '@testing-library/jest-dom/vitest';

// globals are off, so register RTL cleanup explicitly.
afterEach(() => {
  cleanup();
});

// jsdom lacks matchMedia; App reads prefers-color-scheme on first paint.
if (typeof window !== 'undefined' && !window.matchMedia) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (window as any).matchMedia = (q: string) => ({
    matches: false,
    media: q,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  });
}

// jsdom has no ResizeObserver; recharts' ResponsiveContainer needs one to size.
// Fire the callback with a fixed size so charts render deterministically in tests.
class FakeResizeObserver {
  private cb: (entries: unknown[]) => void;
  constructor(cb: (entries: unknown[]) => void) {
    this.cb = cb;
  }
  observe(el: Element) {
    this.cb([{ target: el, contentRect: { width: 800, height: 400 } }]);
  }
  unobserve() {}
  disconnect() {}
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(globalThis as any).ResizeObserver ??= FakeResizeObserver;
