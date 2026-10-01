/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './' so built asset URLs stay relative: the gateway mounts this app
// at /ui/ (static dir, no SPA fallback), and dev serves from /.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { outDir: 'dist' },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
});
