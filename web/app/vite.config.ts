import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Fixed ports — the parity harness (web/parity/) and any screenshot tooling
 * address these by number, so they must not drift. `strictPort` makes a busy
 * port a hard failure instead of a silent +1.
 *
 *   dev      http://localhost:5273
 *   preview  http://localhost:5274
 *
 * 5173 is deliberately avoided: web/frontend (the old UI) already uses the
 * Vite default.
 */
export const DEV_PORT = 5273;
export const PREVIEW_PORT = 5274;

export default defineConfig({
  plugins: [react()],
  server: {
    port: DEV_PORT,
    strictPort: true,
  },
  preview: {
    port: PREVIEW_PORT,
    strictPort: true,
  },
  build: {
    // The design-data modules are large (topic-data.js alone is ~640 kB of
    // verbatim demo data). They are dynamically imported, so they land in
    // their own chunks; raise the warning bar rather than split the verbatim
    // files.
    chunkSizeWarningLimit: 1200,

    // The CSS foundations are verbatim copies and the whole project is a
    // byte-level parity exercise, so the built CSS is left unminified: `dev`
    // and `preview` then serve identical bytes, and a parity failure can be
    // read straight out of dist/ without un-minifying it first. Costs ~5 kB
    // gzipped. Do not turn this on for a size win without re-running the
    // parity baselines.
    cssMinify: false,
  },
});
