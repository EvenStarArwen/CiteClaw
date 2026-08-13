// Shared configuration for the parity harness.
// Viewport definitions are a PRODUCT REQUIREMENT (responsive folding must be preserved),
// so they live here in one place and are referenced by name everywhere else.

import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const PKG_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const WEB_ROOT = path.resolve(PKG_ROOT, '..');
export const DESIGN_REFERENCE_DIR = path.join(WEB_ROOT, 'design-reference');
export const DESIGN_DEMO_FILE = 'KnowledgeLab iPad Demo.html';

/** Baseline viewports — every screen MUST be captured at these two. */
export const BASELINE_VIEWPORTS = [
  { name: 'desktop-1600x900', width: 1600, height: 900, note: 'Baseline desktop' },
  { name: 'ipad-pro-12.9-landscape-1366x1024', width: 1366, height: 1024, note: 'Baseline iPad Pro 12.9 landscape' },
];

/** Responsive sweep — protects the demo's folding behaviour at narrower widths. */
export const SWEEP_VIEWPORTS = [
  { name: 'ipad-portrait-1024x1366', width: 1024, height: 1366, note: 'iPad Pro 12.9 portrait' },
  { name: 'ipad-11-1194x834', width: 1194, height: 834, note: 'iPad 11 landscape' },
  { name: 'w900-900x1200', width: 900, height: 1200, note: 'Narrow tablet / split view' },
  { name: 'w768-768x1024', width: 768, height: 1024, note: 'iPad 9.7 portrait / fold boundary' },
];

export const ALL_VIEWPORTS = [...BASELINE_VIEWPORTS, ...SWEEP_VIEWPORTS];

export function resolveViewports(spec = 'all') {
  if (spec === 'baseline') return BASELINE_VIEWPORTS;
  if (spec === 'sweep') return SWEEP_VIEWPORTS;
  if (spec === 'all') return ALL_VIEWPORTS;
  const wanted = String(spec).split(',').map((s) => s.trim()).filter(Boolean);
  const out = wanted.map((n) => {
    const vp = ALL_VIEWPORTS.find((v) => v.name === n);
    if (!vp) throw new Error(`Unknown viewport "${n}". Known: ${ALL_VIEWPORTS.map((v) => v.name).join(', ')}`);
    return vp;
  });
  if (!out.length) throw new Error('No viewports resolved');
  return out;
}

/**
 * Determinism knobs. These are applied IDENTICALLY to every target (demo and
 * rewrite) — they are a property of the harness, never of one side.
 */
export const DETERMINISM = {
  // Fixed wall clock installed into the page before any script runs.
  // 2026-01-01T00:00:00Z — chosen to be stable and far from DST boundaries.
  epochMs: Date.UTC(2026, 0, 1, 0, 0, 0),
  timezoneId: 'UTC',
  locale: 'en-US',
  // Virtual milliseconds of page time to burn before a screen is considered settled.
  // Long enough for the demo's mount + streaming-replay animations to finish.
  warmupVirtualMs: 12000,
  // Virtual time advanced per tick, interleaved with real time so async work
  // (bundle unpack, font decode, fetch) can make progress.
  virtualTickMs: 250,
  realTickMs: 12,
  // PRNG seed for the Math.random / crypto shims.
  randomSeed: 0x5eed1234,
  // Pixel-stability gate: how many consecutive identical frames end the settle.
  stableFrames: 2,
  stablePollMs: 120,
  stableMaxPolls: 40,
  deviceScaleFactor: 1,
};

/** Anti-aliasing-tolerant diff defaults. Raw counts are always reported too. */
export const DIFF_DEFAULTS = {
  // pixelmatch per-pixel colour distance threshold (0..1). 0.1 tolerates AA/subpixel drift.
  threshold: 0.1,
  includeAA: false,
  // A screen passes when the differing-pixel ratio is at or below this.
  // Default 0 => byte-for-byte parity expected; relax explicitly per run.
  maxDiffRatio: 0,
};
