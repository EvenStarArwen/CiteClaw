// Determinism layer.
//
// The design demo replays a pipeline run with streaming text and animated
// visualisations, and several of its views seed layout from Math.random(). A
// naive screenshot of it is therefore never reproducible. Everything in this
// file exists to make two captures of the SAME target byte-identical, and it is
// applied IDENTICALLY to the demo and to the rewrite — the harness never gives
// one side an advantage the other does not get.
//
// Layers, in the order they take effect:
//   1. Seeded PRNG        — Math.random / crypto.getRandomValues / randomUUID.
//   2. Frozen clock       — Playwright clock API: Date, performance.now,
//                           setTimeout/Interval, requestAnimationFrame/idle.
//   3. Deterministic warm-up — a FIXED number of virtual milliseconds is burned
//                           after load, then the clock is paused. Both targets
//                           therefore see the same amount of page time.
//   4. Font readiness     — document.fonts.ready before any pixel is read.
//   5. Pixel stability    — screenshot until N consecutive frames are identical.
//   6. animations:'disabled' at screenshot time — Playwright fast-forwards
//                           finite CSS/Web animations to their end state and
//                           resets infinite ones. This is a capture-time lever,
//                           not a change to the product's CSS.

import { chromium } from 'playwright';
import { DETERMINISM } from './config.mjs';
import { installCdnMirror } from './cdn-mirror.mjs';

/**
 * Init script injected before ANY page script runs. Must be self-contained
 * (it is serialised into the page) and must not depend on module scope.
 */
function seedScript(seed) {
  // mulberry32 — small, fast, well-distributed, fully reproducible.
  let s = seed >>> 0;
  const src = `(() => {
    let s = ${seed} >>> 0;
    function next() {
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    Math.random = next;
    try {
      const g = globalThis.crypto;
      if (g) {
        Object.defineProperty(g, 'getRandomValues', {
          configurable: true, writable: true,
          value: (arr) => { for (let i = 0; i < arr.length; i++) arr[i] = Math.floor(next() * 256); return arr; },
        });
        Object.defineProperty(g, 'randomUUID', {
          configurable: true, writable: true,
          value: () => {
            const h = '0123456789abcdef';
            let out = '';
            for (let i = 0; i < 36; i++) {
              if (i === 8 || i === 13 || i === 18 || i === 23) out += '-';
              else if (i === 14) out += '4';
              else out += h[Math.floor(next() * 16)];
            }
            return out;
          },
        });
      }
    } catch (_) {}
    // Marker so the harness can assert the shim actually landed.
    globalThis.__parityDeterminism = { seed: ${seed} };
  })();`;
  void s;
  return src;
}

/** Launch a chromium instance configured for reproducible rendering. */
export async function launchBrowser({ headless = true } = {}) {
  return chromium.launch({
    headless,
    args: [
      // Force a stable rasterisation path; GPU rasterisation varies by machine
      // state and is the single biggest source of "same page, different pixels".
      '--disable-lcd-text',
      '--disable-font-subpixel-positioning',
      '--force-color-profile=srgb',
      '--disable-partial-raster',
      '--disable-skia-runtime-opts',
      '--hide-scrollbars',
      '--force-device-scale-factor=1',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows',
      '--deterministic-mode',
    ],
  });
}

/**
 * Create a context whose pages are deterministic from the first byte.
 * The clock is installed on the CONTEXT so it is live before navigation.
 */
export async function newDeterministicContext(browser, viewport, opts = {}) {
  const d = { ...DETERMINISM, ...opts };
  const context = await browser.newContext({
    viewport: { width: viewport.width, height: viewport.height },
    deviceScaleFactor: d.deviceScaleFactor,
    timezoneId: d.timezoneId,
    locale: d.locale,
    colorScheme: 'light',
    reducedMotion: null, // deliberately NOT 'reduce' — that would change appearance.
    forcedColors: null,
    isMobile: false,
    hasTouch: false,
  });
  // Serve the demo's six public-CDN scripts from a pinned local mirror, in a
  // deterministic order. Without this every capture depends on live network
  // latency and the smooth-scrollbar plugin race. See src/cdn-mirror.mjs.
  const unmirroredSink = [];
  if (opts.cdnMirror !== false) {
    await installCdnMirror(context, {
      blockUnmirrored: opts.blockUnmirrored !== false,
      onUnmirrored: (rec) => unmirroredSink.push(rec),
    });
  }
  context.__parityUnmirrored = unmirroredSink;

  await context.addInitScript(seedScript(d.randomSeed));
  // Install the fake clock AND immediately pause it at the epoch. Without the
  // pause the fake clock still ticks with wall time, so the amount of page time
  // a target receives would depend on how fast the machine unpacked the bundle
  // — the exact non-determinism this harness exists to remove. From here on,
  // page time only moves when the harness explicitly calls runFor().
  await context.clock.install({ time: new Date(d.epochMs) });
  await context.clock.pauseAt(new Date(d.epochMs));
  return context;
}

/**
 * Burn a fixed amount of VIRTUAL page time, interleaved with small amounts of
 * real time so that genuinely async work (bundle unpack, font decode, image
 * decode) can proceed. Deterministic because the virtual total is fixed.
 */
export async function warmUpVirtualClock(page, opts = {}) {
  const d = { ...DETERMINISM, ...opts };
  const total = opts.virtualMs ?? d.warmupVirtualMs;
  const ticks = Math.ceil(total / d.virtualTickMs);
  for (let i = 0; i < ticks; i++) {
    await page.clock.runFor(d.virtualTickMs);
    await new Promise((r) => setTimeout(r, d.realTickMs));
  }
  // The clock was paused at install time and only ever moved by runFor(), so
  // every target lands on exactly the same fake timestamp here.
}

/** Wait until webfonts are loaded and laid out. */
export async function waitForFonts(page, timeout = 15000) {
  await page.evaluate(
    (t) =>
      Promise.race([
        document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve(),
        new Promise((r) => setTimeout(r, t)),
      ]).then(() => undefined),
    timeout,
  );
}

/**
 * Screenshot repeatedly until N consecutive frames are byte-identical.
 * Returns the settled buffer plus diagnostics, so an unstable screen is
 * reported rather than silently baked into a baseline.
 */
export async function settleAndShoot(page, shotOptions = {}, opts = {}) {
  const d = { ...DETERMINISM, ...opts };
  let prev = null;
  let identical = 0;
  let polls = 0;
  let buf = null;

  for (; polls < d.stableMaxPolls; polls++) {
    buf = await page.screenshot({ animations: 'disabled', caret: 'hide', scale: 'css', ...shotOptions });
    if (prev && prev.equals(buf)) {
      identical++;
      if (identical >= d.stableFrames - 1) break;
    } else {
      identical = 0;
    }
    prev = buf;
    await new Promise((r) => setTimeout(r, d.stablePollMs));
  }

  return { buffer: buf, polls: polls + 1, stable: identical >= d.stableFrames - 1 };
}

/**
 * Full open-and-settle for one page. Returns the page plus a network log so the
 * caller can assert that a target pulled nothing off the public internet
 * (external requests are a determinism hazard and an offline-demo hazard).
 */
export async function openSettled(context, url, opts = {}) {
  const page = await context.newPage();
  const network = [];
  const consoleErrors = [];
  page.on('request', (req) => {
    const u = req.url();
    if (!u.startsWith('data:') && !u.startsWith('blob:')) network.push({ method: req.method(), url: u });
  });
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text().slice(0, 300));
  });
  page.on('pageerror', (err) => consoleErrors.push('pageerror: ' + String(err && err.message).slice(0, 300)));

  await page.goto(url, { waitUntil: 'load', timeout: 60000 });
  await warmUpVirtualClock(page, opts);
  await waitForFonts(page);
  await page.waitForLoadState('networkidle', { timeout: 30000 }).catch(() => {});
  return { page, network, consoleErrors };
}
