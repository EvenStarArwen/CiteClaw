// Offline, order-deterministic mirror of the demo's public-CDN dependencies.
//
// WHY THIS EXISTS
// The design demo is *almost* self-contained: its sibling .js/.dc.html files,
// the Google Fonts CSS (with the woff2 payloads as data: URIs) and React UMD are
// all inlined into the bundle. Six scripts are NOT — they are still plain
// <script src="https://..."> tags inside Paper Card.dc.html, Runs.dc.html and
// Explore.dc.html:
//
//   cdnjs        smooth-scrollbar@8.8.4  (+ its overscroll plugin)
//   cdn.jsdelivr marked@12.0.2, katex@0.16.11, turndown@7.2.0, turndown-plugin-gfm@1.0.2
//
// Two consequences the harness has to neutralise:
//   1. Live-internet latency makes every capture depend on the network, and the
//      two smooth-scrollbar files RACE — overscroll.min.js frequently wins and
//      throws "Cannot read properties of undefined (reading 'ScrollbarPlugin')".
//   2. A capture run would silently differ depending on whether the machine is
//      online.
//
// So: those URLs are fulfilled from npm packages pinned to the SAME versions,
// and dependent scripts are held until their dependency's global exists. Applied
// identically to every target. The design-reference copies are never touched.
//
// NOTE: this mirrors the demo's *runtime* behaviour, it does not change its
// appearance — it removes a timing race that the demo itself does not handle.

import fsp from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { PKG_ROOT } from './config.mjs';

const NM = path.join(PKG_ROOT, 'node_modules');

/** url pattern -> { file, contentType, waitForGlobal } */
export const CDN_MIRROR = [
  {
    url: 'https://cdnjs.cloudflare.com/ajax/libs/smooth-scrollbar/8.8.4/smooth-scrollbar.min.js',
    file: path.join(NM, 'smooth-scrollbar/dist/smooth-scrollbar.js'),
    contentType: 'text/javascript; charset=utf-8',
    providesGlobal: 'Scrollbar',
  },
  {
    url: 'https://cdnjs.cloudflare.com/ajax/libs/smooth-scrollbar/8.8.4/plugins/overscroll.min.js',
    file: path.join(NM, 'smooth-scrollbar/dist/plugins/overscroll.js'),
    contentType: 'text/javascript; charset=utf-8',
    waitForGlobal: 'Scrollbar',
  },
  {
    url: 'https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js',
    file: path.join(NM, 'marked/marked.min.js'),
    contentType: 'text/javascript; charset=utf-8',
    providesGlobal: 'marked',
  },
  {
    url: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js',
    file: path.join(NM, 'katex/dist/katex.min.js'),
    contentType: 'text/javascript; charset=utf-8',
    providesGlobal: 'katex',
  },
  {
    url: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
    file: path.join(NM, 'katex/dist/katex.min.css'),
    contentType: 'text/css; charset=utf-8',
  },
  {
    url: 'https://cdn.jsdelivr.net/npm/turndown@7.2.0/dist/turndown.js',
    file: path.join(NM, 'turndown/dist/turndown.js'),
    contentType: 'text/javascript; charset=utf-8',
    providesGlobal: 'TurndownService',
  },
  {
    url: 'https://cdn.jsdelivr.net/npm/turndown-plugin-gfm@1.0.2/dist/turndown-plugin-gfm.js',
    file: path.join(NM, 'turndown-plugin-gfm/dist/turndown-plugin-gfm.js'),
    contentType: 'text/javascript; charset=utf-8',
    waitForGlobal: 'TurndownService',
  },
];

const FONT_PREFIX = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/fonts/';

/**
 * Route every external request of a context through the local mirror.
 * @param {import('playwright').BrowserContext} context
 * @param {object} opts
 * @param {(entry: object) => void} [opts.onUnmirrored] called for external requests with no mirror entry
 * @param {boolean} [opts.blockUnmirrored=true] abort (rather than allow) unmirrored external requests
 */
export async function installCdnMirror(context, { onUnmirrored, blockUnmirrored = true } = {}) {
  const byUrl = new Map(CDN_MIRROR.map((e) => [e.url, e]));
  const cache = new Map();
  const unmirrored = [];

  const read = async (file) => {
    if (!cache.has(file)) cache.set(file, await fsp.readFile(file));
    return cache.get(file);
  };

  await context.route(
    (url) => {
      const u = String(url);
      return u.startsWith('http://') || u.startsWith('https://')
        ? !u.startsWith('http://127.0.0.1') && !u.startsWith('http://localhost')
        : false;
    },
    async (route) => {
      const req = route.request();
      const url = req.url();
      const entry = byUrl.get(url);

      if (entry) {
        // Hold a dependent script until its dependency's global is live. This is
        // what removes the smooth-scrollbar / overscroll race deterministically.
        if (entry.waitForGlobal) {
          const frame = req.frame();
          const deadline = Date.now() + 8000;
          while (Date.now() < deadline) {
            const ok = await frame
              .evaluate((g) => typeof window[g] !== 'undefined', entry.waitForGlobal)
              .catch(() => false);
            if (ok) break;
            await new Promise((r) => setTimeout(r, 20));
          }
        }
        await route.fulfill({
          status: 200,
          headers: { 'content-type': entry.contentType, 'cache-control': 'no-store' },
          body: await read(entry.file),
        });
        return;
      }

      if (url.startsWith(FONT_PREFIX)) {
        const f = path.join(NM, 'katex/dist/fonts', path.basename(url));
        const body = await read(f).catch(() => null);
        if (body) {
          await route.fulfill({ status: 200, headers: { 'content-type': 'font/woff2' }, body });
          return;
        }
      }

      const rec = { url, method: req.method(), resourceType: req.resourceType() };
      unmirrored.push(rec);
      if (onUnmirrored) onUnmirrored(rec);
      if (blockUnmirrored) await route.abort('blockedbyclient');
      else await route.continue();
    },
  );

  return { unmirrored };
}

/** SHA256 of each mirrored file — recorded in capture manifests so drift is detectable. */
export async function mirrorFingerprint() {
  const out = [];
  for (const e of CDN_MIRROR) {
    const buf = await fsp.readFile(e.file).catch(() => null);
    out.push({
      url: e.url,
      localFile: path.relative(PKG_ROOT, e.file),
      present: !!buf,
      bytes: buf ? buf.length : 0,
      sha256: buf ? crypto.createHash('sha256').update(buf).digest('hex') : null,
    });
  }
  return out;
}
