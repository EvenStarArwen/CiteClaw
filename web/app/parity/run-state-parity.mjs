#!/usr/bin/env node
/**
 * run-state-parity.mjs — replay parity/states-build.json against BOTH the
 * frozen demo and the rewrite, and compare each settled screenshot.
 *
 * web/parity's own `capture`+`diff` gate the DEFAULT screen at six viewports.
 * This covers the other half of the pilot's brief: every Build state that is
 * reachable without a backend — drawers, popovers, hover paint, selection, the
 * JS-built abstract drill, import mode, the pipeline and config editors.
 *
 *   cd web/app && pnpm dev            # or pnpm preview
 *   node parity/run-state-parity.mjs http://localhost:5273/parity/build
 *
 * Verdicts, matching web/parity's criterion (pixelmatch threshold 0.1):
 *   IDENT  byte-identical PNGs
 *   PASS   0 pixels over threshold (sub-threshold raw delta is reported)
 *   FAIL   anything else. Exit code 1.
 *
 * It reuses web/parity's determinism layer, so both targets get the identical
 * seeded PRNG, frozen clock, virtual warm-up and pixel-stability gate. Do not
 * add a knob here that only one target receives.
 */
import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PARITY = path.resolve(HERE, '../../parity');

const { launchBrowser, newDeterministicContext, openSettled, settleAndShoot, warmUpVirtualClock } =
  await import(path.join(PARITY, 'src/determinism.mjs'));
const { startStaticServer, demoUrl } = await import(path.join(PARITY, 'src/server.mjs'));
const { PNG } = await import(path.join(PARITY, 'node_modules/pngjs/lib/png.js'));
const pixelmatchNs = await import(path.join(PARITY, 'node_modules/pixelmatch/index.js'));
const pixelmatch = pixelmatchNs.default || pixelmatchNs;
// The demo has all seven screens in one document, so every top-bar selector
// matches 3-5 times. Scope everything to the Build page host, on both targets.
const SCOPE = '[data-pg="Build"] ';

const CFG = JSON.parse(fs.readFileSync(path.join(HERE, 'states-build.json'), 'utf8'));
const OUT = path.join(HERE, 'runs', 'states');
fs.mkdirSync(OUT, { recursive: true });
const sha = (b) => crypto.createHash('sha256').update(b).digest('hex').slice(0, 12);
const VP = { width: 1600, height: 900 };

async function run(targetName, url, server) {
  const browser = await launchBrowser();
  const res = {};
  for (const st of CFG.states) {
    const ctx = await newDeterministicContext(browser, VP);
    const { page, consoleErrors } = await openSettled(ctx, url);
    let err = null;
    try {
      for (const step of st.steps) {
        if (step.click) {
          // DOM click, not locator.click(): Playwright's actionability check
          // runs scrollIntoViewIfNeeded, and .sidebar has a latent 14px
          // horizontal overflow (scrollWidth 372 vs clientWidth 358, in the
          // DEMO too) that the sort menu pokes through. Auto-scrolling it
          // shifts the whole panel and is an artifact of the automation, not of
          // either target. Verified: with a DOM click both targets land on the
          // same SHA.
          const n = step.nth ?? 0;
          await page.waitForSelector(SCOPE + step.click, { timeout: 8000 });
          await page.evaluate(([sel, n]) => {
            const el = document.querySelectorAll(sel)[n];
            if (!el) throw new Error('no element ' + sel + ' #' + n);
            el.click();
          }, [SCOPE + step.click, n]);
        } else if (step.hover) {
          await page.locator(SCOPE + step.hover).nth(step.nth ?? 0).hover({ timeout: 8000 });
        } else if (step.fill) {
          await page.locator(SCOPE + step.fill).nth(step.nth ?? 0).fill(step.value, { timeout: 8000 });
        } else if (step.wait) {
          await page.clock.runFor(step.wait);
        }
        await warmUpVirtualClock(page, { virtualMs: 2000 });
      }
      await warmUpVirtualClock(page, { virtualMs: 4000 });
    } catch (e) {
      err = String(e.message).split('\n')[0].slice(0, 140);
    }
    const { buffer, stable } = await settleAndShoot(page);
    const dir = path.join(OUT, targetName);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, `${st.id}-${st.name}.png`), buffer);
    res[st.id] = { sha: sha(buffer), buffer, stable, err, consoleErrors: consoleErrors.slice(0, 2) };
    await ctx.close();
  }
  await browser.close();
  if (server) await server.close();
  return res;
}

const srv = await startStaticServer({ port: 4195 });
const demo = await run('demo', demoUrl(srv.url), srv);
const app = await run('app', process.argv[2] || 'http://localhost:5274/parity/build', null);

let pass = 0, fail = 0;
for (const st of CFG.states) {
  const a = demo[st.id], b = app[st.id];
  let over = 0, raw = 0;
  try {
    const pa = PNG.sync.read(a.buffer), pb = PNG.sync.read(b.buffer);
    over = pixelmatch(pa.data, pb.data, null, pa.width, pa.height, { threshold: 0.1 });
    raw = pixelmatch(pa.data, pb.data, null, pa.width, pa.height, { threshold: 0, includeAA: true });
  } catch (e) { over = -1; }
  const ok = a.sha === b.sha || over === 0;
  ok ? pass++ : fail++;
  console.log(`${ok ? (a.sha === b.sha ? 'IDENT' : 'PASS ') : 'FAIL '} ${st.id} ${st.name.padEnd(30)} demo=${a.sha} app=${b.sha} over=${over} raw=${raw}` +
    (a.err ? ` [demoErr ${a.err}]` : '') + (b.err ? ` [appErr ${b.err}]` : '') +
    (!a.stable || !b.stable ? ' [UNSTABLE]' : '') +
    (a.consoleErrors.length || b.consoleErrors.length ? ` [console ${JSON.stringify(a.consoleErrors)}/${JSON.stringify(b.consoleErrors)}]` : ''));
}
console.log(`\n${pass} pass / ${fail} fail`);
process.exit(fail ? 1 : 0);
