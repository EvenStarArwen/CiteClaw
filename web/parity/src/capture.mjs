// Capture command: render named screens of a target at a set of viewports.

import fsp from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { DETERMINISM, resolveViewports, DESIGN_REFERENCE_DIR } from './config.mjs';
import { startStaticServer } from './server.mjs';
import { launchBrowser, newDeterministicContext, openSettled, settleAndShoot } from './determinism.mjs';
import { mirrorFingerprint } from './cdn-mirror.mjs';
import { runScript } from './actions.mjs';

const sha256 = (buf) => crypto.createHash('sha256').update(buf).digest('hex');

async function loadJson(p) {
  return JSON.parse(await fsp.readFile(p, 'utf8'));
}

/**
 * @param {object} o
 * @param {string} o.target      base URL, or 'design' to serve web/design-reference
 * @param {string} o.screens     path to a screens config JSON
 * @param {string} o.out         output directory for this capture run
 * @param {string} o.viewports   'baseline' | 'sweep' | 'all' | comma list
 */
export async function capture({
  target = 'design',
  screens: screensPath,
  out,
  viewports: viewportSpec = 'all',
  filmstrip = false,
  headless = true,
  label = null,
}) {
  const viewports = resolveViewports(viewportSpec);
  const screensCfg = await loadJson(screensPath);
  const outDir = path.resolve(out);
  await fsp.mkdir(outDir, { recursive: true });
  const scriptDir = path.dirname(path.resolve(screensPath));

  let server = null;
  let baseUrl = target;
  if (target === 'design') {
    server = await startStaticServer({ root: DESIGN_REFERENCE_DIR });
    baseUrl = server.url;
  }

  const browser = await launchBrowser({ headless });
  const runManifest = {
    label: label ?? screensCfg.name ?? path.basename(outDir),
    target,
    baseUrl,
    capturedAt: new Date().toISOString(),
    determinism: DETERMINISM,
    cdnMirror: await mirrorFingerprint(),
    viewports,
    screens: [],
    mirroredExternalRequests: [],
    unmirroredExternalRequests: [],
    consoleErrors: [],
  };

  try {
    for (const screen of screensCfg.screens) {
      const script = screen.script ? await loadJson(path.resolve(scriptDir, screen.script)) : null;

      for (const vp of viewports) {
        const context = await newDeterministicContext(browser, vp);
        const url = new URL(screen.url ?? screensCfg.url ?? '/', baseUrl).toString();
        const { page, network, consoleErrors } = await openSettled(context, url);

        const shotOptions = { fullPage: !!screen.fullPage };
        const screenDir = path.join(outDir, screen.name);
        await fsp.mkdir(screenDir, { recursive: true });

        let finalBuffer;
        let stable;
        let polls;

        if (script) {
          const shots = await runScript(page, script, { shotOptions });
          if (filmstrip) {
            const stripDir = path.join(screenDir, `${vp.name}.steps`);
            await fsp.mkdir(stripDir, { recursive: true });
            for (const s of shots) {
              await fsp.writeFile(path.join(stripDir, `${String(s.index).padStart(2, '0')}-${slug(s.name)}.png`), s.buffer);
            }
          }
          const last = shots[shots.length - 1];
          if (last) ({ buffer: finalBuffer, stable, polls } = last);
        }
        if (!finalBuffer) ({ buffer: finalBuffer, stable, polls } = await settleAndShoot(page, shotOptions));

        const file = path.join(screenDir, `${vp.name}.png`);
        await fsp.writeFile(file, finalBuffer);

        const external = network.filter((r) => /^https?:\/\//.test(r.url) && !r.url.startsWith(baseUrl));
        const unmirrored = context.__parityUnmirrored ?? [];
        runManifest.screens.push({
          screen: screen.name,
          viewport: vp.name,
          width: vp.width,
          height: vp.height,
          fullPage: !!screen.fullPage,
          script: screen.script ?? null,
          file: path.relative(outDir, file),
          bytes: finalBuffer.length,
          sha256: sha256(finalBuffer),
          pixelStable: !!stable,
          settlePolls: polls,
        });
        if (external.length) runManifest.mirroredExternalRequests.push({ screen: screen.name, viewport: vp.name, requests: external });
        if (unmirrored.length) runManifest.unmirroredExternalRequests.push({ screen: screen.name, viewport: vp.name, requests: unmirrored });
        if (consoleErrors.length) runManifest.consoleErrors.push({ screen: screen.name, viewport: vp.name, errors: consoleErrors });

        console.log(
          `[capture] ${screen.name} @ ${vp.name} -> ${path.relative(process.cwd(), file)} ` +
            `(${finalBuffer.length}B, sha ${sha256(finalBuffer).slice(0, 12)}, ${stable ? 'stable' : 'UNSTABLE'} in ${polls} polls)`,
        );
        await context.close();
      }
    }
  } finally {
    await browser.close();
    if (server) await server.close();
  }

  await fsp.writeFile(path.join(outDir, 'run.json'), JSON.stringify(runManifest, null, 2));
  const unstable = runManifest.screens.filter((s) => !s.pixelStable);
  if (unstable.length) {
    console.warn(`[capture] WARNING ${unstable.length} capture(s) never reached pixel stability:`);
    for (const u of unstable) console.warn(`  - ${u.screen} @ ${u.viewport}`);
  }
  return runManifest;
}

export function slug(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}
