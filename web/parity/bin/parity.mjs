#!/usr/bin/env node
// Parity harness CLI.
//
//   parity serve   [--port 4173] [--root <dir>]
//   parity capture --screens <cfg.json> --out <dir> [--target design|<url>] [--viewports baseline|sweep|all|<names>] [--filmstrip] [--label X]
//   parity diff    --a <runDir> --b <runDir> --out <dir> [--threshold 0.1] [--max-diff-ratio 0] [--include-aa]
//   parity script  --script <script.json> --out <dir> [--target design|<url>] [--url <page>] [--viewport <name>]
//   parity sanity  [--screens <cfg.json>] [--viewports ...]   # capture the demo twice, expect ZERO diff
//
// Exit codes: 0 ok, 1 failure (diff gate not met / unstable capture), 2 usage error.

import path from 'node:path';
import fsp from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { PKG_ROOT, DESIGN_REFERENCE_DIR, resolveViewports } from '../src/config.mjs';
import { startStaticServer, demoUrl } from '../src/server.mjs';
import { capture } from '../src/capture.mjs';
import { diffRuns } from '../src/diff.mjs';
import { launchBrowser, newDeterministicContext, openSettled } from '../src/determinism.mjs';
import { runScript } from '../src/actions.mjs';

function parseArgs(argv) {
  const cmd = argv[2];
  const flags = {};
  const positional = [];
  for (let i = 3; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const eq = a.indexOf('=');
      if (eq > -1) flags[a.slice(2, eq)] = a.slice(eq + 1);
      else if (argv[i + 1] && !argv[i + 1].startsWith('--')) flags[a.slice(2)] = argv[++i];
      else flags[a.slice(2)] = true;
    } else positional.push(a);
  }
  return { cmd, flags, positional };
}

const DEFAULT_SCREENS = path.join(PKG_ROOT, 'screens', 'design-demo.json');

async function main() {
  const { cmd, flags } = parseArgs(process.argv);

  switch (cmd) {
    case 'serve': {
      const server = await startStaticServer({
        root: flags.root ? path.resolve(flags.root) : DESIGN_REFERENCE_DIR,
        port: Number(flags.port ?? 4173),
        quiet: false,
      });
      console.log(`[serve] ${server.root}`);
      console.log(`[serve] ${server.url}`);
      console.log(`[serve] demo: ${demoUrl(server.url)}`);
      console.log('[serve] Ctrl-C to stop');
      return new Promise(() => {});
    }

    case 'capture': {
      const run = await capture({
        target: flags.target ?? 'design',
        screens: path.resolve(flags.screens ?? DEFAULT_SCREENS),
        out: path.resolve(flags.out ?? path.join(PKG_ROOT, 'runs', `capture-${Date.now()}`)),
        viewports: flags.viewports ?? 'all',
        filmstrip: !!flags.filmstrip,
        headless: flags.headed ? false : true,
        label: flags.label ?? null,
      });
      const unstable = run.screens.filter((s) => !s.pixelStable);
      process.exitCode = unstable.length ? 1 : 0;
      return;
    }

    case 'diff': {
      if (!flags.a || !flags.b) throw usage('diff requires --a and --b');
      const summary = await diffRuns({
        a: flags.a,
        b: flags.b,
        out: path.resolve(flags.out ?? path.join(PKG_ROOT, 'runs', `diff-${Date.now()}`)),
        threshold: flags.threshold != null ? Number(flags.threshold) : undefined,
        maxDiffRatio: flags['max-diff-ratio'] != null ? Number(flags['max-diff-ratio']) : undefined,
        includeAA: flags['include-aa'] ? true : undefined,
      });
      const bad = summary.counts.fail + summary.counts.missing + summary.counts.extra;
      process.exitCode = bad ? 1 : 0;
      return;
    }

    case 'script': {
      if (!flags.script) throw usage('script requires --script');
      const scriptPath = path.resolve(flags.script);
      const script = JSON.parse(await fsp.readFile(scriptPath, 'utf8'));
      const outDir = path.resolve(flags.out ?? path.join(PKG_ROOT, 'runs', `script-${Date.now()}`));
      await fsp.mkdir(outDir, { recursive: true });

      let server = null;
      let baseUrl = flags.target ?? 'design';
      if (baseUrl === 'design') {
        server = await startStaticServer({ root: DESIGN_REFERENCE_DIR });
        baseUrl = server.url;
      }
      const vp = resolveViewports(flags.viewport ?? 'desktop-1600x900')[0];
      const browser = await launchBrowser({ headless: flags.headed ? false : true });
      try {
        const context = await newDeterministicContext(browser, vp);
        const target = new URL(flags.url ?? script.url ?? `/${encodeURIComponent('KnowledgeLab iPad Demo.html')}`, baseUrl).toString();
        const { page, consoleErrors } = await openSettled(context, target);
        const shots = await runScript(page, script, { shotOptions: { fullPage: !!flags['full-page'] } });
        for (const s of shots) {
          const file = path.join(outDir, `${String(s.index).padStart(2, '0')}-${s.name.replace(/[^a-z0-9]+/gi, '-')}.png`);
          await fsp.writeFile(file, s.buffer);
          console.log(`[script] ${s.action.padEnd(12)} -> ${path.relative(process.cwd(), file)}${s.stable ? '' : '  (UNSTABLE)'}`);
        }
        if (consoleErrors.length) console.warn('[script] console errors:', consoleErrors.slice(0, 5));
        await context.close();
      } finally {
        await browser.close();
        if (server) await server.close();
      }
      return;
    }

    case 'sanity': {
      // ACCEPTANCE TEST: capturing the ORIGINAL demo twice must produce ZERO diff.
      const screens = path.resolve(flags.screens ?? DEFAULT_SCREENS);
      const viewports = flags.viewports ?? 'all';
      const stamp = Date.now();
      const base = path.join(PKG_ROOT, 'runs', `sanity-${stamp}`);
      const aDir = path.join(base, 'A');
      const bDir = path.join(base, 'B');
      console.log('[sanity] pass 1/2 (demo)');
      await capture({ target: 'design', screens, out: aDir, viewports, label: 'demo-pass-A' });
      console.log('[sanity] pass 2/2 (demo, fresh browser + fresh context)');
      await capture({ target: 'design', screens, out: bDir, viewports, label: 'demo-pass-B' });
      const summary = await diffRuns({ a: aDir, b: bDir, out: path.join(base, 'diff'), maxDiffRatio: 0 });
      const ok = summary.counts.fail === 0 && summary.counts.missing === 0 && summary.counts.extra === 0;
      const allIdentical = summary.counts.identical === summary.counts.total;
      console.log(`\n[sanity] ${ok ? 'PASS' : 'FAIL'} — ${summary.counts.identical}/${summary.counts.total} byte-identical`);
      console.log(`[sanity] report: ${path.join(base, 'diff', 'index.html')}`);
      process.exitCode = ok && allIdentical ? 0 : 1;
      return;
    }

    case 'verify-reference': {
      const { verifyManifest } = await import('../src/reference-manifest.mjs');
      const r = await verifyManifest();
      if (r.ok) {
        console.log('[verify-reference] OK — web/design-reference matches its manifest.');
      } else {
        console.error('[verify-reference] DRIFT DETECTED in web/design-reference:');
        for (const f of r.changed) console.error(`  changed: ${f}`);
        for (const f of r.missing) console.error(`  missing: ${f}`);
        for (const f of r.added) console.error(`  added:   ${f}`);
        console.error('These files are a frozen verbatim copy of the design hand-off and must not be edited.');
      }
      process.exitCode = r.ok ? 0 : 1;
      return;
    }

    default:
      throw usage(cmd ? `unknown command "${cmd}"` : 'no command given');
  }
}

function usage(msg) {
  const e = new Error(msg);
  e.usage = true;
  return e;
}

main().catch((err) => {
  if (err && err.usage) {
    console.error(`parity: ${err.message}\n`);
    console.error(fsp.readFile ? 'See web/parity/README.md for usage.' : '');
    process.exit(2);
  }
  console.error(err);
  process.exit(1);
});

void fileURLToPath;
