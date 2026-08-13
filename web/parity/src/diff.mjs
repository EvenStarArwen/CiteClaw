// Diff command: compare two capture runs and emit side-by-side + heatmap PNGs
// and an HTML report.

import fsp from 'node:fs/promises';
import path from 'node:path';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';
import { DIFF_DEFAULTS } from './config.mjs';
import { renderReport } from './report.mjs';

async function readPng(file) {
  return PNG.sync.read(await fsp.readFile(file));
}

/** Pad an image to (w,h) with transparent pixels so mismatched sizes still diff. */
function padTo(png, w, h) {
  if (png.width === w && png.height === h) return png;
  const out = new PNG({ width: w, height: h });
  out.data.fill(0);
  PNG.bitblt(png, out, 0, 0, Math.min(png.width, w), Math.min(png.height, h), 0, 0);
  return out;
}

/** Compose A | diff | B into one strip with 8px gutters. */
function composeTriptych(a, diff, b) {
  const gutter = 8;
  const w = a.width * 3 + gutter * 2;
  const h = Math.max(a.height, b.height, diff.height);
  const out = new PNG({ width: w, height: h });
  // Mid-grey background so gutters read against both light and dark UIs.
  for (let i = 0; i < out.data.length; i += 4) {
    out.data[i] = 0x9a; out.data[i + 1] = 0x9a; out.data[i + 2] = 0x9a; out.data[i + 3] = 0xff;
  }
  PNG.bitblt(a, out, 0, 0, a.width, a.height, 0, 0);
  PNG.bitblt(diff, out, 0, 0, diff.width, diff.height, a.width + gutter, 0);
  PNG.bitblt(b, out, 0, 0, b.width, b.height, a.width * 2 + gutter * 2, 0);
  return out;
}

/**
 * @param {object} o
 * @param {string} o.a  capture run directory (reference / demo)
 * @param {string} o.b  capture run directory (candidate / rewrite)
 * @param {string} o.out output directory for diff artefacts
 */
export async function diffRuns({ a, b, out, threshold, maxDiffRatio, includeAA } = {}) {
  const cfg = {
    threshold: threshold ?? DIFF_DEFAULTS.threshold,
    maxDiffRatio: maxDiffRatio ?? DIFF_DEFAULTS.maxDiffRatio,
    includeAA: includeAA ?? DIFF_DEFAULTS.includeAA,
  };
  const aDir = path.resolve(a);
  const bDir = path.resolve(b);
  const outDir = path.resolve(out);
  await fsp.mkdir(outDir, { recursive: true });

  const aRun = JSON.parse(await fsp.readFile(path.join(aDir, 'run.json'), 'utf8'));
  const bRun = JSON.parse(await fsp.readFile(path.join(bDir, 'run.json'), 'utf8'));

  const key = (s) => `${s.screen}::${s.viewport}`;
  const bIndex = new Map(bRun.screens.map((s) => [key(s), s]));
  const results = [];

  for (const aShot of aRun.screens) {
    const bShot = bIndex.get(key(aShot));
    const base = { screen: aShot.screen, viewport: aShot.viewport, width: aShot.width, height: aShot.height };

    if (!bShot) {
      results.push({ ...base, status: 'missing', message: 'not present in run B' });
      continue;
    }
    bIndex.delete(key(aShot));

    if (aShot.sha256 === bShot.sha256) {
      results.push({
        ...base, status: 'pass', identical: true, diffPixels: 0, totalPixels: 0, diffRatio: 0,
        sizeMismatch: false, aFile: aShot.file, bFile: bShot.file, diffFile: null, sideBySideFile: null,
      });
      continue;
    }

    let imgA = await readPng(path.join(aDir, aShot.file));
    let imgB = await readPng(path.join(bDir, bShot.file));
    const sizeMismatch = imgA.width !== imgB.width || imgA.height !== imgB.height;
    const w = Math.max(imgA.width, imgB.width);
    const h = Math.max(imgA.height, imgB.height);
    imgA = padTo(imgA, w, h);
    imgB = padTo(imgB, w, h);

    const diff = new PNG({ width: w, height: h });
    const diffPixels = pixelmatch(imgA.data, imgB.data, diff.data, w, h, {
      threshold: cfg.threshold,
      includeAA: cfg.includeAA,
      alpha: 0.25,
      diffColor: [255, 0, 90],
      aaColor: [255, 190, 0],
    });
    // Raw, tolerance-free count: how many pixels differ at ALL.
    const rawDiffPixels = countRawDiff(imgA.data, imgB.data);

    const totalPixels = w * h;
    const diffRatio = diffPixels / totalPixels;
    const stem = `${aShot.screen}__${aShot.viewport}`;
    const diffFile = `${stem}.diff.png`;
    const sbsFile = `${stem}.side-by-side.png`;
    await fsp.writeFile(path.join(outDir, diffFile), PNG.sync.write(diff));
    await fsp.writeFile(path.join(outDir, sbsFile), PNG.sync.write(composeTriptych(imgA, diff, imgB)));

    results.push({
      ...base,
      status: diffRatio <= cfg.maxDiffRatio && !sizeMismatch ? 'pass' : 'fail',
      identical: false,
      diffPixels,
      rawDiffPixels,
      totalPixels,
      diffRatio,
      sizeMismatch,
      aSize: `${aShot.width}x${aShot.height}`,
      bSize: `${bShot.width}x${bShot.height}`,
      aFile: aShot.file,
      bFile: bShot.file,
      diffFile,
      sideBySideFile: sbsFile,
    });
  }

  for (const [, bShot] of bIndex) {
    results.push({
      screen: bShot.screen, viewport: bShot.viewport, status: 'extra', message: 'not present in run A',
    });
  }

  const summary = {
    a: { dir: aDir, label: aRun.label, target: aRun.target, capturedAt: aRun.capturedAt },
    b: { dir: bDir, label: bRun.label, target: bRun.target, capturedAt: bRun.capturedAt },
    config: cfg,
    generatedAt: new Date().toISOString(),
    counts: {
      total: results.length,
      pass: results.filter((r) => r.status === 'pass').length,
      fail: results.filter((r) => r.status === 'fail').length,
      missing: results.filter((r) => r.status === 'missing').length,
      extra: results.filter((r) => r.status === 'extra').length,
      identical: results.filter((r) => r.identical).length,
    },
    results,
  };

  await fsp.writeFile(path.join(outDir, 'diff.json'), JSON.stringify(summary, null, 2));
  await fsp.writeFile(path.join(outDir, 'index.html'), renderReport(summary, { aDir, bDir, outDir }));

  for (const r of results) {
    const tag = r.status.toUpperCase().padEnd(7);
    const detail =
      r.status === 'pass' && r.identical
        ? 'identical'
        : r.diffPixels != null
          ? `${r.diffPixels} px over threshold (${(r.diffRatio * 100).toFixed(4)}%), ${r.rawDiffPixels} raw`
          : r.message ?? '';
    console.log(`[diff] ${tag} ${r.screen} @ ${r.viewport} — ${detail}`);
  }
  console.log(`[diff] report: ${path.join(outDir, 'index.html')}`);
  return summary;
}

function countRawDiff(a, b) {
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    if (a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2] || a[i + 3] !== b[i + 3]) n++;
  }
  return n;
}
