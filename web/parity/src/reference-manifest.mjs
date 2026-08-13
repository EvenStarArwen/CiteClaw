// Build / verify the design-reference manifest.
//
// web/design-reference is a FROZEN, VERBATIM copy of the design workspace. The
// manifest records a SHA256 for every file so that accidental edits (or a
// refreshed hand-off) are detectable rather than silent — a parity gate is
// worthless if the reference it compares against can drift.

import fsp from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { DESIGN_REFERENCE_DIR } from './config.mjs';

export const MANIFEST_FILE = 'manifest.json';

async function walk(dir, base = dir, acc = []) {
  for (const entry of (await fsp.readdir(dir, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) await walk(abs, base, acc);
    else if (entry.isFile() && entry.name !== MANIFEST_FILE) acc.push(path.relative(base, abs));
  }
  return acc;
}

async function hashFile(abs) {
  const buf = await fsp.readFile(abs);
  return { bytes: buf.length, sha256: crypto.createHash('sha256').update(buf).digest('hex') };
}

/** @returns {Promise<object>} the manifest object (also written to disk when `write`) */
export async function buildManifest({ root = DESIGN_REFERENCE_DIR, sourceDir, extra = {}, write = true } = {}) {
  const files = await walk(root);
  const entries = [];
  for (const rel of files) {
    const { bytes, sha256 } = await hashFile(path.join(root, rel));
    let sourceSha = null;
    if (sourceDir) {
      const src = path.join(sourceDir, extra.sourceMap?.[rel] ?? rel);
      sourceSha = await hashFile(src).then((h) => h.sha256).catch(() => null);
    }
    entries.push({ path: rel, bytes, sha256, matchesSource: sourceSha == null ? null : sourceSha === sha256 });
  }
  const manifest = {
    generatedAt: new Date().toISOString(),
    sourceDir: sourceDir ?? null,
    fileCount: entries.length,
    totalBytes: entries.reduce((n, e) => n + e.bytes, 0),
    ...extra,
    files: entries,
  };
  if (write) await fsp.writeFile(path.join(root, MANIFEST_FILE), JSON.stringify(manifest, null, 2) + '\n');
  return manifest;
}

/** @returns {Promise<{ok: boolean, changed: string[], missing: string[], added: string[]}>} */
export async function verifyManifest({ root = DESIGN_REFERENCE_DIR } = {}) {
  const manifest = JSON.parse(await fsp.readFile(path.join(root, MANIFEST_FILE), 'utf8'));
  const onDisk = new Set(await walk(root));
  const changed = [];
  const missing = [];
  for (const e of manifest.files) {
    if (!onDisk.has(e.path)) { missing.push(e.path); continue; }
    onDisk.delete(e.path);
    const { sha256 } = await hashFile(path.join(root, e.path));
    if (sha256 !== e.sha256) changed.push(e.path);
  }
  return { ok: !changed.length && !missing.length && !onDisk.size, changed, missing, added: [...onDisk] };
}
