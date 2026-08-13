#!/usr/bin/env node
/**
 * verify-transplants.mjs — assert that everything web/app claims to have copied
 * verbatim out of the design hand-off really is still byte-identical to it.
 *
 * The Build rewrite is a transplant: the screen's CSS, its behaviour class, the
 * shared data modules and the four font stylesheets are the demo's own bytes,
 * not code we wrote. That only stays true if something checks. A single
 * "harmless tidy" inside build-logic.js or one reformatted CSS file is enough
 * to make the parity harness's verdict meaningless while it still reports PASS
 * on a stale baseline.
 *
 *   node scripts/verify-transplants.mjs
 *
 * Exit code 1 on any mismatch. Source of truth is web/design-reference/, which
 * has its own manifest (`node ../parity/bin/parity.mjs verify-reference`) — run
 * that first if you want the whole chain checked.
 */

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const APP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const WEB = path.resolve(APP, '..');
const REF = path.join(WEB, 'design-reference', 'embedded-sources');

const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');
const read = (p) => fs.readFileSync(p);
const lines = (p) => fs.readFileSync(p, 'utf8').split('\n');

let failures = 0;
const ok = (msg) => console.log(`  ok    ${msg}`);
const bad = (msg) => { failures++; console.log(`  FAIL  ${msg}`); };

/** Inner text of a .dc.html's single <helmet><style> block. */
function styleBlock(file) {
  const t = fs.readFileSync(path.join(REF, file), 'utf8');
  const i = t.indexOf('<style>');
  const j = t.indexOf('</style>', i);
  if (i < 0 || j < 0) throw new Error(`no <style> block in ${file}`);
  return t.slice(i + 7, j);
}

console.log('1. Per-screen stylesheets (src/styles/demo-screen-css/)');
for (const [src, dst] of [
  ['Login.dc.html', '01-login.css'],
  ['Home.dc.html', '02-home.css'],
  ['Paper Card.dc.html', '03-paper-card.css'],
  ['Runs.dc.html', '04-runs.css'],
  ['Explore.dc.html', '05-explore.css'],
  ['Settings.dc.html', '06-settings.css'],
  ['System Banners.dc.html', '07-system-banners.css'],
]) {
  const want = styleBlock(src);
  const got = fs.readFileSync(path.join(APP, 'src/styles/demo-screen-css', dst), 'utf8');
  want === got ? ok(`${dst} == ${src} <style> (${want.length} B)`) : bad(`${dst} differs from ${src} <style>`);
}

console.log('2. Build behaviour (src/screens/build/build-logic.js)');
{
  // The transplant is Paper Card.dc.html lines 2365-5874 (the class body, i.e.
  // everything after `class Component extends DCLogic {` and its
  // `rootRef = React.createRef();` field, through the closing brace).
  const want = lines(path.join(REF, 'Paper Card.dc.html')).slice(2364, 5874).join('\n');
  const file = fs.readFileSync(path.join(APP, 'src/screens/build/build-logic.js'), 'utf8');
  const BEGIN = '/* ===== BEGIN VERBATIM TRANSPLANT (Paper Card.dc.html 2365-5874) ===== */\n';
  const END = '/* ===== END VERBATIM TRANSPLANT ===== */';
  const i = file.indexOf(BEGIN), j = file.indexOf(END);
  if (i < 0 || j < 0) {
    bad('build-logic.js: BEGIN/END transplant markers missing');
  } else {
    const got = file.slice(i + BEGIN.length, j).replace(/\n$/, '');
    got === want
      ? ok(`build-logic.js body == Paper Card.dc.html 2365-5874 (${want.length} B, 164 methods)`)
      : bad(`build-logic.js body differs from Paper Card.dc.html 2365-5874 (${got.length} B vs ${want.length} B)`);
  }
}

console.log('3. Shared demo modules (src/design-data/*.js)');
for (const f of [
  'run-mock.js', 'topic-data.js', 'graph-data.js', 'community-data.js',
  'citation-context.js', 'paper-row.js', 'review-draft.js', 'topic-desc.js',
  'import-resolver.js',
]) {
  const a = sha(read(path.join(REF, f)));
  const b = sha(read(path.join(APP, 'src/design-data', f)));
  a === b ? ok(`${f} ${a.slice(0, 12)}`) : bad(`${f}: ${a.slice(0, 12)} (reference) != ${b.slice(0, 12)} (app copy)`);
}

console.log('4. Shell CSS (src/styles/ipad-demo-shell.css)');
{
  const want = styleBlock('KnowledgeLab iPad Demo.dc.html');
  const got = fs.readFileSync(path.join(APP, 'src/styles/ipad-demo-shell.css'), 'utf8');
  want === got ? ok(`ipad-demo-shell.css == shell <style> (${want.length} B)`) : bad('ipad-demo-shell.css differs from the shell <style> block');
}

console.log('5. Self-hosted fonts (public/design-fonts/)');
{
  // These cannot be re-derived from the .dc.html sources: the demo's bundler
  // inlined the four Google Fonts stylesheets (woff2 payloads as data: URIs)
  // into <style> blocks. They are pinned by size + sha256 recorded when they
  // were extracted from the running demo (head style indices 5, 8, 11, 15).
  const expected = {
    'newsreader-roman__hanken-variable.css': 1074456,
    'newsreader-roman__hanken-fixed.css': 1251222,
    'newsreader-italic__hanken-fixed.css': 1974796,
    'hanken-variable.css': 88407,
  };
  for (const [f, size] of Object.entries(expected)) {
    const p = path.join(APP, 'public/design-fonts', f);
    if (!fs.existsSync(p)) { bad(`${f} missing`); continue; }
    const n = fs.statSync(p).size;
    n === size ? ok(`${f} ${n} B`) : bad(`${f}: ${n} B, expected ${size} B`);
  }
}

console.log(failures ? `\n${failures} transplant(s) have drifted from the design hand-off.` : '\nAll transplants verified.');
process.exit(failures ? 1 : 0);
