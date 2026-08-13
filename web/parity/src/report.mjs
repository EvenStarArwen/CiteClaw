// Self-contained HTML report for a diff run. No external assets: it must open
// from disk on any machine, including one with no network.

import path from 'node:path';

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

export function renderReport(summary, { aDir, bDir, outDir }) {
  const rel = (from, file) => (file ? esc(path.relative(outDir, path.join(from, file))) : null);
  const c = summary.counts;
  const verdict = c.fail + c.missing + c.extra === 0 ? 'PASS' : 'FAIL';

  const rows = summary.results
    .map((r) => {
      const badge = `<span class="badge ${r.status}">${r.status}</span>`;
      const numbers =
        r.diffPixels != null
          ? `<td class="num">${r.diffPixels.toLocaleString()}</td>
             <td class="num">${(r.rawDiffPixels ?? 0).toLocaleString()}</td>
             <td class="num">${(r.diffRatio * 100).toFixed(4)}%</td>`
          : `<td class="num">&mdash;</td><td class="num">&mdash;</td><td class="num">&mdash;</td>`;
      const note = r.sizeMismatch ? `<div class="warn">size mismatch: A ${esc(r.aSize)} vs B ${esc(r.bSize)}</div>` : '';
      const detail = r.sideBySideFile
        ? `<details><summary>side-by-side &amp; heatmap</summary>
             <div class="legend">left: A (${esc(summary.a.label)}) &middot; middle: diff heatmap &middot; right: B (${esc(summary.b.label)})</div>
             <img loading="lazy" src="${esc(r.sideBySideFile)}" alt="side by side">
           </details>`
        : r.identical
          ? '<span class="muted">byte-identical</span>'
          : `<span class="muted">${esc(r.message ?? '')}</span>`;
      return `<tr class="${r.status}">
        <td>${badge}</td>
        <td><code>${esc(r.screen)}</code></td>
        <td><code>${esc(r.viewport)}</code></td>
        ${numbers}
        <td>${note}${detail}</td>
      </tr>`;
    })
    .join('\n');

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parity report — ${esc(summary.a.label)} vs ${esc(summary.b.label)}</title>
<style>
  :root { color-scheme: light dark; --bg:#faf8f4; --fg:#1c1a17; --muted:#6f6a61; --line:#e2ddd3; --card:#fff; }
  @media (prefers-color-scheme: dark) { :root { --bg:#16150f; --fg:#ece5d8; --muted:#9a9386; --line:#33302a; --card:#1e1c16; } }
  * { box-sizing: border-box; }
  body { margin:0; padding:24px; background:var(--bg); color:var(--fg);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--muted); margin-bottom:20px; }
  .verdict { display:inline-block; padding:6px 14px; border-radius:999px; font-weight:700; letter-spacing:.04em; }
  .verdict.PASS { background:#1f7a4d; color:#fff; }
  .verdict.FAIL { background:#b3261e; color:#fff; }
  .cards { display:flex; flex-wrap:wrap; gap:10px; margin:16px 0 24px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 16px; min-width:96px; }
  .card b { display:block; font-size:20px; }
  .card span { color:var(--muted); font-size:12px; }
  .wrap { overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:var(--card); }
  table { border-collapse:collapse; width:100%; min-width:840px; }
  th,td { text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  tr.fail { background:rgba(179,38,30,.07); }
  .badge { font-size:11px; font-weight:700; text-transform:uppercase; padding:2px 8px; border-radius:999px; }
  .badge.pass { background:#1f7a4d; color:#fff; }
  .badge.fail { background:#b3261e; color:#fff; }
  .badge.missing, .badge.extra { background:#8a6d1f; color:#fff; }
  .muted { color:var(--muted); }
  .warn { color:#b3261e; font-weight:600; margin-bottom:6px; }
  .legend { color:var(--muted); font-size:12px; margin:8px 0; }
  img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:6px; }
  details summary { cursor:pointer; color:var(--muted); }
  code { font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; }
  footer { margin-top:20px; color:var(--muted); font-size:12px; }
</style></head><body>
<h1>Pixel parity report</h1>
<div class="sub">
  <span class="verdict ${verdict}">${verdict}</span>
  &nbsp; A = <code>${esc(summary.a.label)}</code> (${esc(summary.a.target)}) &nbsp;vs&nbsp;
  B = <code>${esc(summary.b.label)}</code> (${esc(summary.b.target)})<br>
  threshold ${summary.config.threshold} &middot; includeAA ${summary.config.includeAA} &middot;
  pass gate diffRatio &le; ${summary.config.maxDiffRatio} &middot; generated ${esc(summary.generatedAt)}
</div>
<div class="cards">
  <div class="card"><b>${c.total}</b><span>screens</span></div>
  <div class="card"><b>${c.pass}</b><span>pass</span></div>
  <div class="card"><b>${c.fail}</b><span>fail</span></div>
  <div class="card"><b>${c.identical}</b><span>byte-identical</span></div>
  <div class="card"><b>${c.missing + c.extra}</b><span>missing / extra</span></div>
</div>
<div class="wrap"><table>
<thead><tr>
  <th>status</th><th>screen</th><th>viewport</th>
  <th class="num">diff px<br>(thresholded)</th><th class="num">diff px<br>(raw)</th><th class="num">% of frame</th>
  <th>evidence</th>
</tr></thead>
<tbody>
${rows}
</tbody></table></div>
<footer>
  A run: <code>${esc(aDir)}</code><br>
  B run: <code>${esc(bDir)}</code><br>
  &ldquo;raw&rdquo; counts every pixel whose RGBA differs at all; &ldquo;thresholded&rdquo; applies the
  anti-aliasing tolerance and is what the pass/fail gate uses.
</footer>
</body></html>`;
}
