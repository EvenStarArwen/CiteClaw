// paper-row.js — the ONE paper-card contract for KnowledgeLab.
// (also enables CSS :active pressed states on iOS Safari, which needs a touch listener)
if (typeof document !== 'undefined' && !document.__klTouchActive) {
  document.__klTouchActive = 1;
  document.addEventListener('touchstart', function () {}, { passive: true });
}
// Build / Runs / Explore all render their .pr list rows through this, so the
// card anatomy (title, icon-meta venue + author/cite/year rows, selection rail)
// lives in a single place instead of being re-invented per page.
//
// Global (classic script, loaded once via <script src> in each page's helmet):
//   window.KLPaperRow = { PR_ICONS, prCardInner(p, opts), prEtAl, prKfmt, prPdf }
//
// A paper object: { title, authors, venue, year, cites, pdf?, pdfSrc? }.
//
// STAR CONTRACT (see design-system.md § Star / save control). Three rules, and
// breaking any one of them is how the star silently disappeared twice:
//   1. prCardInner is the ONLY place that BUILDS a star (`star:true`). Callers
//      wire behaviour onto `.pr-star`; they never create, clone or re-append one.
//   2. Move it by SEMANTIC anchor only: `.pr-meta` (the icon meta line) or the
//      row itself. Never `row.querySelector('div')` — that positional selector
//      landed the star inside the line-clamped `.pr-title` and it got clipped.
//   3. Visibility is CSS state (`.pr:hover .pr-star`, `[data-saved="1"]`), never
//      an inline opacity toggled from JS. This markup therefore sets NO opacity.
// pdf: full text available (S2 open-access). When the field is absent, prPdf()
// derives a stable demo value from the paper's id/title so every surface agrees.
// pdfSrc:'user' = full text came from the user's own import (tooltip says so).
// opts:
//   railClass  selection-bar class the page wires (default 'rail')
//   lead       html injected before the venue label (e.g. Explore T## chip)
//   venueColor CSS color for the landmark icon + venue caps (default --muted2)
//   etAl       collapse authors to "Surname et al." (default false → full list)
//   topRight   control html placed in a flex row beside the title (Runs row verbs,
//              add-panel pills) — it reserves its own width, never overlaps
//   botRight   control html pinned to the card's bottom-right corner (Runs row verbs)
//   foot       html appended under the meta block (Runs step chip + stats)
(function () {
  var PR_ICONS = {
    user: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><circle cx="12" cy="8" r="4"></circle><path d="M4 21c1.4-3.6 4.4-5.5 8-5.5s6.6 1.9 8 5.5"></path></svg>',
    cal: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><rect x="3" y="5" width="18" height="16" rx="2"></rect><path d="M8 3v4M16 3v4M3 10h18"></path></svg>',
    cite: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><path d="M7 17l-3-6V5h6v6H7zM17 17l-3-6V5h6v6h-3z"></path></svg>',
    ven: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>',
    pdf: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><rect x="4" y="11" width="16" height="10" rx="2"></rect><path d="M8 11V7a4 4 0 0 1 7.5-1.9"></path></svg>',
    clip: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18.4 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>',
    alert: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><circle cx="12" cy="12" r="9"></circle><path d="M12 8v4"></path><path d="M12 16h.01"></path></svg>',
    list: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><path d="M8 6h13"></path><path d="M8 12h13"></path><path d="M8 18h13"></path><path d="M3 6h.01"></path><path d="M3 12h.01"></path><path d="M3 18h.01"></path></svg>'
  };
  function esc(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'); }
  function prAuthors(a) {
    var s = String(a == null ? '' : a), parts = s.split(' \u00b7 ');
    return parts.length < 2 ? s : parts.join(s.indexOf(',') >= 0 ? '; ' : ', ');
  }
  function prEtAl(a) {
    var parts = String(a || '').split(' \u00b7 '), first = (parts[0] || '').trim();
    return parts.length > 1 ? first.split(' ').pop() + ' et al.' : first;
  }
  function prKfmt(n) {
    if (n == null) return '';
    if (n < 1000) return String(n);
    if (n < 999500) { var v = n / 1000; return (v < 10 ? v.toFixed(1).replace(/\.0$/, '') : String(Math.round(v))) + 'k'; }
    var m = n / 1e6; return (m < 10 ? m.toFixed(1).replace(/\.0$/, '') : String(Math.round(m))) + 'M';
  }
  function prPdf(p) {
    if (!p) return false;
    if (p.pdf !== undefined) return !!p.pdf;
    var s = String(p.id || p.title || ''), h = 0;
    for (var i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
    return (Math.abs(h) % 100) < 62;
  }
  function pdfPair(p) {
    // Presence-only (absence is a list's default state), but SALIENT where present — this is
    // an access signal, not a link: it says whether the agent can read the full text.
    // Nature's Open Access treatment: an open padlock + a green label that scans down a column.
    if (!prPdf(p)) return '';
    var tt = p.pdfSrc === 'user' ? 'Full text\u00a0\u00a0from your import: the agent can read this paper in depth' : 'Full text available: the agent can read this paper in depth';
    return '<span title="' + esc(tt) + '" style="display:inline-flex; align-items:center; gap:4px; flex:none; font-weight:600; color:color-mix(in oklab, var(--success) 72%, var(--fg));">' +
      PR_ICONS.pdf + '<span style="white-space:nowrap;">Full text</span></span>';
  }
  function pair(ic, tx, grow, tt, strong) {
    // no title on the authors run: a full-author-list tooltip over a card is noise
    return '<span' + (tt && !grow ? ' title="' + esc(tt) + '"' : '') + ' style="display:inline-flex; align-items:center; gap:4px; color:var(--muted); ' + (grow ? 'flex:1; min-width:0; overflow:hidden;' : 'flex:none;') + '">' +
      '<span style="color:var(--muted2); display:inline-flex; flex:none;">' + ic + '</span>' +
      '<span' + (grow ? ' class="pr-authors"' : '') + ' style="' + (grow ? 'white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0;' : 'white-space:nowrap;') + ' font-variant-numeric:tabular-nums;' + (strong ? ' color:var(--fg); font-weight:600;' : '') + '">' + tx + '</span></span>';
  }
  function prCardInner(p, opts) {
    opts = opts || {};
    var railClass = opts.railClass || 'rail';
    var venueColor = opts.venueColor || 'var(--muted2)';
    var authors = opts.etAl ? prEtAl(p.authors) : prAuthors(p.authors);
    var rail = '<span class="' + railClass + '" style="position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--accent); opacity:0; transition:opacity .16s ease;"></span>';
    // A topRight control shares a flex row with the title instead of floating over it, so
    // the title can never collide with it at ANY control width (⋯ glyph or a text pill).
    var title = '<div class="pr-title" style="font-family:\'Newsreader\',serif; font-size:16.5px; line-height:1.32; font-weight:500; color:var(--fg); transition:color .16s ease; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;' + (opts.topRight ? ' flex:1; min-width:0;' : '') + '">' + esc(p.title) + '</div>';
    if (opts.topRight) title = '<div style="display:flex; align-items:flex-start; gap:10px;">' + title +
      '<div style="flex:none; margin-top:1px; position:relative; z-index:2;">' + opts.topRight + '</div></div>';
    // emphasis: how much weight the venue/meta band carries. NO color system and no
    // accent here — Build/Runs stay ink-only (topic color belongs to Explore).
    var em = opts.emphasis || 'Venue ink';
    var chip = em === 'Venue chip';
    var strongN = em === 'Ink + strong numbers';
    if (!opts.venueColor && em !== 'Muted') venueColor = 'var(--fg2)';
    // venue is caps + its own ink already; bold on top made the whole band shout
    var venLabel = '<span class="pr-venue" title="' + esc(p.venue) + '" style="' + (chip ? 'display:inline-block; max-width:100%; padding:2px 7px; border-radius:5px; background:var(--card2); border:1px solid var(--border); ' : '') + 'flex:1; min-width:0; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">' + esc(p.venue) + '</span>';
    var venRow = '<div style="display:flex; align-items:center; gap:6px; font-size:11.5px; letter-spacing:.03em; text-transform:uppercase; color:' + venueColor + ';">' +
      (opts.lead || '') +
      '<span style="display:inline-flex; flex:none;">' + PR_ICONS.ven + '</span>' + venLabel + '</div>';
    var metaRow = '<div class="pr-meta" style="display:flex; align-items:center; gap:11px; font-size:11.5px; margin-top:' + (chip ? '5px' : '4px') + ';">' +
      pair(PR_ICONS.user, esc(authors), true, p.authors) +
      pair(PR_ICONS.cite, prKfmt(p.cites), false, 'Citations', strongN) +
      pair(PR_ICONS.cal, p.year, false, null, strongN) + pdfPair(p) + '</div>';
    var star = opts.star ? '<button class="pr-star" aria-label="Save paper" style="position:absolute; right:14px; top:50%; transform:translateY(-50%); background:none; border:none; padding:6px; margin:0; cursor:pointer; display:flex; border-radius:8px; z-index:2; transition:opacity .16s ease, background .16s ease;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--star)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M11.48 3.5a.5.5 0 0 1 1 0l2.4 5.05 5.5.7a.5.5 0 0 1 .28.86l-4.05 3.8 1.05 5.5a.5.5 0 0 1-.74.53L12 17.7l-4.9 2.54a.5.5 0 0 1-.73-.53l1.05-5.5-4.05-3.8a.5.5 0 0 1 .28-.86l5.5-.7z"></path></svg></button>' : '';
    return rail + star +
      (opts.botRight ? '<div style="position:absolute; right:12px; bottom:13px; z-index:2;">' + opts.botRight + '</div>' : '') +
      title +
      '<div style="margin-top:7px; display:flex; flex-direction:column; gap:4px;">' + venRow + metaRow + '</div>' +
      (opts.foot || '');
  }
  // COMPACT paper card (pickers, drill-in lists, import review): the prCardInner
  // anatomy at list density — 14px serif title (2-line clamp) + ONE 11px icon meta
  // line. Fields are opt-in per surface; the venue pair is caps and shrinks first.
  // Lead/right controls (star, rank, pill, value column) belong to the CALLER's row
  // chrome; this builds only the shared body so surfaces cannot diverge on it.
  // opts: venue/cites/year (default on), authors (off; etAl by default), venueColor.
  function prMiniInner(p, opts) {
    opts = opts || {};
    var vc = opts.venueColor || null;
    var bit = function (ic, tx, o) {
      o = o || {};
      return '<span style="display:inline-flex; align-items:center; gap:4px; min-width:0;' + (o.grow ? ' flex:0 1 auto; overflow:hidden;' : ' flex:none;') + (o.color ? ' color:' + o.color + ';' : '') + '">' +
        '<span style="color:' + (o.color || 'var(--muted2)') + '; display:inline-flex; flex:none;">' + ic + '</span>' +
        '<span style="white-space:nowrap;' + (o.grow ? ' overflow:hidden; text-overflow:ellipsis;' : '') + (o.caps ? ' letter-spacing:.04em; text-transform:uppercase;' : ' font-variant-numeric:tabular-nums;') + '">' + tx + '</span></span>';
    };
    var bits = [];
    if (opts.venue !== false) bits.push(bit(PR_ICONS.ven, esc(p.venue || '\u2013'), { grow: true, caps: true, color: vc }));
    if (opts.authors) bits.push(bit(PR_ICONS.user, esc(opts.etAl === false ? prAuthors(p.authors) : prEtAl(p.authors)), { grow: true }));
    if (opts.cites !== false) bits.push(bit(PR_ICONS.cite, prKfmt(p.cites || 0)));
    if (opts.year !== false) bits.push(bit(PR_ICONS.cal, p.year || '\u2013'));
    return '<div class="prm-title" style="font-family:\'Newsreader\',serif; font-size:14px; line-height:1.32; font-weight:500; color:var(--fg); display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;">' + esc(p.title) + '</div>' +
      '<div class="prm-meta" style="display:flex; align-items:center; gap:10px; min-width:0; font-size:11px; color:var(--muted);">' + bits.join('') + '</div>';
  }
  // canvas hover tooltip — same anatomy/icons as the card, tighter type
  function prTipInner(p, opts) {
    opts = opts || {};
    var venueColor = opts.venueColor || 'var(--fg2)';
    var authors = opts.etAl === false ? prAuthors(p.authors) : prEtAl(p.authors);
    return '<div style="display:flex; align-items:center; gap:6px; font-size:10.5px; letter-spacing:.05em; text-transform:uppercase; color:' + venueColor + ';">' +
        (opts.lead || '') +
        '<span style="display:inline-flex; flex:none;">' + PR_ICONS.ven + '</span>' +
        '<span style="flex:1; min-width:0; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">' + esc(opts.venueLabel || p.venue) + '</span></div>' +
      '<div style="margin-top:4px; font-family:\'Newsreader\',serif; font-size:13.5px; line-height:1.35; font-weight:500; letter-spacing:0; color:var(--fg); display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; text-wrap:pretty;">' + esc(p.title) + '</div>' +
      '<div style="display:flex; align-items:center; gap:11px; font-size:11.5px; margin-top:5px;">' +
        pair(PR_ICONS.user, esc(authors), true, p.authors) +
        pair(PR_ICONS.cite, prKfmt(p.cites), false, 'Citations') +
        pair(PR_ICONS.cal, p.year, false) + pdfPair(p) + '</div>';
  }
  // Scrollbar-lane probe: expose the platform's real thin-scrollbar width as --sbw on
  // <html> (0 on overlay-scrollbar platforms like iPadOS; ~8px classic). Pages derive
  // --pin-sc from it so panel-edge insets stay symmetric on BOTH platforms.
  (function () {
    var probe = function () {
      try {
        var d = document.createElement('div');
        d.style.cssText = 'position:absolute; visibility:hidden; width:100px; height:100px; overflow:scroll; scrollbar-width:thin;';
        document.body.appendChild(d);
        var w = d.offsetWidth - d.clientWidth;
        d.remove();
        document.documentElement.style.setProperty('--kl-sbw', Math.max(0, Math.min(w, 8)) + 'px');
      } catch (e) {}
    };
    if (document.body) probe(); else document.addEventListener('DOMContentLoaded', probe);
  })();
  window.KLPaperRow = { PR_ICONS: PR_ICONS, prCardInner: prCardInner, prMiniInner: prMiniInner, prTipInner: prTipInner, prEtAl: prEtAl, prKfmt: prKfmt, prPdf: prPdf, prAuthors: prAuthors };
  try { document.dispatchEvent(new CustomEvent('kl-paper-row-ready')); } catch (e) {}
})();
