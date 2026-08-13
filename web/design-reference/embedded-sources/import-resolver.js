// import-resolver.js — KnowledgeLab's shared bulk-import mock (see design-system.md § Import papers).
// One flow, many parsers: extract references → resolve on Semantic Scholar → match-review → add.
// Consumers (Home wizard step 2, Runs Add-papers panel) own the DOM/wiring; this file owns the
// deterministic mock parse + the match-review row/section builders so the two surfaces can't diverge.
// window.KLImport = { parse(files), sample(), groups(entries, dupeChip), rowHtml(e, i, opts), groupHtml(g, rowsArr, opts), candPopHtml(e), fileRowHtml(f) }
(function () {
  function esc(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'); }
  function hash(s) { var h = 0; s = String(s); for (var i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0; return Math.abs(h); }
  function rng(seed) { var s = seed || 1; return function () { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; }
  var TI = [
    'Adaptive weight vectors for decomposition-based multi-objective optimization',
    'A survey of surrogate-assisted evolutionary algorithms for expensive problems',
    'Benchmarking many-objective optimizers on real-world engineering suites',
    'Constraint handling in decomposition frameworks: a unified view',
    'Pareto front estimation with learned scalarizing functions',
    'Neighborhood adaptation strategies for large-scale variable interaction',
    'On the convergence of reference-point methods under noisy evaluations',
    'Transfer optimization across related multi-objective tasks',
    'A knee-point driven algorithm for preference-based optimization',
    'Diversity maintenance beyond crowding distance: a systematic study',
    'Hybridizing local search with decomposition for combinatorial problems',
    'Scalable indicator-based selection with incremental hypervolume updates',
    'Dynamic multi-objective optimization with change-severity detection',
    'Weight vector generation on irregular Pareto fronts',
    'An empirical study of mating restriction in decomposition algorithms',
    'Bayesian preference elicitation for interactive optimization',
    'Robust optimization under decision-space perturbations',
    'Archive strategies for unbounded external populations',
    'Multi-task evolutionary optimization with shared representations',
    'Expensive constraint evaluation and feasibility-first ranking',
    'A parallel island model for decomposition-based search',
    'Objective reduction via correlation-aware subspace selection',
    'Warm-starting evolutionary search from historical runs',
    'Landscape analysis features that predict algorithm performance',
    'Termination criteria for anytime multi-objective optimizers',
    'Normalization pitfalls in many-objective benchmarking'
  ];
  var VE = ['IEEE Transactions on Evolutionary Computation', 'Evolutionary Computation', 'GECCO', 'Swarm and Evolutionary Computation', 'IEEE Congress on Evolutionary Computation', 'ACM Computing Surveys', 'Applied Soft Computing', 'Information Sciences', 'Neurocomputing', 'arXiv'];
  var SUR = ['Zhang', 'Li', 'Deb', 'Ishibuchi', 'Coello', 'Jin', 'Tan', 'Wang', 'Sato', 'Trivedi', 'Chugh', 'Miettinen', 'Osaba', 'Bader', 'Zitzler', 'Cheng', 'He', 'Yuan', 'Gao', 'Nojima'];
  var EXT_N = { bib: [24, 12], ris: [16, 10], csv: [10, 6], txt: [5, 4], zip: [5, 4], pdf: [1, 0] };
  function extOf(name) { var m = String(name).toLowerCase().match(/\.([a-z0-9]+)$/); return m ? m[1] : ''; }
  function mkEntry(seedName, i, ext) {
    var r = rng(hash(seedName + ':' + i));
    var ti = TI[Math.floor(r() * TI.length)];
    var a1 = SUR[Math.floor(r() * SUR.length)], a2 = SUR[Math.floor(r() * SUR.length)];
    var au = String.fromCharCode(65 + Math.floor(r() * 24)) + '. ' + a1 + (r() < .6 ? '\u00a0\u00a0' + String.fromCharCode(65 + Math.floor(r() * 24)) + '. ' + a2 : '');
    var yr = 2014 + Math.floor(r() * 13);
    var ve = VE[Math.floor(r() * VE.length)];
    var ci = Math.floor(Math.pow(r(), 2.2) * 880) + 3;
    var isPdf = ext === 'pdf';
    var st = 'ok', reason = '';
    var d = r();
    if (seedName === 'refs.bib') { if (i === 3 || i === 14 || i === 25) st = 'none'; else if (i === 5 || i === 22) st = 'multi'; else if (i === 9) st = 'dupe'; }
    else if (d < .07) st = 'none'; else if (d < .13) st = 'multi'; else if (d < .17) st = 'dupe';
    if (st === 'none') reason = isPdf ? 'No DOI in the PDF: title match below threshold' : 'Title not found on Semantic Scholar';
    return {
      ti: ti, au: au, yr: yr, ve: ve, ci: ci,
      src: isPdf ? seedName : seedName + '\u00a0\u00a0line ' + (6 + i * 3),
      state: st, reason: reason,
      cand: st === 'multi' ? [
        { ve: ve, yr: yr, ci: ci, note: 'Journal version' },
        { ve: 'arXiv', yr: yr - 1, ci: Math.max(2, Math.floor(ci * .4)), note: 'Preprint' }
      ] : null,
      checked: st === 'ok',
      pdf: isPdf ? true : undefined,
      pdfSrc: isPdf ? 'user' : undefined
    };
  }
  function parse(files) {
    var out = { files: [], entries: [] };
    (files || []).forEach(function (f) {
      var name = typeof f === 'string' ? f : f.name;
      var ext = extOf(name);
      if (ext === 'zip') {
        var h0 = hash(name), nz = EXT_N.zip[0] + (h0 % (EXT_N.zip[1] + 1));
        out.files.push({ name: name, ext: 'zip', n: nz });
        for (var z = 0; z < nz; z++) out.entries.push(mkEntry(name.replace(/\.zip$/i, '') + '-' + (z + 1) + '.pdf', z, 'pdf'));
        return;
      }
      if (!EXT_N[ext]) { out.files.push({ name: name, ext: ext || '?', n: 0, err: 'Unsupported format: use .bib, .ris, .csv, a DOI list, or PDFs' }); return; }
      var h = hash(name), n = EXT_N[ext][0] + (EXT_N[ext][1] ? h % (EXT_N[ext][1] + 1) : 0);
      if (name === 'refs.bib') n = 34;
      out.files.push({ name: name, ext: ext, n: n });
      for (var i = 0; i < n; i++) out.entries.push(mkEntry(name, i, ext));
    });
    return out;
  }
  function sample() { return parse(['refs.bib']); }
  var ORDER = [
    { key: 'multi', label: 'Needs a decision' },
    { key: 'none', label: "Couldn't match" },
    { key: 'ok', label: 'Matched' },
    { key: 'dupe', label: null }
  ];
  function groups(entries, dupeLabel) {
    return ORDER.map(function (g) {
      var rows = [];
      entries.forEach(function (e, i) { if (e.state === g.key) rows.push(i); });
      return { key: g.key, label: g.key === 'dupe' ? (dupeLabel || 'Already in the corpus') : g.label, rows: rows };
    }).filter(function (g) { return g.rows.length; });
  }
  var CK_ON = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="var(--primary-fg)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"></path></svg>';
  // Fallback glyphs only for the (never-shipped) case where paper-row.js hasn't
  // loaded; at runtime every consumer loads both, so the shared PR_ICONS win.
  var KI_ICONS = {
    user: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><circle cx="12" cy="8" r="4"></circle><path d="M4 21c1.4-3.6 4.4-5.5 8-5.5s6.6 1.9 8 5.5"></path></svg>',
    cal: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><rect x="3" y="5" width="18" height="16" rx="2"></rect><path d="M8 3v4M16 3v4M3 10h18"></path></svg>',
    clip: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18.4 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>',
    alert: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><circle cx="12" cy="12" r="9"></circle><path d="M12 8v4"></path><path d="M12 16h.01"></path></svg>'
  };
  function icons() { var P = typeof window !== 'undefined' && window.KLPaperRow; return (P && P.PR_ICONS) || KI_ICONS; }
  function kiPair(ic, tx, grow) {
    return '<span style="display:inline-flex; align-items:center; gap:4px; min-width:0;' + (grow ? ' flex:0 1 auto; overflow:hidden;' : ' flex:none;') + '">' +
      '<span style="color:var(--muted2); display:inline-flex; flex:none;">' + ic + '</span>' +
      '<span style="white-space:nowrap;' + (grow ? ' overflow:hidden; text-overflow:ellipsis;' : '') + ' font-variant-numeric:tabular-nums;">' + tx + '</span></span>';
  }
  // Compact paper-card contract (design-system.md § Compact paper cards): serif
  // title + icon-led meta. State lives on the GROUP block; a per-row control only
  // where it acts (checkbox = include, matches pill = pick a record). Rows carry
  // no dividers — groupHtml owns them.
  function rowHtml(e, i, opts) {
    opts = opts || {};
    var I = icons();
    var pad = opts.pad || '11px 14px';
    var right = '';
    if (e.state === 'ok') right = '<span class="ki-ck" role="checkbox" aria-checked="' + (e.checked ? 'true' : 'false') + '" data-i="' + i + '" style="width:16px; height:16px; border-radius:5px; flex:none; margin-top:2px; display:flex; align-items:center; justify-content:center; cursor:pointer; ' + (e.checked ? 'background:var(--primary); border:1.5px solid var(--primary);' : 'background:var(--card); border:1.5px solid var(--muted2);') + ' transition:background .12s ease, border-color .12s ease;">' + (e.checked ? CK_ON : '') + '</span>';
    else if (e.state === 'multi') right = '<button class="ki-multi" type="button" data-i="' + i + '" style="flex:none; margin-top:1px; border:1px solid var(--border); background:var(--card); border-radius:999px; padding:3px 9px; font-family:inherit; font-size:10.5px; font-weight:600; color:var(--fg2); cursor:pointer; white-space:nowrap; transition:background .14s ease, color .14s ease;">' + e.cand.length + ' matches</button>';
    return '<div class="ki-row" data-i="' + i + '" data-st="' + e.state + '" style="display:flex; align-items:flex-start; gap:10px; padding:' + pad + ';' + (e.state === 'dupe' ? ' opacity:.55;' : '') + '">' +
      '<div style="flex:1; min-width:0; display:flex; flex-direction:column; gap:5px;">' +
        '<div style="font-family:\'Newsreader\',serif; font-size:14px; line-height:1.32; font-weight:500; color:var(--fg); display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;">' + esc(e.ti) + '</div>' +
        '<div style="display:flex; align-items:center; gap:10px; min-width:0; font-size:11px; color:var(--muted);">' +
          kiPair(I.user, esc(e.au), 1) + kiPair(I.cal, e.yr) + kiPair(I.clip || I.cal, esc(e.src)) + '</div>' +
        (e.state === 'none' ? '<div style="display:flex; align-items:center; gap:6px; font-size:11px; line-height:1.4; color:color-mix(in oklab, var(--error) 62%, var(--fg));"><span style="display:inline-flex; flex:none; color:color-mix(in oklab, var(--error) 50%, var(--muted2));">' + (I.alert || '') + '</span><span>' + esc(e.reason) + '</span></div>' : '') +
      '</div>' + right + '</div>';
  }
  // Group block: ONE sectioning skin for every match-review surface — an inset card
  // (r12, --card) with a sentence-case header + count pill (error tint on "Couldn't
  // match") and a Show/Hide collapse. flat:true drops the card frame for hosts that
  // already draw one (Home's wizard box); pass first:true on the first flat group.
  function groupHtml(g, rowsArr, opts) {
    opts = opts || {};
    var open = opts.open !== false;
    var pillErr = g.key === 'none';
    var line = opts.flat ? 'var(--divider)' : 'color-mix(in oklab, var(--fg) 7%, transparent)';
    var pill = '<span style="font-size:10.5px; font-weight:700; padding:1px 7px; border-radius:999px; font-variant-numeric:tabular-nums;' +
      (pillErr ? ' background:color-mix(in oklab, var(--error) 9%, var(--card)); border:1px solid color-mix(in oklab, var(--error) 24%, var(--border)); color:color-mix(in oklab, var(--error) 62%, var(--fg));' : ' background:var(--card); color:var(--muted);') + '">' + rowsArr.length + '</span>';
    var head = '<button type="button" class="ki-sec" data-k="' + g.key + '" aria-expanded="' + (open ? 'true' : 'false') + '" style="width:100%; display:flex; align-items:center; gap:8px; padding:11px 14px; border:none; background:none; cursor:pointer; font-family:inherit; text-align:left;">' +
      '<span style="display:inline-flex; flex:none; color:var(--muted2); transition:transform .18s ease;' + (open ? ' transform:rotate(90deg);' : '') + '"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"></path></svg></span>' +
      '<span style="font-size:13px; font-weight:600; color:var(--fg);">' + g.label + '</span>' + pill +
      '<span style="margin-left:auto; font-size:12px; color:var(--fg2);">' + (open ? 'Hide' : 'Show') + '</span></button>';
    var sep = '<div style="height:1px; background:' + line + '; margin:0 14px;"></div>';
    var body = '<div class="ki-body" data-k="' + g.key + '" style="border-top:1px solid ' + line + ';' + (open ? '' : ' display:none;') + '">' + rowsArr.join(sep) + (opts.extra || '') + '</div>';
    if (opts.flat) return '<div class="ki-grp" data-k="' + g.key + '" style="' + (opts.first ? '' : 'border-top:1px solid var(--divider);') + '">' + head + body + '</div>';
    return '<div class="ki-grp" data-k="' + g.key + '" style="background:var(--group-bg); border-radius:12px; margin:0 0 10px; overflow:hidden;">' + head + body + '</div>';
  }
  function candPopHtml(e) {
    return '<div style="font-size:10.5px; font-weight:600; letter-spacing:.07em; text-transform:uppercase; color:var(--muted2); padding:2px 8px 6px;">Which record?</div>' +
      e.cand.map(function (c, j) {
        return '<button class="ki-cand" type="button" data-c="' + j + '" style="width:100%; text-align:left; display:flex; flex-direction:column; gap:2px; padding:7px 8px; border:none; background:none; border-radius:8px; font-family:inherit; cursor:pointer;">' +
          '<span style="font-size:10.5px; letter-spacing:.04em; text-transform:uppercase; color:var(--fg2); font-weight:500;">' + esc(c.ve) + '\u00a0\u00a0' + c.yr + '</span>' +
          '<span style="font-size:11px; color:var(--muted);">' + c.note + '\u00a0\u00a0' + c.ci + ' citations</span></button>';
      }).join('');
  }
  var EXTC = { bib: 'BIB', ris: 'RIS', csv: 'CSV', txt: 'TXT', pdf: 'PDF', zip: 'ZIP' };
  function fileRowHtml(f, i, opts) {
    var pad = (opts && opts.pad) || '9px var(--pin)';
    return '<div class="ki-file" data-f="' + i + '" style="display:flex; align-items:center; gap:10px; padding:' + pad + ';">' +
      '<span style="flex:none; width:34px; font-size:9.5px; font-weight:700; letter-spacing:.04em; color:' + (f.err ? 'color-mix(in oklab, var(--error) 62%, var(--fg))' : 'var(--fg2)') + '; background:var(--icon-bg); border:1px solid var(--border); border-radius:6px; padding:3px 0; text-align:center;">' + esc(EXTC[f.ext] || f.ext.toUpperCase() || '?') + '</span>' +
      '<span style="flex:1; min-width:0; font-size:12.5px; color:var(--fg); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">' + esc(f.name) + '</span>' +
      '<span class="ki-fst" style="flex:none; display:inline-flex; align-items:center; gap:6px; font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums;"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" style="animation:rnSpin .9s linear infinite;"><path d="M21 12a9 9 0 1 1-6.2-8.56"></path></svg>Parsing\u2026</span>' +
      '</div>';
  }
  window.KLImport = { parse: parse, sample: sample, groups: groups, rowHtml: rowHtml, groupHtml: groupHtml, candPopHtml: candPopHtml, fileRowHtml: fileRowHtml, extOf: extOf };
})();
