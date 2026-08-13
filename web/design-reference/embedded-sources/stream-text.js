// Shared token-by-token reveal for agent-written text.
// Used by the agent thread and the "In this corpus" summary so both stream the
// same way instead of snapping in. Works on any HTML fragment: the markup is
// parsed once, its text nodes are emptied, then refilled word by word.
export function streamHtml(host, html, opts) {
  opts = opts || {};
  const cps = opts.cps || 220;            // characters per second
  const caretOn = opts.caret !== false;
  host.innerHTML = html;
  const blocks = [...host.children];
  const texts = [];
  const walk = n => {
    for (const c of n.childNodes) {
      if (c.nodeType === 3) { const v = c.nodeValue; if (v.trim()) { texts.push({ n: c, full: v }); c.nodeValue = ''; } }
      else if (c.nodeType === 1) walk(c);
    }
  };
  walk(host);
  blocks.forEach(b => { b.style.visibility = 'hidden'; });
  let caret = null;
  if (caretOn) {
    caret = document.createElement('span');
    caret.className = 'rv-caret';
  }
  let i = 0, pos = 0, raf = 0, last = 0, done = false;
  const showBlockOf = node => {
    let p = node;
    while (p && p.parentNode !== host) p = p.parentNode;
    if (p && p.style.visibility === 'hidden') {
      p.style.visibility = '';
      if (p.animate) p.animate([{ opacity: 0, transform: 'translateY(4px)' }, { opacity: 1, transform: 'none' }], { duration: 220, easing: 'ease-out' });
    }
  };
  const finish = () => {
    if (done) return;
    done = true;
    cancelAnimationFrame(raf);
    texts.forEach(t => { t.n.nodeValue = t.full; });
    blocks.forEach(b => { b.style.visibility = ''; });
    if (caret && caret.parentNode) caret.remove();
    if (opts.onDone) opts.onDone();
  };
  const tick = ts => {
    if (!host.isConnected) { done = true; return; }
    const dt = last ? Math.min(120, ts - last) : 16;
    last = ts;
    let budget = Math.max(1, Math.round(cps * dt / 1000));
    while (budget > 0 && i < texts.length) {
      const t = texts[i];
      if (pos === 0) showBlockOf(t.n);
      // advance to the end of the next word
      let next = pos + budget;
      const sp = t.full.indexOf(' ', next);
      next = sp === -1 || sp - next > 12 ? Math.min(t.full.length, next) : sp + 1;
      budget -= next - pos;
      pos = next;
      t.n.nodeValue = t.full.slice(0, pos);
      if (caret) { const p = t.n.parentNode; if (p) p.appendChild(caret); }
      if (pos >= t.full.length) { i++; pos = 0; }
    }
    if (opts.onTick) opts.onTick();
    if (i >= texts.length) return finish();
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
  return { cancel() { done = true; cancelAnimationFrame(raf); if (caret && caret.parentNode) caret.remove(); }, finish: finish };
}
