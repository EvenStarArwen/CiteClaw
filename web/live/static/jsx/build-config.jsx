/* eslint-disable */
// Section D (Build mode) — Block + filter-tree configuration.
// LEFT: the selected node's filter TREE (Sequential/Parallel/Any/Not/Route + leaves),
//       rendered vertically, with an inline "Add" popover grouped by Leaf / Composite.
// RIGHT: parameters for whichever filter (or the block itself) is currently selected.

const COMPOSITE_KINDS = ["Sequential", "Parallel", "Any", "Not", "Route"];

const LEAF_KINDS = [
  { kind: "YearFilter",            label: "Year",              desc: "[min, max] range" },
  { kind: "CitationFilter",        label: "Citation",          desc: "min cites scaled by age" },
  { kind: "SimilarityFilter",      label: "Similarity",        desc: "Ref / Cit / Semantic" },
  { kind: "LLMFilter",             label: "LLM",               desc: "batched screening" },
  { kind: "TitleKeywordFilter",    label: "Title keyword",     desc: "string / boolean over title" },
  { kind: "AbstractKeywordFilter", label: "Abstract keyword",  desc: "same DSL, abstract" },
  { kind: "VenueKeywordFilter",    label: "Venue keyword",     desc: "same DSL, venue" },
];
const COMPOSITE_OPTS = [
  { kind: "Sequential", label: "Match ALL", desc: "AND · every child must pass" },
  { kind: "Any",        label: "Match ANY", desc: "OR · at least one passes" },
  { kind: "Not",        label: "Exclude",   desc: "NOT · invert one child" },
  { kind: "Route",      label: "Route",     desc: "if / elif / else dispatch" },
  { kind: "Parallel",   label: "Parallel",  desc: "broadcast · union outputs" },
];
// Plain-English labels for composite filter groups (the raw CLI names
// Sequential/Any confuse users). Pill text + the muted meta line beside it.
const COMPOSITE_LABEL = { Sequential: "Match ALL", Any: "Match ANY", Not: "Exclude", Route: "Route", Parallel: "Parallel" };
const COMPOSITE_META  = { Sequential: "AND · all must pass", Any: "OR · one is enough", Not: "invert the child", Route: "if / elif / else", Parallel: "union of branches" };
// Compact pill labels for the long keyword-filter kinds — full names blew up
// the row and squeezed the summary down to useless fragments.
const LEAF_PILL = { TitleKeywordFilter: "Title kw", AbstractKeywordFilter: "Abs kw", VenueKeywordFilter: "Venue kw" };

// --- tree helpers ---------------------------------------------------------
let __fid = 1000;
const nextFid = () => "f" + (++__fid);
function defaultParams(kind) {
  switch (kind) {
    case "YearFilter":            return { min: 2019, max: 2025 };
    case "CitationFilter":        return { beta: 30 };
    case "SimilarityFilter":      return { threshold: 0.025, measures: [{ kind: "RefSim" }] };
    case "LLMFilter":             return { scope: "title_abstract", formula: "q1", queries: { q1: "" }, model: "", effort: "" };
    case "TitleKeywordFilter":    return { match: "substring", expression: "" };
    case "AbstractKeywordFilter": return { match: "substring", expression: "" };
    case "VenueKeywordFilter":    return { match: "starts_with", expression: '"Nature"' };
    default: return {};
  }
}
function newNode(kind) {
  if (COMPOSITE_KINDS.includes(kind)) {
    if (kind === "Not") return { id: nextFid(), kind, layer: { id: nextFid(), kind: "YearFilter", params: defaultParams("YearFilter") } };
    if (kind === "Route") {
      // Start with one venue condition + an empty ELSE, so the shape of
      // if / else routing is visible immediately.
      return { id: nextFid(), kind,
        routes: [{ id: nextFid(), if: { kind: "VenueIn", values: ["Nature"] }, pass_to: null }],
        else: null };
    }
    return { id: nextFid(), kind, children: [] };
  }
  return { id: nextFid(), kind, params: defaultParams(kind) };
}
// Recursively find + update + remove nodes by id. Route nodes hold their
// children in routes[].pass_to and .else, so every walker visits those too.
function findNode(t, id) {
  if (!t) return null;
  if (t.id === id) return t;
  for (const c of (t.children || [])) { const r = findNode(c, id); if (r) return r; }
  if (t.layer) { const r = findNode(t.layer, id); if (r) return r; }
  for (const rt of (t.routes || [])) {
    if (rt && rt.pass_to) { const r = findNode(rt.pass_to, id); if (r) return r; }
  }
  if (t.else) { const r = findNode(t.else, id); if (r) return r; }
  return null;
}
function mapTree(t, id, fn) {
  if (!t) return t;
  if (t.id === id) return fn(t);
  if (t.children) return { ...t, children: t.children.map(c => mapTree(c, id, fn)) };
  if (t.layer)    return { ...t, layer: mapTree(t.layer, id, fn) };
  if (t.routes) {
    return { ...t,
      routes: t.routes.map(r => (r && r.pass_to ? { ...r, pass_to: mapTree(r.pass_to, id, fn) } : r)),
      else: t.else ? mapTree(t.else, id, fn) : t.else };
  }
  return t;
}
function removeFromTree(t, id) {
  if (!t) return t;
  if (t.children) {
    const filtered = t.children.filter(c => c.id !== id);
    const mapped = filtered.map(c => removeFromTree(c, id));
    return { ...t, children: mapped };
  }
  if (t.layer && t.layer.id === id) return { ...t, layer: null };
  if (t.layer) return { ...t, layer: removeFromTree(t.layer, id) };
  if (t.routes) {
    return { ...t,
      routes: t.routes.map(r => {
        if (!r || !r.pass_to) return r;
        if (r.pass_to.id === id) return { ...r, pass_to: null };
        return { ...r, pass_to: removeFromTree(r.pass_to, id) };
      }),
      else: t.else ? (t.else.id === id ? null : removeFromTree(t.else, id)) : t.else };
  }
  return t;
}
// Swap the node with `id` with its previous/next SIBLING (dir −1 / +1) so
// filters can be inserted anywhere: add at the end, then move it up.
function moveInTree(t, id, dir) {
  if (!t) return t;
  if (t.children) {
    const i = t.children.findIndex(c => c.id === id);
    if (i >= 0) {
      const j = i + dir;
      if (j < 0 || j >= t.children.length) return t;
      const kids = t.children.slice();
      [kids[i], kids[j]] = [kids[j], kids[i]];
      return { ...t, children: kids };
    }
    return { ...t, children: t.children.map(c => moveInTree(c, id, dir)) };
  }
  if (t.layer) return { ...t, layer: moveInTree(t.layer, id, dir) };
  if (t.routes) {
    return { ...t,
      routes: t.routes.map(r => (r && r.pass_to ? { ...r, pass_to: moveInTree(r.pass_to, id, dir) } : r)),
      else: t.else ? moveInTree(t.else, id, dir) : t.else };
  }
  return t;
}
// Deep-copy a filter subtree with FRESH ids everywhere, so a pasted filter is
// fully independent of its original (edits never leak between the two).
function cloneFilterNode(node) {
  const c = JSON.parse(JSON.stringify(node));
  const walk = (n) => {
    if (!n || typeof n !== "object") return;
    if (n.id) n.id = nextFid();
    (n.children || []).forEach(walk);
    if (n.layer) walk(n.layer);
    (n.routes || []).forEach(r => {
      if (!r) return;
      if (r.id) r.id = nextFid();          // route ROWS carry ids too
      if (r.pass_to) walk(r.pass_to);
    });
    if (n.else) walk(n.else);
  };
  walk(c);
  return c;
}
// Human-readable label for one Route condition.
function predicateSummary(p) {
  if (!p) return "always";
  if (p.kind === "VenueIn") {
    const vs = (p.values || []).filter(v => String(v).trim());
    return vs.length ? "venue contains " + vs.join(" / ") : "venue contains … (empty)";
  }
  if (p.kind === "CitAtLeast") return `citations ≥ ${p.n ?? "…"}`;
  if (p.kind === "YearAtLeast") return `year ≥ ${p.n ?? "…"}`;
  return p.kind || "?";
}
// One shared filter clipboard for the whole builder — copy in one step's
// screener, paste in any other (the "copy mask / paste mask" pattern). The
// clip is snapshotted at copy time, so later edits to the source don't follow.
let FILTER_CLIP = null;
function copyFilterToClip(node) { FILTER_CLIP = JSON.parse(JSON.stringify(node)); }
function getFilterClip() { return FILTER_CLIP; }
// Short label + summary for the paste entry (leaf → its catalog label,
// composite → its plain-English label + leaf count).
function clipLabel(n) {
  if (!n) return "";
  if (COMPOSITE_KINDS.includes(n.kind)) return COMPOSITE_LABEL[n.kind] || n.kind;
  const leaf = LEAF_KINDS.find(l => l.kind === n.kind);
  return (leaf ? leaf.label : n.kind.replace("Filter", "")) + " filter";
}
function clipDesc(n) {
  if (!n) return "";
  if (COMPOSITE_KINDS.includes(n.kind)) {
    const leaves = (t) => {
      if (!COMPOSITE_KINDS.includes(t.kind)) return 1;
      let kids = t.children || (t.layer ? [t.layer] : []);
      if (t.kind === "Route")
        kids = [...(t.routes || []).map(r => r && r.pass_to), t.else].filter(Boolean);
      return kids.reduce((s, c) => s + leaves(c), 0);
    };
    const k = leaves(n);
    return `group with ${k} filter${k === 1 ? "" : "s"} inside`;
  }
  return filterSummary(n);
}

// --- components -----------------------------------------------------------

function ConfigField({ label, children, wide, full, hint, info }) {
  return (
    <div className={"field" + (wide ? " field-wide" : "") + (full ? " field-full" : "")}>
      <span className="field-label">{label}{info && <InfoDot text={info} />}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </div>
  );
}

// The current year — the hard ceiling for any "max year" control (a paper
// can't be published in the future). Read at render time in the browser.
const NOW_YEAR = new Date().getFullYear();

function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

// A small info icon that toggles a description popover. The popover is
// PORTALED to <body> and fixed-positioned to the LEFT of the icon, so it
// spills out of the (right) config sidebar into the central panel instead of
// being clipped by the panel's overflow.
function InfoDot({ text }) {
  const [open, setOpen] = React.useState(false);
  const [pos, setPos] = React.useState(null);
  const ref = React.useRef(null);
  const place = () => {
    const el = ref.current; if (!el) return;
    const r = el.getBoundingClientRect();
    const W = 240, gap = 10, margin = 8;
    let left = r.left - gap - W, arrow = "right";          // prefer opening left
    if (left < margin) { left = Math.min(r.right + gap, window.innerWidth - W - margin); arrow = "left"; }
    const top = Math.max(margin + 48, Math.min(r.top + r.height / 2, window.innerHeight - margin - 48));
    setPos({ top, left, arrow });
  };
  React.useEffect(() => {
    if (!open) return;
    place();
    const onDoc = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === "Escape") setOpen(false); };
    const reflow = () => place();
    document.addEventListener("mousedown", onDoc);
    window.addEventListener("keydown", onKey);
    window.addEventListener("scroll", reflow, true);
    window.addEventListener("resize", reflow);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", reflow, true);
      window.removeEventListener("resize", reflow);
    };
  }, [open]);
  return (
    <span className="info-dot-wrap" ref={ref}>
      <button type="button" className={"info-dot" + (open ? " is-open" : "")}
        onClick={(e) => { e.stopPropagation(); setOpen(v => !v); }}
        title="What is this?" aria-label="What is this?">
        <Icon name="info" size={13} />
      </button>
      {open && pos && ReactDOM.createPortal(
        <span className={"info-pop info-pop-" + pos.arrow} role="tooltip"
          style={{ top: pos.top, left: pos.left }}>{text}</span>,
        document.body
      )}
    </span>
  );
}

// A −/value/+ integer control whose buttons GRAY OUT (and no-op) at the bounds,
// so the user simply cannot pick an illegal value. The ceiling is enforced live
// even while typing; the floor is applied on blur so 4-digit years type freely.
function NumStepper({ value, onChange, lo, hi, step }) {
  const s = step || 1;
  const v = Number.isFinite(value) ? value : (lo != null ? lo : 0);
  const clampHi = (x) => (Number.isFinite(x) ? (hi != null ? Math.min(hi, x) : x) : v);
  const clamp = (x) => {
    if (!Number.isFinite(x)) return v;
    if (lo != null) x = Math.max(lo, x);
    if (hi != null) x = Math.min(hi, x);
    return x;
  };
  const atLo = lo != null && v <= lo;
  const atHi = hi != null && v >= hi;
  return (
    <div className="numstep">
      <button type="button" className="numstep-btn" disabled={atLo}
        onClick={() => onChange(clamp(v - s))} aria-label="Decrease">−</button>
      <input className="numstep-val" type="number" value={v}
        onChange={e => onChange(clampHi(+e.target.value))}
        onBlur={e => onChange(clamp(+e.target.value))} />
      <button type="button" className="numstep-btn" disabled={atHi}
        onClick={() => onChange(clamp(v + s))} aria-label="Increase">+</button>
    </div>
  );
}

// --- named-formula consistency (keyword / query editors) -----------------
// Identifiers referenced by a boolean formula (everything that isn't & | ! ( )).
function _formulaTokens(formula) {
  const m = String(formula || "").match(/[A-Za-z_][A-Za-z0-9_]*/g) || [];
  return Array.from(new Set(m));
}
// Cross-check a name→value dict against the formula that references the names.
// errors block a legal config; warnings just advise. Phrasing is user-facing.
function checkNamedFormula(dict, formula, noun) {
  noun = noun || "keyword";
  const names = Object.keys(dict || {});
  const used = _formulaTokens(formula);
  const errors = [], warnings = [];
  if (!String(formula || "").trim()) errors.push("The formula is empty — write an expression such as " + (names[0] || (noun === "query" ? "q1" : "k1")) + ".");
  names.forEach(n => {
    if (!n.trim()) errors.push("A " + noun + " has no name.");
    if ((dict[n] == null || dict[n] === "")) warnings.push(cap(noun) + " “" + n + "” has no " + (noun === "query" ? "prompt" : "text") + " yet.");
  });
  used.forEach(t => { if (!names.includes(t)) errors.push("The formula uses “" + t + "”, but no " + noun + " named “" + t + "” is defined."); });
  names.forEach(n => { if (n.trim() && !used.includes(n)) warnings.push(cap(noun) + " “" + n + "” is never used in the formula."); });
  return { errors, warnings };
}

// Red errors + amber warnings under a formula editor.
function ValidationNotes({ errors, warnings }) {
  if (!(errors && errors.length) && !(warnings && warnings.length)) return null;
  return (
    <div className="val-notes field-full">
      {(errors || []).map((m, i) => (
        <div key={"e" + i} className="val-note val-note-err"><Icon name="alert-circle" size={12} /><span>{m}</span></div>
      ))}
      {(warnings || []).map((m, i) => (
        <div key={"w" + i} className="val-note val-note-warn"><Icon name="alert-triangle" size={12} /><span>{m}</span></div>
      ))}
    </div>
  );
}

// One editable name→value row. The NAME is editable (identifier-safe chars only)
// and commits on blur/Enter; a rename to an empty or duplicate name reverts.
// Query prompts get a TEXTAREA (drag the lower-right corner to grow it) —
// screening prompts run long, and a one-line input scrolls like an address bar.
function KVRow({ name, value, siblings, noun, onRename, onValue, onRemove }) {
  const [draft, setDraft] = React.useState(name);
  React.useEffect(() => { setDraft(name); }, [name]);
  const nk = draft.trim();
  const bad = !nk || (nk !== name && siblings.includes(nk));
  const commit = () => {
    if (bad) { setDraft(name); return; }
    if (nk !== name) onRename(name, nk);
  };
  const multiline = noun === "query";
  return (
    <div className={"cfg-kv" + (multiline ? " cfg-kv-multiline" : "")}>
      <input className={"cfg-kv-k" + (bad ? " is-bad" : "")} value={draft}
        onChange={e => setDraft(e.target.value.replace(/[^A-Za-z0-9_]/g, ""))}
        onBlur={commit}
        onKeyDown={e => { if (e.key === "Enter") e.target.blur(); }}
        title="Rename this identifier — reference it in the formula" />
      {multiline ? (
        <textarea className="cfg-kv-v cfg-kv-ta" rows={2} value={value}
          placeholder="yes/no question about the paper… (drag the corner to enlarge)"
          onChange={e => onValue(name, e.target.value)} />
      ) : (
        <input className="cfg-kv-v" value={value}
          placeholder="literal string to match…"
          onChange={e => onValue(name, e.target.value)} />
      )}
      <button className="cfg-measure-del" onClick={() => onRemove(name)} title={"Remove " + noun}>×</button>
    </div>
  );
}

// The shared Formula + named-dict editor used by keyword filters and the LLM
// filter (queries). Renaming a key also rewrites the matching token in the
// formula so the two stay consistent; the validation still flags real typos.
function NamedDictEditor({ p, patch, field, noun, formulaInfo }) {
  const dict = p[field] || {};
  const names = Object.keys(dict);
  const { errors, warnings } = checkNamedFormula(dict, p.formula, noun);

  const setValue = (nm, v) => patch({ [field]: { ...dict, [nm]: v } });
  const rename = (oldK, newK) => {
    const renamed = {};
    names.forEach(k => { renamed[k === oldK ? newK : k] = dict[k]; });
    const nf = String(p.formula || "").replace(new RegExp("\\b" + oldK + "\\b", "g"), newK);
    patch({ [field]: renamed, formula: nf });
  };
  const remove = (nm) => { const next = { ...dict }; delete next[nm]; patch({ [field]: next }); };
  const add = () => {
    const base = noun === "query" ? "q" : "k";
    let i = names.length + 1, nm = base + i;
    while (dict[nm] !== undefined) nm = base + (++i);
    patch({ [field]: { ...dict, [nm]: "" } });
  };

  return (
    <>
      <ConfigField label="Formula" wide info={formulaInfo}
        hint="Operators: & (and) · | (or) · ! (not) · ( )">
        <input value={p.formula || ""} onChange={e => patch({ formula: e.target.value })}
          placeholder={names[0] || (noun === "query" ? "q1" : "k1")} />
      </ConfigField>
      <ConfigField label={noun === "query" ? "Queries" : "Keywords"} full
        hint={noun === "query" ? "name → prompt" : "name → literal string"}
        info={noun === "query"
          ? "Each row is a named yes/no question the model answers about the paper; reference the names in the formula above."
          : "Each row maps a short name to a literal string to look for; reference the names in the formula above."}>
        <div className="cfg-kv-list">
          {names.map(nm => (
            <KVRow key={nm} name={nm} value={dict[nm]} noun={noun}
              siblings={names.filter(n => n !== nm)}
              onRename={rename} onValue={setValue} onRemove={remove} />
          ))}
          <button className="cfg-measure-add" onClick={add}>
            <Icon name="plus" size={11} /> Add {noun}
          </button>
        </div>
      </ConfigField>
      <ValidationNotes errors={errors} warnings={warnings} />
    </>
  );
}

// --- direct keyword expressions -------------------------------------------
// Keyword filters take ONE boolean expression over literal terms — e.g.
//   ("large language model" | LLM) & "scientific discovery"
// The backend compiles it to the CLI's named formula+keywords shape.

// Older nodes (and the shipped default pipeline) carry {formula, keywords};
// render them as an equivalent expression so nothing breaks on upgrade.
function exprFromLegacy(p) {
  if (p.expression != null) return p.expression;
  if (p.keyword) return '"' + p.keyword + '"';
  if (!p.formula) return "";
  const dict = p.keywords || {};
  return String(p.formula).replace(/[A-Za-z_][A-Za-z0-9_]*/g, (nm) =>
    dict[nm] != null ? '"' + dict[nm] + '"' : nm);
}

function checkKeywordExpression(expr) {
  const errors = [], warnings = [];
  const s = String(expr || "").trim();
  if (!s) { errors.push("The expression is empty — enter at least one keyword."); return { errors, warnings }; }
  if (s.includes("*"))
    errors.push("Wildcards (*) aren’t supported — terms already match as substrings (or whole words / prefixes via the match mode).");
  if ((s.match(/"/g) || []).length % 2)
    errors.push("Unbalanced quotes — every opening \" needs a closing \".");
  const blanked = s.replace(/"[^"]*"/g, "x");
  let depth = 0, under = false;
  for (const ch of blanked) {
    if (ch === "(") depth++;
    else if (ch === ")") { depth--; if (depth < 0) under = true; }
  }
  if (depth !== 0 || under) errors.push("Unbalanced parentheses.");
  const leftover = blanked.replace(/[A-Za-z0-9_À-￿][A-Za-z0-9_À-￿-]*/g, " ").replace(/[&|!()\s]/g, "");
  if (leftover)
    errors.push("Unexpected character" + (leftover.length > 1 ? "s" : "") + ": " +
      [...new Set(leftover)].join(" ") + " — allowed: words, \"quoted phrases\", & | ! and parentheses.");
  if (/[&|]\s*$/.test(blanked)) errors.push("The expression ends with a dangling operator.");
  if (/^\s*[&|]/.test(blanked)) errors.push("The expression starts with an operator.");
  return { errors, warnings };
}

function KeywordExpressionEditor({ p, patch }) {
  const expr = exprFromLegacy(p);
  const { errors, warnings } = checkKeywordExpression(expr);
  return (
    <>
      <ConfigField label="Keywords" wide full
        info={'One boolean expression over the literal terms to look for. Quote multi-word phrases; bare single words work as-is. Terms are case-insensitive and compared per the match mode. Example: ("large language model" | LLM) & "scientific discovery"'}
        hint={'& and · | or · ! not · ( ) group · "…" phrase — no * wildcards'}>
        <textarea className="cfg-kv-ta cfg-expr-ta" rows={2} value={expr}
          placeholder={'("large language model" | LLM) & "scientific discovery"'}
          onChange={e => patch({ expression: e.target.value, formula: undefined, keywords: undefined, keyword: undefined })} />
      </ConfigField>
      <ValidationNotes errors={errors} warnings={warnings} />
    </>
  );
}

const MEASURE_INFO = {
  RefSim: "Jaccard overlap between this paper's reference list and the anchor paper's. No parameters.",
  CitSim: "Jaccard overlap between the sets of papers that cite each one. “Cited at least” lets any paper with that many citations pass outright.",
  SemanticSim: "Cosine similarity of title+abstract embeddings (SPECTER2 by default).",
};
function measureDefaults(kind) {
  if (kind === "CitSim") return { kind: "CitSim", pass_if_cited_at_least: 200 };
  if (kind === "SemanticSim") return { kind: "SemanticSim", embedder: "s2" };
  return { kind: "RefSim" };
}

// Voyage API-key entry, shown when a SemanticSim measure picks the voyage
// embedder. The key is stored locally (.env.local via the settings store),
// NOT in the pipeline JSON. Voyage runs aren't enabled yet — this only makes
// the UI complete; translate coerces voyage → s2 at run time.
function VoyageKeyField() {
  const settings = useLive("settings");
  const present = !!(settings.keys && settings.keys.voyage_api_key);
  const [val, setVal] = React.useState("");
  const [saved, setSaved] = React.useState(false);
  const commit = () => {
    const v = val.trim();
    if (!v) return;
    saveSettings({ voyage_api_key: v }).then(() => { setVal(""); setSaved(true); }).catch(() => {});
  };
  return (
    <div className="voyage-key">
      <div className="voyage-key-row">
        <input type="password" className="voyage-key-input" value={val}
          placeholder={present ? "•••••••• saved (leave blank to keep)" : "Voyage API key"}
          onChange={e => { setVal(e.target.value); setSaved(false); }}
          onBlur={commit}
          onKeyDown={e => { if (e.key === "Enter") e.target.blur(); }} />
        {(present || saved) && <span className="voyage-key-ok">✓ set</span>}
      </div>
      <div className="voyage-key-note">
        Voyage embeddings aren’t enabled in this version — runs use s2 (SPECTER2).
        Your key is stored locally for when they are.
      </div>
    </div>
  );
}

// Per-filter LLM model override — "Default" follows the Settings pick; the
// list mirrors the Settings catalog (prices + which model this build actually
// runs). Different filters may use different models: the CLI's per-block
// `model:` override, surfaced per use-case.
function ModelOverrideSelect({ value, onChange }) {
  const settings = useLive("settings");
  const models = settings.models || [];
  const byProv = {};
  models.forEach(m => (byProv[m.provider] || (byProv[m.provider] = [])).push(m));
  const known = models.some(m => m.id === value);
  return (
    <select value={value} onChange={e => onChange(e.target.value)}>
      <option value="">
        Default (Settings{settings.model ? " · " + settings.model : ""})
      </option>
      {["gemini", "openai"].map(prov => (
        <optgroup key={prov} label={prov === "gemini" ? "Google Gemini" : "OpenAI"}>
          {(byProv[prov] || []).map(m => (
            <option key={m.id} value={m.id}>
              {m.label} — ${m.input}/${m.output}{m.supported ? " ✓" : " · not yet runnable"}
            </option>
          ))}
        </optgroup>
      ))}
      {value && !known && <option value={value}>{value}</option>}
    </select>
  );
}

function AddFilterPopover({ onPick, onClose, allowComposite = true, anchorRef }) {
  const [pos, setPos] = React.useState(null);
  const clip = getFilterClip();
  React.useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  React.useLayoutEffect(() => {
    const compute = () => {
      const el = anchorRef?.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const popW = 280, popH = clip ? 420 : 360;
      const margin = 8;
      // Prefer below; flip up if no room
      let top = r.bottom + 4;
      if (top + popH > window.innerHeight - margin) {
        top = Math.max(margin, r.top - 4 - popH);
      }
      let left = r.left;
      if (left + popW > window.innerWidth - margin) {
        left = Math.max(margin, window.innerWidth - popW - margin);
      }
      setPos({ top, left });
    };
    compute();
    window.addEventListener("scroll", compute, true);
    window.addEventListener("resize", compute);
    return () => {
      window.removeEventListener("scroll", compute, true);
      window.removeEventListener("resize", compute);
    };
  }, [anchorRef]);

  return (
    <>
      <div className="ft-pop-scrim" onClick={onClose} />
      <div
        className="ft-pop ft-pop-fixed"
        style={pos ? { top: pos.top, left: pos.left } : { visibility: "hidden" }}
      >
        {clip && (
          <div className="ft-pop-group">
            <div className="ft-pop-label">Clipboard</div>
            <div className="ft-pop-list">
              <button className="ft-pop-item ft-pop-item-paste"
                onClick={() => { onPick(cloneFilterNode(clip)); onClose(); }}>
                <span className="ft-pop-name"><Icon name="clipboard-paste" size={11} /> Paste {clipLabel(clip)}</span>
                <span className="ft-pop-desc">{clipDesc(clip)}</span>
              </button>
            </div>
          </div>
        )}
        <div className="ft-pop-group">
          <div className="ft-pop-label">Leaf filters</div>
          <div className="ft-pop-list">
            {LEAF_KINDS.map(l => (
              <button key={l.kind} className="ft-pop-item" onClick={() => { onPick(newNode(l.kind)); onClose(); }}>
                <span className="ft-pop-name">{l.label}</span>
                <span className="ft-pop-desc">{l.desc}</span>
              </button>
            ))}
          </div>
        </div>
        {allowComposite && (
          <div className="ft-pop-group">
            <div className="ft-pop-label">Composites</div>
            <div className="ft-pop-list">
              {COMPOSITE_OPTS.map(c => (
                <button key={c.kind} className="ft-pop-item ft-pop-item-comp" onClick={() => { onPick(newNode(c.kind)); onClose(); }}>
                  <span className="ft-pop-name">{c.label}</span>
                  <span className="ft-pop-desc">{c.desc}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
}

// Hover actions shared by leaf rows and composite headers: move among
// siblings (↑/↓ — this is how a filter gets INSERTED BEFORE an existing one:
// add it at the end, then walk it up), copy to the clipboard, remove.
function FilterNodeActs({ node, canUp, canDown, onMove, onRemove }) {
  const [copied, setCopied] = React.useState(false);
  React.useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 1200);
    return () => clearTimeout(t);
  }, [copied]);
  const movable = canUp || canDown;
  return (
    <span className="ft-node-acts">
      {movable && (
        <>
          <button className="ft-act" disabled={!canUp} title="Move up"
            onClick={(e) => { e.stopPropagation(); onMove(node.id, -1); }}>
            <Icon name="arrow-up" size={11} />
          </button>
          <button className="ft-act" disabled={!canDown} title="Move down"
            onClick={(e) => { e.stopPropagation(); onMove(node.id, +1); }}>
            <Icon name="arrow-down" size={11} />
          </button>
        </>
      )}
      <button className={"ft-act" + (copied ? " is-copied" : "")}
        title="Copy filter — paste it from any “Add” menu, in this step or another"
        onClick={(e) => { e.stopPropagation(); copyFilterToClip(node); setCopied(true); }}>
        <Icon name={copied ? "check" : "copy"} size={11} />
      </button>
      <button className="ft-leaf-x" onClick={(e) => { e.stopPropagation(); onRemove(node.id); }} title="Remove">×</button>
    </span>
  );
}

// An empty Route branch slot: an "Add filter" button opening the same
// popover as everywhere else; the pick lands in that route's pass_to / else.
function RouteSlot({ onPick }) {
  const [adding, setAdding] = React.useState(false);
  const btnRef = React.useRef(null);
  return (
    <div className="ft-add-wrap">
      <button ref={btnRef} className="ft-add" onClick={(e) => { e.stopPropagation(); setAdding(true); }}>
        <Icon name="plus" size={11} /> Add filter for this branch
      </button>
      {adding && (
        <AddFilterPopover allowComposite anchorRef={btnRef}
          onPick={onPick} onClose={() => setAdding(false)} />
      )}
    </div>
  );
}

function FilterTreeNode({ node, depth, selectedId, onSelect, onAddChild, onRemove, onMove, onAddToRoute, canUp, canDown, plain }) {
  const [adding, setAdding] = React.useState(false);
  const addBtnRef = React.useRef(null);
  const isComposite = COMPOSITE_KINDS.includes(node.kind);
  const selected = selectedId === node.id;

  if (!isComposite) {
    return (
      <div className={"ft-leaf" + (selected ? " is-selected" : "")} onClick={(e) => { e.stopPropagation(); onSelect(node.id); }}
        title={filterSummary(node)}>
        <span className="ft-leaf-kind">{LEAF_PILL[node.kind] || node.kind.replace("Filter", "")}</span>
        {/* inner span so a container query can hide the summary entirely when
            the row is too narrow to show anything readable */}
        <span className="ft-leaf-body"><span className="ft-leaf-body-txt">{filterSummary(node)}</span></span>
        <FilterNodeActs node={node} canUp={canUp} canDown={canDown} onMove={onMove} onRemove={onRemove} />
      </div>
    );
  }

  // ── Route: the IF / ELIF / ELSE chips ARE the structure — no extra
  // "Route" header level. Conditions render prominently next to the chips
  // (click one to edit); each branch's filters list directly below with
  // their AND implied. The route's own hover actions live on the IF row.
  if (node.kind === "Route") {
    const routes = node.routes || [];
    const passProps = { depth: depth + 1, selectedId, onSelect, onAddChild, onRemove, onMove, onAddToRoute };
    const slot = (label, sub, slotId, acts) => (
      <div className="ft-route" key={slotId}>
        <div className="ft-route-head" onClick={(e) => { e.stopPropagation(); onSelect(node.id); }}
          title="First matching condition decides the branch — click to edit the conditions">
          <span className="ft-route-if">{label.word}</span>
          <span className="ft-route-pred">{label.pred}</span>
          {acts}
        </div>
        <div className="ft-route-target">
          {sub
            ? <FilterTreeNode node={sub} {...passProps} plain={sub.kind === "Sequential"} />
            : <RouteSlot onPick={(n) => onAddToRoute(node.id, slotId, n)} />}
        </div>
      </div>
    );
    return (
      <div className={"ft-route-group" + (selected ? " is-selected" : "")}>
        {routes.map((r, i) => slot(
          { word: i === 0 ? "IF" : "ELIF", pred: predicateSummary(r.if) }, r.pass_to, r.id,
          i === 0
            ? <FilterNodeActs node={node} canUp={canUp} canDown={canDown} onMove={onMove} onRemove={onRemove} />
            : null))}
        {slot({ word: "ELSE", pred: "everything else" }, node.else, "__else", null)}
      </div>
    );
  }

  const kids = node.kind === "Not"
    ? (node.layer ? [node.layer] : [])
    : (node.children || []);
  const kidsMovable = node.kind !== "Not" && kids.length > 1;

  // ── Trivial Sequential wrappers (the root cascade and Route branch
  // bodies) render as a bare list — AND-in-order is implied, so a
  // "MATCH ALL" header level only added noise.
  if (node.kind === "Sequential" && (plain || depth === 0)) {
    return (
      <div className="ft-seq-plain">
        {kids.map((c, i) => (
          <FilterTreeNode
            key={c.id}
            node={c} depth={depth + 1}
            selectedId={selectedId}
            onSelect={onSelect}
            onAddChild={onAddChild}
            onRemove={onRemove}
            onMove={onMove}
            onAddToRoute={onAddToRoute}
            canUp={kidsMovable && i > 0}
            canDown={kidsMovable && i < kids.length - 1}
          />
        ))}
        <div className="ft-add-wrap">
          <button ref={addBtnRef} className="ft-add" onClick={(e) => { e.stopPropagation(); setAdding(true); }}>
            <Icon name="plus" size={11} /> Add filter
          </button>
          {adding && (
            <AddFilterPopover
              allowComposite
              anchorRef={addBtnRef}
              onPick={(n) => onAddChild(node.id, n)}
              onClose={() => setAdding(false)}
            />
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={"ft-comp ft-comp-" + node.kind.toLowerCase() + (selected ? " is-selected" : "")}>
      <div className="ft-comp-head" onClick={(e) => { e.stopPropagation(); onSelect(node.id); }}
        title={(COMPOSITE_META[node.kind] || "") + " · " + kids.length + (kids.length === 1 ? " filter" : " filters")}>
        <span className="ft-comp-kind">{COMPOSITE_LABEL[node.kind] || node.kind}</span>
        <span className="ft-comp-meta"><span className="ft-comp-meta-txt">
          {COMPOSITE_META[node.kind] ? COMPOSITE_META[node.kind] + " · " : ""}
          {kids.length} {kids.length === 1 ? "filter" : "filters"}
        </span></span>
        <FilterNodeActs node={node} canUp={canUp} canDown={canDown} onMove={onMove} onRemove={onRemove} />
      </div>
      <div className={"ft-comp-body" + (node.kind === "Parallel" ? " ft-comp-body-parallel" : "")}>
        {kids.map((c, i) => (
          <FilterTreeNode
            key={c.id}
            node={c} depth={depth + 1}
            selectedId={selectedId}
            onSelect={onSelect}
            onAddChild={onAddChild}
            onRemove={onRemove}
            onMove={onMove}
            onAddToRoute={onAddToRoute}
            canUp={kidsMovable && i > 0}
            canDown={kidsMovable && i < kids.length - 1}
          />
        ))}
        {(node.kind !== "Not" || kids.length === 0) && (
          <div className="ft-add-wrap">
            <button ref={addBtnRef} className="ft-add" onClick={(e) => { e.stopPropagation(); setAdding(true); }}>
              <Icon name="plus" size={11} /> Add {node.kind === "Not" ? "inner layer" : "child"}
            </button>
            {adding && (
              <AddFilterPopover
                allowComposite
                anchorRef={addBtnRef}
                onPick={(n) => onAddChild(node.id, n)}
                onClose={() => setAdding(false)}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function filterSummary(n) {
  const p = n.params || {};
  const queryText = (p) => {
    if (!p.queries) return p.formula || "";
    const keys = Object.keys(p.queries);
    if (keys.length === 1 && p.queries[keys[0]]) {
      return `"${p.queries[keys[0]]}"`;
    }
    const vals = Object.values(p.queries).filter(Boolean);
    if (vals.length === 0) return p.formula || "";
    return `${vals.length} conditions · "${vals[0]}" …`;
  };
  const keywordText = (p) => {
    if (p.expression != null) return p.expression;
    if (!p.keywords) return p.formula || p.keyword || "";
    const vals = Object.values(p.keywords).filter(Boolean);
    if (vals.length === 0) return p.formula || "";
    return vals.join(" · ");
  };
  const matchLabel = {
    substring: "contains",
    whole_word: "word",
    starts_with: "starts with",
  };
  const scopeLabel = {
    title: "Title",
    title_abstract: "Abstract",
    venue: "Venue",
    full_text: "Full text",
  };
  switch (n.kind) {
    case "Route":                 return `first match wins · ${(n.routes || []).length} condition${(n.routes || []).length === 1 ? "" : "s"} + else`;
    case "YearFilter":            return `${p.min ?? "…"} – ${p.max ?? "…"}`;
    case "CitationFilter":        return `β = ${p.beta ?? "…"} cites/yr of age`;
    case "SimilarityFilter":      return `similarity ≥ ${p.threshold ?? ""} · ${(p.measures || []).length} measures`;
    case "LLMFilter":             return `${scopeLabel[p.scope] || p.scope} · ${queryText(p)}${p.model ? ` · ${p.model}` : ""}`;
    case "TitleKeywordFilter":    return `Title ${matchLabel[p.match] || p.match} ${keywordText(p)}`;
    case "AbstractKeywordFilter": return `Abstract ${matchLabel[p.match] || p.match} ${keywordText(p)}`;
    case "VenueKeywordFilter":    return `Venue ${matchLabel[p.match] || p.match} ${keywordText(p)}`;
    default: return n.kind;
  }
}

function FilterTree({ screener, selectedId, onSelect, onAddRoot, onAddChild, onRemove, onMove, onAddToRoute }) {
  const [adding, setAdding] = React.useState(false);
  const rootAddRef = React.useRef(null);

  if (!screener) {
    return (
      <div className="ft-empty">
        <em>No screener defined.</em>
        <div className="ft-add-wrap" style={{ marginTop: 10 }}>
          <button ref={rootAddRef} className="ft-add" onClick={() => setAdding(true)}>
            <Icon name="plus" size={11} /> Add first filter
          </button>
          {adding && (
            <AddFilterPopover
              allowComposite
              anchorRef={rootAddRef}
              onPick={(n) => onAddRoot(n)}
              onClose={() => setAdding(false)}
            />
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="ft-tree">
      <FilterTreeNode
        node={screener}
        depth={0}
        selectedId={selectedId}
        onSelect={onSelect}
        onAddChild={onAddChild}
        onRemove={onRemove}
        onMove={onMove || (() => {})}
        onAddToRoute={onAddToRoute || (() => {})}
      />
    </div>
  );
}

// One Route condition row: predicate kind + value, reorder, remove. The
// venue list edits through a local draft (committed on blur/Enter) so typing
// commas doesn't fight the parser.
function RouteCondRow({ route, index, count, onPatch, onMove, onRemove }) {
  const p = route.if || {};
  const joined = (p.values || []).join(", ");
  const [venueDraft, setVenueDraft] = React.useState(joined);
  React.useEffect(() => { setVenueDraft(joined); }, [route.id, joined]);
  const commitVenues = () => {
    const vs = venueDraft.split(",").map(s => s.trim()).filter(Boolean);
    onPatch({ if: { kind: "VenueIn", values: vs } });
  };
  const setKind = (k) => {
    if (k === "VenueIn") onPatch({ if: { kind: "VenueIn", values: p.values || [] } });
    else onPatch({ if: { kind: k, n: p.n ?? (k === "YearAtLeast" ? 2020 : 100) } });
  };
  return (
    <div className="cfg-route-row">
      <span className="cfg-route-word">{index === 0 ? "IF" : "ELIF"}</span>
      <select className="cfg-route-kind" value={p.kind || "VenueIn"} onChange={e => setKind(e.target.value)}>
        <option value="VenueIn">venue contains</option>
        <option value="CitAtLeast">citations ≥</option>
        <option value="YearAtLeast">year ≥</option>
      </select>
      {(p.kind || "VenueIn") === "VenueIn" ? (
        <input className="cfg-route-val" value={venueDraft} placeholder="Nature, Science, Cell"
          onChange={e => setVenueDraft(e.target.value)}
          onBlur={commitVenues}
          onKeyDown={e => { if (e.key === "Enter") e.target.blur(); }}
          title="Comma-separated venue substrings, matched case-insensitively" />
      ) : (
        <input className="cfg-route-val" type="number" value={p.n ?? 0}
          onChange={e => onPatch({ if: { kind: p.kind, n: Math.max(0, +e.target.value || 0) } })} />
      )}
      <span className="cfg-route-acts">
        <button className="ft-act" disabled={index === 0} title="Check earlier"
          onClick={() => onMove(-1)}><Icon name="arrow-up" size={11} /></button>
        <button className="ft-act" disabled={index >= count - 1} title="Check later"
          onClick={() => onMove(+1)}><Icon name="arrow-down" size={11} /></button>
        <button className="cfg-measure-del" onClick={onRemove}
          title="Remove this condition — its branch filters go with it">×</button>
      </span>
    </div>
  );
}

// --- Right-side inspector panes ------------------------------------------
function FilterParams({ node, onPatch, onPatchNode }) {
  if (!node) return null;
  const p = node.params || {};
  const set = (k, v) => onPatch({ ...p, [k]: v });

  switch (node.kind) {
    case "Route": {
      const routes = node.routes || [];
      const setRoutes = (next) => onPatchNode({ routes: next });
      const patchRow = (i, patch) => setRoutes(routes.map((r, j) => (j === i ? { ...r, ...patch } : r)));
      const moveRow = (i, dir) => {
        const j = i + dir;
        if (j < 0 || j >= routes.length) return;
        const next = routes.slice();
        [next[i], next[j]] = [next[j], next[i]];
        setRoutes(next);
      };
      const removeRow = (i) => setRoutes(routes.filter((_, j) => j !== i));
      const addRow = () => setRoutes([...routes, { id: nextFid(), if: { kind: "VenueIn", values: [] }, pass_to: null }]);
      return (
        <>
          <div className="cfg-note">
            <Icon name="info" size={13} />
            <span>
              Conditions are checked <strong>top to bottom</strong> — the first
              match decides which branch screens the paper; papers matching
              none take the <strong>ELSE</strong> branch. The branches
              themselves live in the filter tree (go Back and expand the
              Route).
            </span>
          </div>
          <ConfigField label="Conditions" full>
            <div className="cfg-kv-list">
              {routes.map((r, i) => (
                <RouteCondRow key={r.id} route={r} index={i} count={routes.length}
                  onPatch={(patch) => patchRow(i, patch)}
                  onMove={(dir) => moveRow(i, dir)}
                  onRemove={() => removeRow(i)} />
              ))}
              <button className="cfg-measure-add" onClick={addRow}>
                <Icon name="plus" size={11} /> Add condition
              </button>
            </div>
          </ConfigField>
        </>
      );
    }
    case "YearFilter":
      return (
        <>
          <ConfigField label="Min year" info="Papers published before this year are rejected. Can't go above the max year.">
            <NumStepper value={p.min} lo={1900} hi={p.max ?? NOW_YEAR} onChange={v => set("min", v)} />
          </ConfigField>
          <ConfigField label="Max year" info="Papers published after this year are rejected. Can't go below the min year, and can't exceed the current year.">
            <NumStepper value={p.max} lo={p.min ?? 1900} hi={NOW_YEAR} onChange={v => set("max", v)} />
          </ConfigField>
        </>
      );
    case "CitationFilter":
      return (
        <>
          <ConfigField label="β" hint="min cites per year of age"
            info="Minimum citations required per year since publication — NOT an absolute count. A paper passes when its citation count ≥ β × max(1, age in years), so young papers clear a lower bar than old ones.">
            <input type="number" step="5" value={p.beta} onChange={e => set("beta", +e.target.value)} />
          </ConfigField>
          <ConfigField label="Formula" wide hint="cites ≥ β · max(1, age)">
            <input disabled value={`cites >= ${p.beta ?? "β"} * max(1, ${NOW_YEAR} - year)`} />
          </ConfigField>
        </>
      );
    case "SimilarityFilter": {
      const measures = p.measures || [];
      const setKind = (i, kind) => set("measures", measures.map((m, j) => j === i ? measureDefaults(kind) : m));
      const patchMeasure = (i, patch) => set("measures", measures.map((m, j) => j === i ? { ...m, ...patch } : m));
      const removeMeasure = (i) => set("measures", measures.filter((_, j) => j !== i));
      const addMeasure = () => set("measures", [...measures, measureDefaults("RefSim")]);
      return (
        <>
          <ConfigField label="Threshold" info="A paper passes when its best measure score is at least this value (scores run 0–1).">
            <input type="number" step="0.005" min="0" max="1" value={p.threshold}
              onChange={e => set("threshold", +e.target.value)} />
          </ConfigField>
          <ConfigField label="No data" info="What to do when no measure can score a paper (e.g. it has no references or embedding).">
            <select value={p.on_no_data || "pass"} onChange={e => set("on_no_data", e.target.value)}>
              <option value="pass">pass through</option>
              <option value="reject">reject</option>
            </select>
          </ConfigField>
          <ConfigField label="Measures" full hint="Max over normalized scores"
            info="Each measure scores a paper 0–1 against the anchor; the filter keeps the highest score.">
            <div className="cfg-measure-list">
              {measures.map((m, i) => (
                <div key={i} className="cfg-measure2">
                  <div className="cfg-measure2-head">
                    <select className="cfg-measure2-kind" value={m.kind} onChange={e => setKind(i, e.target.value)}>
                      <option value="RefSim">RefSim</option>
                      <option value="CitSim">CitSim</option>
                      <option value="SemanticSim">SemanticSim</option>
                    </select>
                    <InfoDot text={MEASURE_INFO[m.kind]} />
                    <button className="cfg-measure-del" onClick={() => removeMeasure(i)} title="Remove measure">×</button>
                  </div>
                  {m.kind === "CitSim" && (
                    <label className="cfg-measure2-param">
                      <span>Cited at least</span>
                      <input type="number" min="0" step="10" value={m.pass_if_cited_at_least ?? 200}
                        onChange={e => patchMeasure(i, { pass_if_cited_at_least: Math.max(0, +e.target.value || 0) })} />
                    </label>
                  )}
                  {m.kind === "SemanticSim" && (
                    <>
                      <label className="cfg-measure2-param">
                        <span>Embedder</span>
                        <select value={m.embedder || "s2"} onChange={e => patchMeasure(i, { embedder: e.target.value })}>
                          <option value="s2">s2 · SPECTER2</option>
                          <option value="voyage">voyage</option>
                        </select>
                      </label>
                      {m.embedder === "voyage" && <VoyageKeyField />}
                    </>
                  )}
                  {m.kind === "RefSim" && <div className="cfg-measure2-none">No parameters.</div>}
                </div>
              ))}
              <button className="cfg-measure-add" onClick={addMeasure}>
                <Icon name="plus" size={11} /> Add measure
              </button>
            </div>
          </ConfigField>
        </>
      );
    }
    case "LLMFilter":
      return (
        <>
          <ConfigField label="Scope" info="What text the model reads when judging each paper.">
            <select value={p.scope} onChange={e => set("scope", e.target.value)}>
              <option value="title">Title</option>
              <option value="title_abstract">Abstract</option>
              <option value="venue">Venue</option>
              <option value="full_text">Full text</option>
            </select>
          </ConfigField>
          <ConfigField label="Model"
            info="Which LLM screens THIS filter. Default follows Settings; overriding lets each filter pick its own model — e.g. cheap triage on titles, a stronger model on abstracts. The run checks that the matching API key is set.">
            <ModelOverrideSelect value={p.model || ""} onChange={v => set("model", v)} />
          </ConfigField>
          <ConfigField label="Effort"
            info="Reasoning effort for this filter only. Default follows Settings.">
            <select value={p.effort || ""} onChange={e => set("effort", e.target.value)}>
              <option value="">Default (Settings)</option>
              {["minimal", "low", "medium", "high"].map(x => (
                <option key={x} value={x}>{x}</option>
              ))}
            </select>
          </ConfigField>
          <NamedDictEditor p={p} patch={(obj) => onPatch({ ...p, ...obj })} field="queries" noun="query"
            formulaInfo="Boolean expression over the named questions below." />
        </>
      );
    case "TitleKeywordFilter":
    case "AbstractKeywordFilter":
    case "VenueKeywordFilter":
      return (
        <>
          <ConfigField label="Match" info="How each literal string is compared to the text: as a substring (anywhere), a whole word, or a prefix (starts-with).">
            <select value={p.match} onChange={e => set("match", e.target.value)}>
              <option value="substring">substring</option>
              <option value="whole_word">whole word</option>
              <option value="starts_with">starts with</option>
            </select>
          </ConfigField>
          <KeywordExpressionEditor p={p} patch={(obj) => onPatch({ ...p, ...obj })} />
        </>
      );
    default:
      return <div className="cfg-empty">Composite container · no parameters.</div>;
  }
}

function BlockParams({ node, onPatchConfig }) {
  const c = node.config || {};
  if (node.kind === "seed") {
    return (
      <>
        <ConfigField label="Query" wide hint="Free-text or boolean · Semantic Scholar">
          <input value={c.query} onChange={e => onPatchConfig({ query: e.target.value })} />
        </ConfigField>
        <ConfigField label="Year window">
          <select value={c.years} onChange={e => onPatchConfig({ years: e.target.value })}>
            <option>2015-2025</option><option>2019-2025</option><option>2022-2025</option>
          </select>
        </ConfigField>
        <ConfigField label="Max seeds"><input type="number" value={c.maxSeeds} onChange={e => onPatchConfig({ maxSeeds: +e.target.value })} /></ConfigField>
      </>
    );
  }
  if (node.kind === "fwd") {
    return (
      <ConfigField label="Max citing papers" wide
        hint="per source · highest-cited kept first"
        info="For each source paper, keep only this many citing papers — the highest-cited ones — before screening. This is the CLI's max_citations (default 100).">
        <input type="number" min="1" step="10" value={c.maxCitations ?? 100}
          onChange={e => onPatchConfig({ maxCitations: Math.max(1, +e.target.value || 1) })} />
      </ConfigField>
    );
  }
  if (node.kind === "bwd") {
    return (
      <div className="cfg-note">
        <Icon name="info" size={13} />
        <span>Backward screening walks <strong>every</strong> reference of each paper — there is no fan-out cap. Shape what's kept with the filter pipeline below.</span>
      </div>
    );
  }
  if (node.kind === "rerank") {
    const div = c.diversity || "off";
    return (
      <>
        <ConfigField label="Metric"
          info="How each paper is scored before the top-K cut. citation = raw citation count; pagerank = PageRank centrality over the collected citation graph.">
          <select value={c.metric || "citation"} onChange={e => onPatchConfig({ metric: e.target.value })}>
            <option value="citation">citation · raw count</option>
            <option value="pagerank">pagerank · graph centrality</option>
          </select>
        </ConfigField>
        <ConfigField label="Keep top-K" info="How many of the highest-scored papers to keep (the CLI's k).">
          <input type="number" min="1" step="10" value={c.targetN ?? 100}
            onChange={e => onPatchConfig({ targetN: Math.max(1, +e.target.value || 1) })} />
        </ConfigField>
        <ConfigField label="Diversity" wide
          info="Off keeps a plain top-K. Otherwise the signal is clustered on the spot and the top-K is spread across the clusters (floor-then-proportional) so one dense topic can't crowd out the rest. louvain / walktrap partition the citation graph; topic model clusters SPECTER2 embeddings (UMAP + HDBSCAN).">
          <select value={div} onChange={e => onPatchConfig({ diversity: e.target.value })}>
            <option value="off">off · plain top-K</option>
            <option value="louvain">cluster-diverse · louvain</option>
            <option value="walktrap">cluster-diverse · walktrap</option>
            <option value="topic_model">cluster-diverse · topic model</option>
          </select>
        </ConfigField>
        {div === "walktrap" && (
          <ConfigField label="Communities" info="How many communities Walktrap targets (the CLI's n_communities).">
            <input type="number" min="2" step="1" value={c.divCommunities ?? 3}
              onChange={e => onPatchConfig({ divCommunities: Math.max(2, +e.target.value || 2) })} />
          </ConfigField>
        )}
        {div === "topic_model" && (
          <>
            <ConfigField label="Min cluster" info="HDBSCAN's minimum cluster size (the CLI's min_cluster_size).">
              <input type="number" min="2" step="1" value={c.divMinCluster ?? 5}
                onChange={e => onPatchConfig({ divMinCluster: Math.max(2, +e.target.value || 2) })} />
            </ConfigField>
            <ConfigField label="Neighbors" info="UMAP's n_neighbors. Leave 0 for the adaptive default.">
              <input type="number" min="0" step="1" value={c.divNeighbors ?? 0}
                onChange={e => onPatchConfig({ divNeighbors: Math.max(0, +e.target.value || 0) })} />
            </ConfigField>
            <div className="cfg-note">
              <Icon name="info" size={13} />
              <span>Topic-model clustering needs the optional extra
                (<code>pip install 'citeclaw[topic_model]'</code>) on the machine
                running the server — the run stops with a clear error if it's
                missing.</span>
            </div>
          </>
        )}
      </>
    );
  }
  if (node.kind === "search") {
    return (
      <>
        <div className="cfg-note cfg-note-warn">
          <Icon name="alert-triangle" size={13} />
          <span><strong>Experimental.</strong> The CLI's search agent is a
            placeholder right now — this step is wired up end-to-end, but
            don't expect strong results yet. It asks the LLM to design
            Semantic Scholar queries from the topic and screens whatever
            comes back with the filter pipeline below.</span>
        </div>
        <ConfigField label="Max iterations"
          info="How many query-design → search → screen rounds the agent may run (the CLI's agent.max_iterations).">
          <NumStepper value={c.maxIterations ?? 4} lo={1} hi={10} onChange={v => onPatchConfig({ maxIterations: v })} />
        </ConfigField>
        <ConfigField label="Papers / iteration" wide hint="cap on candidates fetched per round"
          info="Total candidate cap per iteration (the CLI's agent.max_papers_per_iteration). Each 1000 costs one Semantic Scholar request.">
          <input type="number" min="100" step="100" value={c.maxPerIteration ?? 10000}
            onChange={e => onPatchConfig({ maxPerIteration: Math.max(100, +e.target.value || 100) })} />
        </ConfigField>
      </>
    );
  }
  return <div className="cfg-empty">No parameters for this block kind.</div>;
}

function GlobalParams() {
  const [params, setParams] = React.useState({
    maxTokens: 4096,
    maxPapers: 2000,
    parallelism: 8,
    timeoutSec: 60,
    llmModel: "claude-haiku-4.5",
    embedModel: "mxbai-embed-large",
    s2Key: "sk-s2-********4c2a",
    openaiKey: "sk-oai-********9f1e",
    anthropicKey: "sk-ant-********7b3d",
    cacheEnabled: true,
    retryFailures: true,
  });
  const set = (patch) => setParams(p => ({ ...p, ...patch }));
  const mask = (v) => v ? v.slice(0, 6) + "••••" + v.slice(-4) : "";
  return (
    <div className="gp-wrap">
      <div className="gp-section">
        <div className="gp-section-head">Budgets</div>
        <div className="gp-grid">
          <label className="gp-kv">
            <span className="gp-k">max_tokens</span>
            <input className="gp-v gp-v-num" type="number" value={params.maxTokens}
              onChange={e => set({ maxTokens: +e.target.value })} />
          </label>
          <label className="gp-kv">
            <span className="gp-k">max_papers</span>
            <input className="gp-v gp-v-num" type="number" value={params.maxPapers}
              onChange={e => set({ maxPapers: +e.target.value })} />
          </label>
          <label className="gp-kv">
            <span className="gp-k">parallelism</span>
            <input className="gp-v gp-v-num" type="number" value={params.parallelism}
              onChange={e => set({ parallelism: +e.target.value })} />
          </label>
          <label className="gp-kv">
            <span className="gp-k">timeout_sec</span>
            <input className="gp-v gp-v-num" type="number" value={params.timeoutSec}
              onChange={e => set({ timeoutSec: +e.target.value })} />
          </label>
        </div>
      </div>

      <div className="gp-section">
        <div className="gp-section-head">Models</div>
        <div className="gp-grid gp-grid-stack">
          <label className="gp-kv">
            <span className="gp-k">llm_model</span>
            <select className="gp-v" value={params.llmModel}
              onChange={e => set({ llmModel: e.target.value })}>
              <option>claude-haiku-4.5</option>
              <option>claude-sonnet-4.5</option>
              <option>claude-opus-4</option>
              <option>gpt-4o</option>
              <option>gpt-4o-mini</option>
            </select>
          </label>
          <label className="gp-kv">
            <span className="gp-k">embed_model</span>
            <select className="gp-v" value={params.embedModel}
              onChange={e => set({ embedModel: e.target.value })}>
              <option>mxbai-embed-large</option>
              <option>text-embedding-3-large</option>
              <option>voyage-large-2</option>
            </select>
          </label>
        </div>
      </div>

      <div className="gp-section">
        <div className="gp-section-head">API keys</div>
        <div className="gp-grid gp-grid-stack">
          <label className="gp-kv gp-kv-key">
            <span className="gp-k">semantic_scholar</span>
            <input className="gp-v gp-v-key" type="text" readOnly value={mask(params.s2Key)} />
          </label>
          <label className="gp-kv gp-kv-key">
            <span className="gp-k">openai</span>
            <input className="gp-v gp-v-key" type="text" readOnly value={mask(params.openaiKey)} />
          </label>
          <label className="gp-kv gp-kv-key">
            <span className="gp-k">anthropic</span>
            <input className="gp-v gp-v-key" type="text" readOnly value={mask(params.anthropicKey)} />
          </label>
          <button className="gp-key-btn">
            <Icon name="key" size={11} /> Manage keys…
          </button>
        </div>
      </div>

      <div className="gp-section">
        <div className="gp-section-head">Execution</div>
        <div className="gp-grid gp-grid-stack">
          <label className="gp-toggle">
            <input type="checkbox" checked={params.cacheEnabled}
              onChange={e => set({ cacheEnabled: e.target.checked })} />
            <span className="gp-toggle-lbl">Enable response cache</span>
          </label>
          <label className="gp-toggle">
            <input type="checkbox" checked={params.retryFailures}
              onChange={e => set({ retryFailures: e.target.checked })} />
            <span className="gp-toggle-lbl">Retry failed fetches (×3)</span>
          </label>
        </div>
      </div>
    </div>
  );
}

function BuildConfig({ node, onPatchConfig, onUpdateScreener }) {
  const [selectedFilterId, setSelectedFilterId] = React.useState(null);

  // Reset selection when switching blocks
  React.useEffect(() => { setSelectedFilterId(null); }, [node?.id]);

  if (!node) {
    return (
      <section className="config config-split">
        <div className="config-empty">
          <em>Select a pipeline block to configure its parameters.</em>
        </div>
      </section>
    );
  }

  const isScreener = node.kind === "fwd" || node.kind === "bwd";
  const selectedFilter = isScreener && node.screener ? findNode(node.screener, selectedFilterId) : null;

  // Root-level screener ops
  const setScreener = (t) => onUpdateScreener(t);
  const addRoot = (child) => setScreener(child);
  const addChild = (parentId, child) => {
    setScreener(mapTree(node.screener, parentId, (p) => {
      if (p.kind === "Not") return { ...p, layer: child };
      return { ...p, children: [...(p.children || []), child] };
    }));
    setSelectedFilterId(child.id);
  };
  const removeFilter = (id) => {
    if (node.screener && node.screener.id === id) { setScreener(null); setSelectedFilterId(null); return; }
    setScreener(removeFromTree(node.screener, id));
    if (selectedFilterId === id) setSelectedFilterId(null);
  };
  const patchFilter = (id, params) => {
    setScreener(mapTree(node.screener, id, (n) => ({ ...n, params })));
  };

  return (
    <section className="config config-split">
      <div className="config-head">
        <div className="config-head-left">
          <div className="config-title">{node.name}</div>
          <span className="config-sub">{node.localId} · {node.kind}</span>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <button className="btn btn-ghost" title="Duplicate">
            <Icon name="copy" size={12} /> Duplicate
          </button>
          <button className="btn btn-ghost" title="Remove">
            <Icon name="trash-2" size={12} />
          </button>
        </div>
      </div>

      <div className={"config-cols" + (isScreener ? "" : " is-no-filter")}>
        {/* LEFT: vertical filter tree (or block summary for non-screener blocks) */}
        {isScreener && (
        <div className="config-col config-col-tree">
          <div className="config-col-head">
            <span className="config-col-title">
              Filter pipeline
            </span>
            <span className="config-col-sub"></span>
          </div>
          <div className="config-col-body">
            <FilterTree
                screener={node.screener}
                selectedId={selectedFilterId}
                onSelect={setSelectedFilterId}
                onAddRoot={(n) => { setScreener(n); setSelectedFilterId(n.id); }}
                onAddChild={addChild}
                onRemove={removeFilter}
              />
          </div>
        </div>
        )}

        {/* MIDDLE: parameters for the selected node (filter or block) */}
        <div className="config-col config-col-inspect">
          <div className="config-col-head">
            <span className="config-col-title">
              {selectedFilter ? selectedFilter.kind : node.name}
            </span>
            <span className="config-col-sub">
              {selectedFilter ? "filter parameters" : (isScreener ? "screener parameters" : "block parameters")}
            </span>
          </div>
          <div className="config-col-body config-inspect-grid">
            {selectedFilter
              ? <FilterParams node={selectedFilter} onPatch={(params) => patchFilter(selectedFilter.id, params)} />
              : <BlockParams node={node} onPatchConfig={onPatchConfig} />}
          </div>
        </div>

        {/* RIGHT: Global run parameters (shared across all blocks) */}
        <div className="config-col config-col-globals">
          <div className="config-col-head">
            <span className="config-col-title">Global parameters</span>
            <span className="config-col-sub">run-wide settings</span>
          </div>
          <div className="config-col-body">
            <GlobalParams />
          </div>
        </div>
      </div>
    </section>
  );
}

Object.assign(window, { BuildConfig, GlobalParams });
