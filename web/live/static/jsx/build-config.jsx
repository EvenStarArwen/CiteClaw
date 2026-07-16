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
  { kind: "Sequential", label: "Sequential", desc: "AND · short-circuit" },
  { kind: "Any",        label: "Any",        desc: "OR · short-circuit" },
  { kind: "Not",        label: "Not",        desc: "invert one child" },
  { kind: "Route",      label: "Route",      desc: "if / elif / else" },
  { kind: "Parallel",   label: "Parallel",   desc: "broadcast · union outputs" },
];

// --- tree helpers ---------------------------------------------------------
let __fid = 1000;
const nextFid = () => "f" + (++__fid);
function defaultParams(kind) {
  switch (kind) {
    case "YearFilter":            return { min: 2019, max: 2025 };
    case "CitationFilter":        return { beta: 30 };
    case "SimilarityFilter":      return { threshold: 0.025, measures: [{ kind: "RefSim" }] };
    case "LLMFilter":             return { scope: "title_abstract", formula: "q1", queries: { q1: "" } };
    case "TitleKeywordFilter":    return { match: "substring", formula: "k1", keywords: { k1: "" } };
    case "AbstractKeywordFilter": return { match: "substring", formula: "k1", keywords: { k1: "" } };
    case "VenueKeywordFilter":    return { match: "starts_with", formula: "k1", keywords: { k1: "Nature" } };
    default: return {};
  }
}
function newNode(kind) {
  if (COMPOSITE_KINDS.includes(kind)) {
    if (kind === "Not") return { id: nextFid(), kind, layer: { id: nextFid(), kind: "YearFilter", params: defaultParams("YearFilter") } };
    if (kind === "Route") return { id: nextFid(), kind, routes: [], else: null };
    return { id: nextFid(), kind, children: [] };
  }
  return { id: nextFid(), kind, params: defaultParams(kind) };
}
// Recursively find + update + remove nodes by id
function findNode(t, id) {
  if (!t) return null;
  if (t.id === id) return t;
  for (const c of (t.children || [])) { const r = findNode(c, id); if (r) return r; }
  if (t.layer) { const r = findNode(t.layer, id); if (r) return r; }
  return null;
}
function mapTree(t, id, fn) {
  if (!t) return t;
  if (t.id === id) return fn(t);
  if (t.children) return { ...t, children: t.children.map(c => mapTree(c, id, fn)) };
  if (t.layer)    return { ...t, layer: mapTree(t.layer, id, fn) };
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
  return t;
}

// --- components -----------------------------------------------------------

function ConfigField({ label, children, wide, full, hint }) {
  return (
    <div className={"field" + (wide ? " field-wide" : "") + (full ? " field-full" : "")}>
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </div>
  );
}

function AddFilterPopover({ onPick, onClose, allowComposite = true, anchorRef }) {
  const [pos, setPos] = React.useState(null);
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
      const popW = 280, popH = 360;
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

function FilterTreeNode({ node, depth, selectedId, onSelect, onAddChild, onRemove }) {
  const [adding, setAdding] = React.useState(false);
  const addBtnRef = React.useRef(null);
  const isComposite = COMPOSITE_KINDS.includes(node.kind);
  const selected = selectedId === node.id;

  if (!isComposite) {
    return (
      <div className={"ft-leaf" + (selected ? " is-selected" : "")} onClick={(e) => { e.stopPropagation(); onSelect(node.id); }}>
        <span className="ft-leaf-kind">{node.kind.replace("Filter", "")}</span>
        <span className="ft-leaf-body">{filterSummary(node)}</span>
        <button className="ft-leaf-x" onClick={(e) => { e.stopPropagation(); onRemove(node.id); }} title="Remove">×</button>
      </div>
    );
  }

  const kids = node.kind === "Not"
    ? (node.layer ? [node.layer] : [])
    : (node.children || []);

  return (
    <div className={"ft-comp ft-comp-" + node.kind.toLowerCase() + (selected ? " is-selected" : "")}>
      <div className="ft-comp-head" onClick={(e) => { e.stopPropagation(); onSelect(node.id); }}>
        <span className="ft-comp-kind">{node.kind}</span>
        <span className="ft-comp-meta">{kids.length} {kids.length === 1 ? "child" : "children"}</span>
        <button className="ft-leaf-x" onClick={(e) => { e.stopPropagation(); onRemove(node.id); }} title="Remove">×</button>
      </div>
      <div className={"ft-comp-body" + (node.kind === "Parallel" ? " ft-comp-body-parallel" : "")}>
        {kids.map(c => (
          <FilterTreeNode
            key={c.id}
            node={c} depth={depth + 1}
            selectedId={selectedId}
            onSelect={onSelect}
            onAddChild={onAddChild}
            onRemove={onRemove}
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
    if (!p.keywords) return p.formula || "";
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
    title_abstract: "Title & abstract",
    venue: "Venue",
    full_text: "Full text",
  };
  switch (n.kind) {
    case "YearFilter":            return `${p.min ?? "…"} – ${p.max ?? "…"}`;
    case "CitationFilter":        return `at least ${p.beta ?? "…"} citations`;
    case "SimilarityFilter":      return `similarity ≥ ${p.threshold ?? ""} · ${(p.measures || []).length} measures`;
    case "LLMFilter":             return `${scopeLabel[p.scope] || p.scope} · ${queryText(p)}`;
    case "TitleKeywordFilter":    return `Title ${matchLabel[p.match] || p.match} ${keywordText(p)}`;
    case "AbstractKeywordFilter": return `Abstract ${matchLabel[p.match] || p.match} ${keywordText(p)}`;
    case "VenueKeywordFilter":    return `Venue ${matchLabel[p.match] || p.match} ${keywordText(p)}`;
    default: return n.kind;
  }
}

function FilterTree({ screener, selectedId, onSelect, onAddRoot, onAddChild, onRemove }) {
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
      />
    </div>
  );
}

// --- Right-side inspector panes ------------------------------------------
function FilterParams({ node, onPatch }) {
  if (!node) return null;
  const p = node.params || {};
  const set = (k, v) => onPatch({ ...p, [k]: v });

  switch (node.kind) {
    case "YearFilter":
      return (
        <>
          <ConfigField label="Min year"><input type="number" value={p.min} onChange={e => set("min", +e.target.value)} /></ConfigField>
          <ConfigField label="Max year"><input type="number" value={p.max} onChange={e => set("max", +e.target.value)} /></ConfigField>
          <ConfigField label="Preset" wide>
            <select onChange={e => onPatch({ min: +e.target.value.split("-")[0], max: +e.target.value.split("-")[1] })} value={`${p.min}-${p.max}`}>
              <option>2015-2025</option><option>2019-2025</option><option>2022-2025</option>
            </select>
          </ConfigField>
        </>
      );
    case "CitationFilter":
      return (
        <>
          <ConfigField label="β" hint="min cites per year of age">
            <input type="number" step="5" value={p.beta} onChange={e => set("beta", +e.target.value)} />
          </ConfigField>
          <ConfigField label="Formula" wide hint="cites ≥ β · max(1, age)">
            <input disabled value="cites >= beta * max(1, 2026 - year)" />
          </ConfigField>
        </>
      );
    case "SimilarityFilter":
      return (
        <>
          <ConfigField label="Threshold"><input type="number" step="0.005" value={p.threshold} onChange={e => set("threshold", +e.target.value)} /></ConfigField>
          <ConfigField label="Measures" full hint="Max over normalized scores">
            <div className="cfg-measure-list">
              {(p.measures || []).map((m, i) => (
                <div key={i} className="cfg-measure">
                  <span className="cfg-measure-kind">{m.kind}</span>
                  <span className="cfg-measure-params">
                    {m.kind === "CitSim" && <>cited ≥ {m.pass_if_cited_at_least ?? "—"}</>}
                    {m.kind === "SemanticSim" && <>embedder: {m.embedder ?? "s2"}</>}
                  </span>
                  <button className="cfg-measure-del" onClick={() => set("measures", p.measures.filter((_, j) => j !== i))}>×</button>
                </div>
              ))}
              <button className="cfg-measure-add" onClick={() => set("measures", [...(p.measures || []), { kind: "RefSim" }])}>
                <Icon name="plus" size={11} /> Add measure
              </button>
            </div>
          </ConfigField>
        </>
      );
    case "LLMFilter":
      return (
        <>
          <ConfigField label="Scope">
            <select value={p.scope} onChange={e => set("scope", e.target.value)}>
              <option>title</option><option>title_abstract</option><option>venue</option><option>full_text</option>
            </select>
          </ConfigField>
          <ConfigField label="Formula" wide hint="Boolean over named queries — & | !">
            <input value={p.formula} onChange={e => set("formula", e.target.value)} />
          </ConfigField>
          <ConfigField label="Queries" full hint="name → prompt">
            <div className="cfg-kv-list">
              {Object.entries(p.queries || {}).map(([k, v]) => (
                <div key={k} className="cfg-kv">
                  <input className="cfg-kv-k" value={k} readOnly />
                  <input className="cfg-kv-v" value={v} onChange={e => set("queries", { ...p.queries, [k]: e.target.value })} />
                  <button className="cfg-measure-del" onClick={() => {
                    const { [k]: _, ...rest } = p.queries; set("queries", rest);
                  }}>×</button>
                </div>
              ))}
              <button className="cfg-measure-add" onClick={() => {
                const name = "q" + (Object.keys(p.queries || {}).length + 1);
                set("queries", { ...(p.queries || {}), [name]: "" });
              }}>
                <Icon name="plus" size={11} /> Add query
              </button>
            </div>
          </ConfigField>
        </>
      );
    case "TitleKeywordFilter":
    case "AbstractKeywordFilter":
    case "VenueKeywordFilter":
      return (
        <>
          <ConfigField label="Match">
            <select value={p.match} onChange={e => set("match", e.target.value)}>
              <option>substring</option><option>whole_word</option><option>starts_with</option>
            </select>
          </ConfigField>
          <ConfigField label="Formula" wide hint="Boolean over named keywords — & | !">
            <input value={p.formula} onChange={e => set("formula", e.target.value)} />
          </ConfigField>
          <ConfigField label="Keywords" full hint="name → literal string">
            <div className="cfg-kv-list">
              {Object.entries(p.keywords || {}).map(([k, v]) => (
                <div key={k} className="cfg-kv">
                  <input className="cfg-kv-k" value={k} readOnly />
                  <input className="cfg-kv-v" value={v} onChange={e => set("keywords", { ...p.keywords, [k]: e.target.value })} />
                  <button className="cfg-measure-del" onClick={() => {
                    const { [k]: _, ...rest } = p.keywords; set("keywords", rest);
                  }}>×</button>
                </div>
              ))}
              <button className="cfg-measure-add" onClick={() => {
                const name = "k" + (Object.keys(p.keywords || {}).length + 1);
                set("keywords", { ...(p.keywords || {}), [name]: "" });
              }}>
                <Icon name="plus" size={11} /> Add keyword
              </button>
            </div>
          </ConfigField>
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
  if (node.kind === "fwd" || node.kind === "bwd") {
    return (
      <>
        <ConfigField label="Depth k" hint="How many hops"><input type="number" value={c.depth} onChange={e => onPatchConfig({ depth: +e.target.value })} /></ConfigField>
        <ConfigField label="Max children" hint="Per parent paper"><input type="number" value={c.maxChildren} onChange={e => onPatchConfig({ maxChildren: +e.target.value })} /></ConfigField>
        <ConfigField label="Direction" wide>
          <input disabled value={node.kind === "fwd" ? "outgoing citations (citers)" : "references"} />
        </ConfigField>
      </>
    );
  }
  if (node.kind === "rerank") {
    return (
      <>
        <ConfigField label="λ (diversity)" hint="0 = relevance, 1 = diversity">
          <input type="number" step="0.1" min="0" max="1" value={c.lambda} onChange={e => onPatchConfig({ lambda: +e.target.value })} />
        </ConfigField>
        <ConfigField label="Target N"><input type="number" value={c.targetN} onChange={e => onPatchConfig({ targetN: +e.target.value })} /></ConfigField>
        <ConfigField label="Strategy" wide>
          <select defaultValue="MMR"><option>MMR</option><option>Cluster-based</option><option>Random baseline</option></select>
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
