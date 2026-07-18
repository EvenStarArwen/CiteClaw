/* eslint-disable */
// Settings — API keys + model picker with prices.
// A small addition to the design (in its own cream style), opened from the
// top-bar gear. Only gemini-3.1-flash-lite actually runs in this version;
// other selections are allowed but the run reports "not supported yet".

function SettingsModal({ open, onClose, pipeline }) {
  const settings = useLive("settings");
  const [kGemini, setKGemini] = React.useState("");
  const [kOpenai, setKOpenai] = React.useState("");
  const [kS2, setKS2] = React.useState("");
  // Set keys render frozen ("● Set") — typing a replacement requires an
  // explicit Reset per key, so the old masked-input-with-cryptic-placeholder
  // state is gone. editing[field] = true → that key's input is unlocked.
  const [editing, setEditing] = React.useState({});
  const [model, setModel] = React.useState(settings.model);
  const [effort, setEffort] = React.useState(settings.effort);
  const [maxPapers, setMaxPapers] = React.useState(settings.maxPapers || 200);
  const [busy, setBusy] = React.useState(false);
  const [msg, setMsg] = React.useState(null);

  React.useEffect(() => {
    if (open) {
      refreshSettings();
      setMsg(null); setKGemini(""); setKOpenai(""); setKS2(""); setEditing({});
    }
  }, [open]);
  React.useEffect(() => {
    setModel(settings.model); setEffort(settings.effort);
    setMaxPapers(settings.maxPapers || 200);
  }, [settings.model, settings.effort, settings.maxPapers]);
  React.useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape" && open) onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  React.useEffect(() => { if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.75 } }); });

  if (!open) return null;

  const models = settings.models || [];
  const keys = settings.keys || {};
  const selMeta = models.find(m => m.id === model);
  const byProvider = { gemini: [], openai: [] };
  models.forEach(m => { (byProvider[m.provider] || (byProvider[m.provider] = [])).push(m); });

  const save = async () => {
    setBusy(true); setMsg(null);
    const patch = { model, reasoning_effort: effort, max_papers: maxPapers };
    if (kGemini.trim()) patch.gemini_api_key = kGemini.trim();
    if (kOpenai.trim()) patch.openai_api_key = kOpenai.trim();
    if (kS2.trim()) patch.s2_api_key = kS2.trim();
    try {
      await saveSettings(patch);
      setBusy(false);
      onClose();  // a successful save is done — don't make the user hunt for Cancel
    } catch (e) {
      setMsg({ ok: false, text: e.message || "save failed" });
      setBusy(false);
    }
  };

  // How many LLM screening filters the current pipeline contains — drives the
  // "no LLM key set" notice. Saving is never blocked by it. Plain computation
  // (no hook): this sits below the `if (!open)` early return, where a hook
  // would change the hook order between closed and open renders.
  let llmCount = 0;
  if (window.forEachLlmFilter) forEachLlmFilter(pipeline || [], () => llmCount++);

  const T = {
    backdrop: { position: "fixed", inset: 0, background: "rgba(30,39,53,0.30)", zIndex: 200,
      display: "flex", alignItems: "center", justifyContent: "center", padding: 20 },
    card: { width: "min(560px, 96vw)", maxHeight: "90vh", overflow: "auto", background: "var(--cc-panel)",
      border: "1px solid var(--cc-rule-strong)", borderRadius: 8, boxShadow: "0 12px 40px rgba(30,39,53,0.18)" },
    head: { display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "14px 18px", borderBottom: "1px solid var(--cc-rule)", background: "var(--cc-chrome)" },
    body: { padding: 18 },
    label: { display: "block", fontSize: 10, letterSpacing: "0.08em", textTransform: "uppercase",
      color: "var(--cc-ink-3)", marginBottom: 6, marginTop: 14 },
    keyLabel: { display: "flex", alignItems: "center", gap: 8, fontSize: 10, letterSpacing: "0.08em",
      textTransform: "uppercase", color: "var(--cc-ink-3)", marginBottom: 6, marginTop: 14 },
    input: { width: "100%", boxSizing: "border-box", padding: "8px 10px", fontSize: 13,
      border: "1px solid var(--cc-btn-border)", borderRadius: 4, background: "var(--cc-paper)", color: "var(--cc-ink-1)" },
    setChip: { display: "inline-flex", alignItems: "center", gap: 5,
      color: "var(--cc-positive)", fontWeight: 700, letterSpacing: "0.06em" },
    setDot: { width: 8, height: 8, borderRadius: "50%", background: "var(--cc-positive)", flex: "none" },
    frozen: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8,
      padding: "5px 6px 5px 11px", border: "1px solid var(--cc-rule)", borderRadius: 4,
      background: "var(--cc-hover-soft)" },
    frozenDots: { fontFamily: "var(--cc-font-mono)", fontSize: 11, letterSpacing: 3,
      color: "var(--cc-ink-3)", userSelect: "none" },
    note: { fontSize: 11, color: "var(--cc-ink-3)", marginTop: 6, lineHeight: 1.5 },
    foot: { display: "flex", justifyContent: "flex-end", gap: 8, padding: "14px 18px",
      borderTop: "1px solid var(--cc-rule)" },
    price: { fontFamily: "var(--cc-font-mono)", color: "var(--cc-ink-2)" },
  };

  // One key row. Not set → plain input. Set → frozen bar (green dot + masked
  // value) with a Reset button; Reset unlocks an empty input ("Keep current"
  // refreezes without changes). The stored key never round-trips to the page.
  const keyRow = (field, labelText, present, val, setVal, ph) => {
    const unlocked = !present || !!editing[field];
    return (
      <div>
        <label style={T.keyLabel}>
          <span>{labelText}</span>
          {present && <span style={T.setChip}><span style={T.setDot} /> Set</span>}
        </label>
        {unlocked ? (
          <div style={{ display: "flex", gap: 6 }}>
            <input type="password" style={{ ...T.input, flex: 1, minWidth: 0 }} value={val}
              onChange={e => setVal(e.target.value)} placeholder={ph}
              autoComplete="off" autoFocus={!!editing[field]} />
            {present && (
              <button className="btn btn-ghost" style={{ flex: "none" }}
                onClick={() => { setVal(""); setEditing(ed => ({ ...ed, [field]: false })); }}>
                Keep current
              </button>
            )}
          </div>
        ) : (
          <div style={T.frozen}>
            <span style={T.frozenDots}>••••••••••••••••</span>
            <button className="btn btn-ghost" style={{ flex: "none" }}
              title="Replace this key with a new value"
              onClick={() => setEditing(ed => ({ ...ed, [field]: true }))}>
              <Icon name="rotate-ccw" size={11} /> Reset
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <div style={T.backdrop} onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div style={T.card}>
        <div style={T.head}>
          <span style={{ fontWeight: 600, fontSize: 14 }}>Settings</span>
          <button className="ph-btn" onClick={onClose} title="Close"><Icon name="x" size={15} /></button>
        </div>
        <div style={T.body}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--cc-ink-2)" }}>API keys</div>
          <div style={T.note}>
            Stored privately on your computer (<span className="cc-mono">.env.local</span>, git-ignored).
            Never uploaded anywhere.
          </div>
          {keyRow("gemini_api_key", "Gemini API key", keys.gemini_api_key, kGemini, setKGemini, "AIza…")}
          {keyRow("openai_api_key", "OpenAI API key", keys.openai_api_key, kOpenai, setKOpenai, "sk-…")}
          {keyRow("s2_api_key", "Semantic Scholar API key", keys.s2_api_key, kS2, setKS2, "free from semanticscholar.org/product/api")}
          {!keys.s2_api_key && (
            <div className="cfg-note" style={{ marginTop: 10 }}>
              <Icon name="info" size={13} />
              <span>
                <b>Required for real runs.</b> Seed search and every expansion step call
                Semantic Scholar, and keyless access is so heavily rate-limited that
                runs stall or fail. Get a free key at semanticscholar.org/product/api —
                you can still save now and come back when you have it.
              </span>
            </div>
          )}
          {llmCount > 0 && !keys.gemini_api_key && !keys.openai_api_key && (
            <div className="cfg-note" style={{ marginTop: 10 }}>
              <span style={{ color: "var(--cc-warning, #b8860b)", display: "inline-flex", flex: "none", marginTop: 1 }}>
                <Icon name="alert-triangle" size={13} />
              </span>
              <span>
                Your pipeline screens with LLMs ({llmCount} LLM filter{llmCount > 1 ? "s" : ""})
                but no LLM API key is set — the run will refuse to start. Add a Gemini
                key (this version runs Gemini 3.1 Flash-Lite). Saving is fine without it.
              </span>
            </div>
          )}

          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--cc-ink-2)", marginTop: 20 }}>Model</div>
          <label style={T.label}>Default screening model</label>
          <select style={T.input} value={model} onChange={e => setModel(e.target.value)}>
            {["gemini", "openai"].map(prov => (
              <optgroup key={prov} label={prov === "gemini" ? "Google Gemini" : "OpenAI"}>
                {(byProvider[prov] || []).map(m => (
                  <option key={m.id} value={m.id}>
                    {m.label} — ${m.input}/${m.output} per 1M{m.supported ? "  ✓ supported" : ""}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>

          <label style={T.label}>Default reasoning effort</label>
          <select style={T.input} value={effort} onChange={e => setEffort(e.target.value)}>
            {["minimal", "low", "medium", "high"].map(e => <option key={e} value={e}>{e}</option>)}
          </select>

          {selMeta && (
            <div style={T.note}>
              <span style={T.price}>${selMeta.input} in / ${selMeta.output} out</span> per 1M tokens.
              {selMeta.supported
                ? " Supported in this version."
                : " ⚠ Not supported yet — this first release only runs Gemini 3.1 Flash-Lite. Selecting this will report an error on Run."}
            </div>
          )}
          <div style={T.note}>
            These are the defaults. Every LLM filter can pick its own model and
            effort in its config (Build tab → click the filter) — e.g. cheap
            triage on titles, a stronger model on abstracts.
          </div>

          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--cc-ink-2)", marginTop: 20 }}>Run parameters</div>
          <label style={T.label}>Max papers</label>
          <input type="number" min="1" max="5000" step="10" style={T.input}
            value={maxPapers} onChange={e => setMaxPapers(Math.max(1, +e.target.value || 1))} />
          <div style={T.note}>
            Upper bound on how many papers a run may collect before it stops.
          </div>

          {msg && (
            <div style={{ marginTop: 14, fontSize: 12, color: msg.ok ? "var(--cc-positive)" : "var(--cc-danger)" }}>
              {msg.text}
            </div>
          )}
        </div>
        <div style={T.foot}>
          <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={save} disabled={busy}>
            {busy ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}

// Small alert dialog shown when a run can't start (missing API key, no seeds,
// unsupported model …). Mirrors the Settings modal's cream chrome.
function RunErrorDialog({ error, onClose, onOpenSettings }) {
  React.useEffect(() => {
    if (!error) return;
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [error, onClose]);
  React.useEffect(() => { if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.75 } }); });

  if (!error) return null;
  const wantsSettings = /settings|api key|\bkey\b/i.test(error);

  const T = {
    backdrop: { position: "fixed", inset: 0, background: "rgba(30,39,53,0.30)", zIndex: 240,
      display: "flex", alignItems: "center", justifyContent: "center", padding: 20 },
    card: { width: "min(440px, 96vw)", background: "var(--cc-panel)", overflow: "hidden",
      border: "1px solid var(--cc-rule-strong)", borderRadius: 8, boxShadow: "0 12px 40px rgba(30,39,53,0.18)" },
    head: { display: "flex", alignItems: "center", gap: 9,
      padding: "14px 18px", borderBottom: "1px solid var(--cc-rule)", background: "var(--cc-chrome)" },
    body: { padding: 18, fontSize: 13, color: "var(--cc-ink-1)", lineHeight: 1.55 },
    foot: { display: "flex", justifyContent: "flex-end", gap: 8, padding: "12px 18px", borderTop: "1px solid var(--cc-rule)" },
  };
  return (
    <div style={T.backdrop} onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div style={T.card} role="alertdialog" aria-modal="true">
        <div style={T.head}>
          <span style={{ color: "var(--cc-danger)", display: "inline-flex" }}><Icon name="alert-triangle" size={16} /></span>
          <span style={{ fontWeight: 600, fontSize: 14 }}>Can’t start the run</span>
        </div>
        <div style={T.body}>{error}</div>
        <div style={T.foot}>
          <button className="btn btn-ghost" onClick={onClose}>Dismiss</button>
          {wantsSettings && (
            <button className="btn btn-primary" onClick={onOpenSettings}>
              <Icon name="settings" size={13} /> Open Settings
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { SettingsModal, RunErrorDialog });
