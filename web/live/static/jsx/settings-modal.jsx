/* eslint-disable */
// Settings — API keys + model picker with prices.
// A small addition to the design (in its own cream style), opened from the
// top-bar gear. Only gemini-3.1-flash-lite actually runs in this version;
// other selections are allowed but the run reports "not supported yet".

function SettingsModal({ open, onClose }) {
  const settings = useLive("settings");
  const [kGemini, setKGemini] = React.useState("");
  const [kOpenai, setKOpenai] = React.useState("");
  const [kS2, setKS2] = React.useState("");
  const [model, setModel] = React.useState(settings.model);
  const [effort, setEffort] = React.useState(settings.effort);
  const [maxPapers, setMaxPapers] = React.useState(settings.maxPapers || 200);
  const [busy, setBusy] = React.useState(false);
  const [msg, setMsg] = React.useState(null);

  React.useEffect(() => {
    if (open) { refreshSettings(); setMsg(null); setKGemini(""); setKOpenai(""); setKS2(""); }
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
      setMsg({ ok: true, text: "Saved." });
      setKGemini(""); setKOpenai(""); setKS2("");
    } catch (e) {
      setMsg({ ok: false, text: e.message || "save failed" });
    }
    setBusy(false);
  };

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
    input: { width: "100%", boxSizing: "border-box", padding: "8px 10px", fontSize: 13,
      border: "1px solid var(--cc-btn-border)", borderRadius: 4, background: "var(--cc-paper)", color: "var(--cc-ink-1)" },
    set: { color: "var(--cc-positive)", fontSize: 11, marginLeft: 8 },
    note: { fontSize: 11, color: "var(--cc-ink-3)", marginTop: 6, lineHeight: 1.5 },
    foot: { display: "flex", justifyContent: "flex-end", gap: 8, padding: "14px 18px",
      borderTop: "1px solid var(--cc-rule)" },
    price: { fontFamily: "var(--cc-font-mono)", color: "var(--cc-ink-2)" },
  };

  const keyRow = (labelText, present, val, setVal, ph) => (
    <div>
      <label style={T.label}>{labelText}{present && <span style={T.set}>✓ set</span>}</label>
      <input type="password" style={T.input} value={val} onChange={e => setVal(e.target.value)}
        placeholder={present ? "•••••••• (leave blank to keep)" : ph} autoComplete="off" />
    </div>
  );

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
          {keyRow("Gemini API key", keys.gemini_api_key, kGemini, setKGemini, "AIza…")}
          {keyRow("OpenAI API key", keys.openai_api_key, kOpenai, setKOpenai, "sk-…")}
          {keyRow("Semantic Scholar API key", keys.s2_api_key, kS2, setKS2, "optional — works without, slower")}

          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--cc-ink-2)", marginTop: 20 }}>Model</div>
          <label style={T.label}>Screening model</label>
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

          <label style={T.label}>Reasoning effort</label>
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

Object.assign(window, { SettingsModal });
