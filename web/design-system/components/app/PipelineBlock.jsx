import React from "react";

const ICONS = {
  seed:   <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/>,
  fwd:    <g><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></g>,
  bwd:    <g><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></g>,
  rerank: <g><line x1="21" y1="4" x2="14" y2="4"/><line x1="10" y1="4" x2="3" y2="4"/><line x1="21" y1="12" x2="12" y2="12"/><line x1="8" y1="12" x2="3" y2="12"/><line x1="21" y1="20" x2="16" y2="20"/><line x1="12" y1="20" x2="3" y2="20"/><line x1="14" y1="2" x2="14" y2="6"/><line x1="8" y1="10" x2="8" y2="14"/><line x1="16" y1="18" x2="16" y2="22"/></g>,
  rsc:    <g><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></g>,
};
const KIND_LABEL = { seed: "Source", fwd: "Expander", bwd: "Expander", rerank: "Reranker", rsc: "Reranker" };

/** A pipeline step block ("specimen" style). Selected = green border + ring. */
export function PipelineBlock({ kind = "fwd", num, name, id, filters = 0, selected, sync, onClick, style }) {
  return (
    <div onClick={onClick} style={{
      width: 210, boxSizing: "border-box", position: "relative", cursor: "pointer",
      background: "var(--cc-card)", borderRadius: "var(--cc-r-card)", padding: "13px 14px",
      border: selected ? "1px solid var(--cc-accent)" : "1px solid var(--cc-rule)",
      boxShadow: selected ? "var(--cc-ring)" : "var(--cc-shadow-1)", ...style,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", fontFamily: "var(--cc-font-mono)", fontSize: 10.5, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.11em", color: "var(--cc-ink-3)", marginBottom: 7 }}>
        <span>{KIND_LABEL[kind] || "Step"}</span><span style={{ color: "var(--cc-ink-2)" }}>№ {num}</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 9 }}>
        <span style={{ flex: "0 0 auto", color: "var(--cc-ink-2)", display: "inline-flex" }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">{ICONS[kind]}</svg>
        </span>
        <span style={{ fontSize: 14, fontWeight: 600, color: "var(--cc-ink-1)", letterSpacing: "-0.01em", lineHeight: 1.25 }}>{name}</span>
      </div>
      <div style={{ height: 1, background: "var(--cc-rule)", marginBottom: 8 }} />
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6, fontFamily: "var(--cc-font-mono)", fontSize: 11.5, color: "var(--cc-ink-3)" }}>
        <span style={{ fontWeight: 600, color: "var(--cc-ink-2)" }}>{id}</span>
        {filters > 0 && <span style={{ fontSize: 11, padding: "2px 8px", background: "var(--cc-hover-soft)", border: "1px solid var(--cc-rule)", borderRadius: "var(--cc-r-chip)", color: "var(--cc-ink-2)", whiteSpace: "nowrap" }}>{filters} filters</span>}
      </div>
      {sync && <span style={{ position: "absolute", top: -8, right: -6, fontFamily: "var(--cc-font-mono)", fontSize: 8.5, fontWeight: 600, color: "var(--cc-ink-3)", background: "var(--cc-card)", border: "1px solid var(--cc-rule)", borderRadius: "var(--cc-r-chip)", padding: "1px 6px" }}>{sync}</span>}
    </div>
  );
}
