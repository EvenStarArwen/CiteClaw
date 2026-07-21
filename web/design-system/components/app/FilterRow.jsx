import React from "react";

/** A leaf filter row inside a step's filter pipeline. Kind pill + summary + chevron. */
export function FilterRow({ kind, summary, selected, onClick, style }) {
  return (
    <div onClick={onClick} style={{
      display: "grid", gridTemplateColumns: "auto 1fr auto", alignItems: "center", gap: 10,
      padding: "8px 10px", background: "var(--cc-card)", borderRadius: "var(--cc-r-card)", cursor: "pointer",
      border: selected ? "1px solid var(--cc-accent)" : "1px solid var(--cc-rule-strong)",
      boxShadow: selected ? "var(--cc-ring)" : "none", ...style,
    }}>
      <span style={{ fontFamily: "var(--cc-font-mono)", fontSize: 11, fontWeight: 600, color: "var(--cc-ink-2)", textTransform: "uppercase", letterSpacing: "0.05em", padding: "2px 7px", background: "var(--cc-hover-soft)", borderRadius: "var(--cc-r-pill-sm)", whiteSpace: "nowrap" }}>{kind}</span>
      <span style={{ fontFamily: "var(--cc-font-mono)", fontSize: 11.5, color: "var(--cc-ink-1)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{summary}</span>
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--cc-ink-faint)" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
    </div>
  );
}
