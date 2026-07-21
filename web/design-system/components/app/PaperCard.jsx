import React from "react";

/** The unified paper/seed card used in every list (Build seeds, seed set,
    Run accepted/rejected, Explore). A FIXED 18px lead column keeps every
    title left-aligned whether or not a lead glyph is present. */
export function PaperCard({ title, meta, trailing, lead = "none", trailingKind = "cites", selected, onClick, style }) {
  const card = {
    padding: "10px 12px", borderRadius: "var(--cc-r-card)", cursor: onClick ? "pointer" : "default",
    display: "flex", flexDirection: "column", gap: 4, background: "var(--cc-card)",
    border: selected ? "1px solid var(--cc-accent)" : "1px solid var(--cc-rule)",
    boxShadow: selected ? "var(--cc-ring)" : "var(--cc-shadow-2)", ...style,
  };
  const trail = trailingKind === "score"
    ? { fontFamily: "var(--cc-font-mono)", fontSize: 11, fontWeight: 600, color: "var(--cc-accent)", background: "var(--cc-accent-wash)", borderRadius: "var(--cc-r-pill-sm)", padding: "2px 7px", whiteSpace: "nowrap" }
    : { fontFamily: "var(--cc-font-mono)", fontSize: 11, fontWeight: 500, color: "var(--cc-ink-2)", whiteSpace: "nowrap", fontVariantNumeric: "tabular-nums" };
  return (
    <div onClick={onClick} style={card}>
      <div style={{ display: "grid", gridTemplateColumns: "18px minmax(0,1fr) auto", gap: 9, alignItems: "start" }}>
        <span style={{ width: 18, height: 18, display: "inline-flex", alignItems: "center", justifyContent: "center", marginTop: 1 }}>
          {lead === "seed" && <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--cc-accent)" }} />}
          {lead === "star" && <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--cc-warning)" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>}
          {lead === "reject" && <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--cc-danger)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m18 6-12 12M6 6l12 12"/></svg>}
        </span>
        <div style={{ minWidth: 0, fontSize: 13, fontWeight: 500, color: "var(--cc-ink-1)", lineHeight: 1.35, letterSpacing: "-0.005em", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{title}</div>
        <span style={trail}>{trailing}</span>
      </div>
      <div style={{ paddingLeft: 27, fontFamily: "var(--cc-font-mono)", fontSize: 11, color: "var(--cc-ink-3)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{meta}</div>
    </div>
  );
}
