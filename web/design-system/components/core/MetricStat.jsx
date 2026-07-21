import React from "react";

/** Dashboard metric — small uppercase label, large tabular value, mono sub. */
export function MetricStat({ label, value, sub, style }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, padding: "10px 14px", ...style }}>
      <span style={{ fontFamily: "var(--cc-font-sans)", fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--cc-ink-3)" }}>{label}</span>
      <span style={{ fontFamily: "var(--cc-font-sans)", fontSize: 19, fontWeight: 600, letterSpacing: "-0.02em", color: "var(--cc-ink-1)", fontVariantNumeric: "tabular-nums" }}>{value}</span>
      {sub && <span style={{ fontFamily: "var(--cc-font-mono)", fontSize: 9, color: "var(--cc-ink-faint)" }}>{sub}</span>}
    </div>
  );
}
