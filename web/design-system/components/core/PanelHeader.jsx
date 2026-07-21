import React from "react";

/** Panel header — uppercase mono eyebrow on the left, mono count/actions on the right. */
export function PanelHeader({ label, count, actions, style }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "11px 14px 10px", background: "var(--cc-chrome)", borderBottom: "1px solid var(--cc-rule-strong)", ...style }}>
      <span style={{ fontFamily: "var(--cc-font-sans)", fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--cc-ink-2)" }}>{label}</span>
      {actions ? actions : (count != null && (
        <span style={{ fontFamily: "var(--cc-font-mono)", fontSize: 10, color: "var(--cc-ink-3)" }}>{count}</span>
      ))}
    </div>
  );
}
