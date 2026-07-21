import React from "react";

/** Segmented control — the Build/Run/Explore switch and dashboard tabs.
    Active segment is a raised card with green text. */
export function SegmentedToggle({ options = [], value, onChange, size = "md" }) {
  const pad = size === "sm" ? "4px 11px" : "5px 15px";
  const fs = size === "sm" ? 11 : 13;
  return (
    <div style={{ display: "inline-flex", padding: 3, gap: 2, background: "var(--cc-hover-deep)", borderRadius: "var(--cc-r-toggle)" }}>
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button key={o.value} onClick={() => onChange && onChange(o.value)}
            style={{
              padding: pad, fontFamily: "var(--cc-font-sans)", fontSize: fs,
              fontWeight: on ? 600 : 500, color: on ? "var(--cc-ink-accent)" : "var(--cc-ink-2)",
              background: on ? "var(--cc-card)" : "transparent", border: "none",
              borderRadius: "var(--cc-r-toggle-in)", cursor: "pointer",
              boxShadow: on ? "var(--cc-shadow-1)" : "none",
            }}>{o.label}</button>
        );
      })}
    </div>
  );
}
