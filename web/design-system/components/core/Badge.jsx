import React from "react";

/** Small pill for status chips, count pills, filter kinds. */
export function Badge({ tone = "neutral", children, style, ...rest }) {
  const tones = {
    neutral: { color: "var(--cc-ink-2)", background: "var(--cc-hover-soft)", border: "1px solid var(--cc-rule)" },
    accent:  { color: "var(--cc-accent)", background: "var(--cc-accent-wash)", border: "1px solid color-mix(in srgb, var(--cc-accent) 25%, transparent)" },
    warning: { color: "#6f6238", background: "color-mix(in srgb, var(--cc-warning) 16%, var(--cc-card))", border: "1px solid color-mix(in srgb, var(--cc-warning) 30%, transparent)" },
    danger:  { color: "var(--cc-danger)", background: "color-mix(in srgb, var(--cc-danger) 12%, var(--cc-card))", border: "1px solid color-mix(in srgb, var(--cc-danger) 28%, transparent)" },
  };
  return (
    <span {...rest} style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      fontFamily: "var(--cc-font-mono)", fontSize: 11, fontWeight: 600,
      padding: "2px 8px", borderRadius: "var(--cc-r-chip)", whiteSpace: "nowrap",
      ...tones[tone], ...style,
    }}>{children}</span>
  );
}
