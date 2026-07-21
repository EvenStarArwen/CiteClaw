import React from "react";

/** Primary action / secondary / ghost button. Green fill = primary. */
export function Button({ variant = "primary", size = "md", icon, children, disabled, style, ...rest }) {
  const sizes = {
    sm: { padding: "5px 10px", fontSize: 12 },
    md: { padding: "6px 12px", fontSize: 13 },
    lg: { padding: "8px 14px", fontSize: 14 },
  };
  const variants = {
    primary:   { background: "var(--cc-accent)", color: "var(--cc-on-accent)", border: "1px solid var(--cc-accent)", boxShadow: "var(--cc-shadow-1)" },
    secondary: { background: "var(--cc-card)", color: "var(--cc-ink-1)", border: "1px solid var(--cc-rule-strong)" },
    ghost:     { background: "transparent", color: "var(--cc-ink-2)", border: "1px solid transparent" },
  };
  return (
    <button
      disabled={disabled}
      {...rest}
      style={{
        display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 6,
        fontFamily: "var(--cc-font-sans)", fontWeight: variant === "primary" ? 600 : 500,
        lineHeight: 1, borderRadius: "var(--cc-r-ctrl)", cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.55 : 1, transition: "background var(--cc-dur-fast) var(--cc-ease-out)",
        ...sizes[size], ...variants[variant], ...style,
      }}
    >
      {icon}{children}
    </button>
  );
}
