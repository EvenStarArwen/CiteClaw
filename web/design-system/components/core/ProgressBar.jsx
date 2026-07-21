import React from "react";

/** Determinate (value 0–100) or indeterminate (value null) green progress bar. */
export function ProgressBar({ value = null, height = 6, style }) {
  const indet = value == null;
  return (
    <div style={{ height, borderRadius: 999, background: "var(--cc-hover-deep)", overflow: "hidden", position: "relative", ...style }}>
      {indet ? (
        <div style={{ position: "absolute", top: 0, bottom: 0, width: "40%", borderRadius: 999, background: "var(--cc-accent)", animation: "ccbIndet 1.5s ease-in-out infinite" }} />
      ) : (
        <div style={{ height: "100%", width: Math.max(0, Math.min(100, value)) + "%", borderRadius: 999, background: "var(--cc-accent)" }} />
      )}
      <style>{"@keyframes ccbIndet{0%{transform:translateX(-100%)}100%{transform:translateX(280%)}}"}</style>
    </div>
  );
}
