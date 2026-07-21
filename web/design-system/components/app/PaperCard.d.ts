import * as React from "react";
export interface PaperCardProps {
  title: string;
  /** Mono meta line, e.g. "Boiko et al. · 2023 · Nature". */
  meta: React.ReactNode;
  /** Right-aligned value — cites string or a score. */
  trailing: React.ReactNode;
  /**
   * Lead-column glyph. The column is always 18px wide so titles align.
   * @startingPoint section="App" subtitle="Unified paper / seed list card" viewport="360x92"
   */
  lead?: "none" | "seed" | "star" | "reject";
  trailingKind?: "cites" | "score";
  selected?: boolean;
  onClick?: () => void;
  style?: React.CSSProperties;
}
export declare function PaperCard(props: PaperCardProps): JSX.Element;
