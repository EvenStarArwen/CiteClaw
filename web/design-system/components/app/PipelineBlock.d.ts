import * as React from "react";
export interface PipelineBlockProps {
  /** Determines the icon + category label. */
  kind?: "seed" | "fwd" | "bwd" | "rerank" | "rsc";
  /** Two-digit step number for the № caption, e.g. "02". */
  num: string;
  name: string;
  /** Local id, e.g. "FWD-02". */
  id: string;
  /** Filter count shown as a pill (0 = hidden). */
  filters?: number;
  selected?: boolean;
  /** Linked-copy badge text, e.g. "⇄ FWD-02". */
  sync?: string;
  onClick?: () => void;
  style?: React.CSSProperties;
}
export declare function PipelineBlock(props: PipelineBlockProps): JSX.Element;
