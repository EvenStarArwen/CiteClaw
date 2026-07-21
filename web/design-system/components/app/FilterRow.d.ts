import * as React from "react";
export interface FilterRowProps {
  /** Short uppercase kind pill, e.g. "YEAR", "LLM", "KEYWORD". */
  kind: string;
  /** One-line summary of the filter's config. */
  summary: React.ReactNode;
  selected?: boolean;
  onClick?: () => void;
  style?: React.CSSProperties;
}
export declare function FilterRow(props: FilterRowProps): JSX.Element;
