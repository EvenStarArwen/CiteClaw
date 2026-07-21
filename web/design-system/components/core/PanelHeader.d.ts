import * as React from "react";
export interface PanelHeaderProps {
  /** Uppercase eyebrow label, e.g. "SEEDS", "CONFIGURE STEP". */
  label: string;
  /** Mono count shown on the right (e.g. "6 results", "4 / 9"). */
  count?: React.ReactNode;
  /** Optional actions node replacing the count. */
  actions?: React.ReactNode;
  style?: React.CSSProperties;
}
export declare function PanelHeader(props: PanelHeaderProps): JSX.Element;
