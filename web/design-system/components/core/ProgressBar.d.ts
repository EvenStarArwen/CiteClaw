import * as React from "react";
export interface ProgressBarProps {
  /** 0–100 for determinate; null/undefined for an indeterminate "working" bar. */
  value?: number | null;
  height?: number;
  style?: React.CSSProperties;
}
export declare function ProgressBar(props: ProgressBarProps): JSX.Element;
