import * as React from "react";
export interface MetricStatProps {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  style?: React.CSSProperties;
}
export declare function MetricStat(props: MetricStatProps): JSX.Element;
