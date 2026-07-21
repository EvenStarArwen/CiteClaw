import * as React from "react";
export interface SegmentOption { value: string; label: string; }
export interface SegmentedToggleProps {
  options: SegmentOption[];
  value: string;
  onChange?: (value: string) => void;
  size?: "sm" | "md";
}
export declare function SegmentedToggle(props: SegmentedToggleProps): JSX.Element;
