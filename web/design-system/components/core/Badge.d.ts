import * as React from "react";
export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** neutral = counts/ids · accent = live/active · warning · danger */
  tone?: "neutral" | "accent" | "warning" | "danger";
}
export declare function Badge(props: BadgeProps): JSX.Element;
