import * as React from "react";
export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  /**
   * Visual weight. `primary` is the green action button — use it once per view
   * for the single most important action (Run pipeline, Explore subtree).
   * @startingPoint section="Core" subtitle="Green action + secondary + ghost" viewport="360x120"
   */
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
  icon?: React.ReactNode;
}
export declare function Button(props: ButtonProps): JSX.Element;
