/**
 * Types for the verbatim transplant in build-logic.js.
 *
 * Deliberately thin. The class has 164 methods and its internals are the
 * demo's, not ours; typing them would invite "improvements" to a file that
 * must stay byte-identical to the source. The only contract the app has with
 * it is: construct it with a ref and the props, then call componentDidMount.
 */

import type { RefObject } from 'react';

/** The props the demo shell passes the "Paper Card" screen. */
export interface BuildLogicProps {
  cardLayout?: string;
  pipelineStyle?: string;
  colorScheme?: string;
  logoStyle?: string;
  layout?: string;
  pageState?: string;
  cardEmphasis?: string;
  overscrollBounce?: boolean;
  /** Not passed by the shell; the logic falls back to 'light'. */
  theme?: string;
}

export declare class BuildLogic {
  constructor(rootRef: RefObject<HTMLElement>, props: BuildLogicProps);
  props: BuildLogicProps;
  rootRef: RefObject<HTMLElement>;
  /** Runs the demo's whole mount sequence against rootRef.current. */
  componentDidMount(): void;
  componentDidUpdate(prev: BuildLogicProps): void;
}
