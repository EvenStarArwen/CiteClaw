// Ambient types for the VERBATIM demo module `run-mock.js`.
//
// This module is an IIFE, not an ES module: importing it for side effect
// assigns `window.KLRunMock` and dispatches a `kl-run-mock-ready` event on
// `document`. Read the surface through `globalThis.KLRunMock` (see
// design-data/index.ts, which does the waiting for you).

export {};

declare global {
  interface KLRunMock {
    PIPELINE: unknown;
    FILTERS: unknown;
    /** The RUN-37 demo run. */
    RUN37: unknown;
    RECO: unknown;
    RUN_LIBRARY: unknown;
    NEXT_RUN_NO: number;
    REPLAY: unknown;
    makeReplay: (...args: unknown[]) => unknown;
    buildTimeline: (...args: unknown[]) => unknown;
  }
  // eslint-disable-next-line no-var
  var KLRunMock: KLRunMock | undefined;
}
