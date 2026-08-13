// Ambient types for the VERBATIM demo module `paper-row.js`.
//
// IIFE, not an ES module: importing it for side effect assigns
// `window.KLPaperRow` and dispatches `kl-paper-row-ready` on `document`. It is
// the ONE paper-card markup contract for KnowledgeLab — screens must render
// paper cards through it rather than re-implementing the markup.

export {};

declare global {
  interface KLPaperRow {
    PR_ICONS: Record<string, string>;
    /** Inner HTML for a full paper card. */
    prCardInner: (...args: unknown[]) => string;
    /** Inner HTML for the compact row. */
    prMiniInner: (...args: unknown[]) => string;
    /** Inner HTML for the hover tooltip. */
    prTipInner: (...args: unknown[]) => string;
    prEtAl: (...args: unknown[]) => string;
    prKfmt: (...args: unknown[]) => string;
    prPdf: (...args: unknown[]) => string;
    prAuthors: (...args: unknown[]) => string;
  }
  // eslint-disable-next-line no-var
  var KLPaperRow: KLPaperRow | undefined;
}
