// Ambient types for the VERBATIM demo module `import-resolver.js`.
//
// IIFE, not an ES module: importing it for side effect assigns
// `window.KLImport`. Despite the name it is not a build tool — "resolver"
// means "resolve an imported bibliography entry against Semantic Scholar".
// It is the shared back end for Home's import wizard, Runs' Add papers and
// the Build sidebar's Import mode, and it owns the triage order
// (Needs a decision -> Couldn't match -> Matched -> Already in the corpus)
// plus the deterministic 34-entry refs.bib fixture.
//
// Copied from web/design-reference/embedded-sources/import-resolver.js;
// sha256 recorded in PROVENANCE.json.

export {};

declare global {
  interface KLImport {
    /** Markup for one file row in the parse list. */
    fileRowHtml: (...args: unknown[]) => string;
    /** Triage sections, in the fixed product order. */
    groups: (...args: unknown[]) => unknown;
    /** Markup for the "N matches" candidate popover. */
    candPopHtml: (...args: unknown[]) => string;
    [key: string]: unknown;
  }
  // eslint-disable-next-line no-var
  var KLImport: KLImport | undefined;
}
