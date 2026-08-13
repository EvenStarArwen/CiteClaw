// Ambient types for the VERBATIM demo module `citation-context.js`.

/** One citing passage. */
export interface DesignCitationItem {
  [key: string]: unknown;
}

/** Citing passages grouped by citing paper. */
export interface DesignCitationGroup {
  /** PAPERS index of the citing paper. */
  p: number;
  items: DesignCitationItem[];
}

/** PAPERS index of the paper these contexts are about. */
export const SUBJECT: number;
export const GROUPS: DesignCitationGroup[];
/** HTML paragraphs of the synthesized summary. */
export const SUMMARY: string[];
export const SUMMARY_FOOT: string;
