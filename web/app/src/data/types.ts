/**
 * data/types.ts — the DataSource boundary.
 *
 * This is the ONE seam between screens and where the bytes come from. Screens
 * import `useDataSource()` and call these methods; they never import from
 * `../design-data` and never call `fetch`. Swapping the design-data provider
 * for a CiteClaw API adapter must therefore require zero edits inside screen
 * code — that is the whole point of this file, and the reason to resist adding
 * provider-specific fields to it.
 *
 * Design rules for anyone extending this:
 *
 *  1. Everything is async. The design-data provider resolves immediately; a
 *     real adapter will not. If a screen is written against a synchronous
 *     getter it will have to be rewritten when the API lands.
 *  2. The vocabulary is the DEMO's vocabulary (`ti`, `au`, `lc`, PAPERS-index
 *     edges …), not the backend's. The demo is the source of visual truth, so
 *     the adapter's job is to translate CiteClaw's shapes INTO these. Renaming
 *     fields here to match the backend would push the translation into every
 *     screen.
 *  3. Methods return the whole collection, as the demo's data modules do. The
 *     demo has no pagination anywhere; inventing it here would be designing UI
 *     that does not exist (see web/wiring/escalations.md before adding any).
 */

import type {
  DesignTopic,
  DesignPaper,
  DesignEdge,
  DesignCommunity,
  DesignCitationGroup,
} from '../design-data/index';

export type {
  DesignTopic,
  DesignPaper,
  DesignEdge,
  DesignCommunity,
  DesignCitationGroup,
};

/** The corpus: topics (semantic space) plus every paper in the run. */
export interface CorpusSnapshot {
  topics: DesignTopic[];
  papers: DesignPaper[];
  /** One-line description per topic id. May be empty for a real run. */
  topicDescriptions: Record<number, string>;
}

/** The citation graph and its Leiden communities (network space). */
export interface NetworkSnapshot {
  /** `[sourceIndex, targetIndex, weight]` — indices into `CorpusSnapshot.papers`. */
  edges: DesignEdge[];
  communities: DesignCommunity[];
  /** CSS colour expressions per community, index-aligned with `communities`. */
  communityColors: string[];
  peripheralColor: string;
  /** Communities of this size or smaller collapse into the "Peripheral" card. */
  peripheralMax: number;
  peripheralId: number;
}

/** "In this corpus" — the citing passages for one subject paper. */
export interface CitationContextSnapshot {
  /** Index into `CorpusSnapshot.papers` of the paper being cited. */
  subject: number;
  groups: DesignCitationGroup[];
  /** HTML paragraphs of the synthesized summary. */
  summary: string[];
  summaryFoot: string;
}

/**
 * The run/pipeline surface. Deliberately `unknown`-shaped for now: the demo's
 * run mock is a large ad-hoc object and the engine's event protocol is still
 * being changed (decisions ledger Q4/Q17). The Runs rewrite agent should
 * narrow these types here, in one place, rather than casting at call sites.
 */
export interface RunSnapshot {
  pipeline: unknown;
  filters: unknown;
  /** The demo's RUN-37. */
  currentRun: unknown;
  recommendations: unknown;
  library: unknown;
  nextRunNumber: number;
}

/** The literature-review draft (Explore). Markdown with demo citekeys. */
export interface ReviewDraft {
  markdown: string;
}

/**
 * A provider of everything the screens render.
 *
 * Implementations must be safe to call repeatedly; caching is the provider's
 * business, not the screen's.
 */
export interface DataSource {
  /** Stable identifier, e.g. `'design-data'` or `'citeclaw-api'`. */
  readonly id: string;
  /** Human-readable, for the parity playground's provider badge. */
  readonly label: string;

  getCorpus(): Promise<CorpusSnapshot>;
  getNetwork(): Promise<NetworkSnapshot>;
  /**
   * @param paperIndex index into `CorpusSnapshot.papers`. The design-data
   * provider only has contexts for one paper and rejects anything else, which
   * is exactly the kind of gap the parity harness should surface loudly rather
   * than render as an empty panel.
   */
  getCitationContext(paperIndex: number): Promise<CitationContextSnapshot>;
  getRun(): Promise<RunSnapshot>;
  getReviewDraft(): Promise<ReviewDraft>;
}

/**
 * Thrown when a provider genuinely cannot answer — not wired yet, no such
 * record, backend unreachable. Screens must render an explicit state for this;
 * swallowing it produces the silent failures called out in
 * web/wiring/missing-states.md.
 */
export class DataSourceError extends Error {
  readonly kind: 'not-wired' | 'not-found' | 'unreachable';
  readonly sourceId: string;

  constructor(
    kind: 'not-wired' | 'not-found' | 'unreachable',
    sourceId: string,
    message: string,
  ) {
    super(message);
    this.name = 'DataSourceError';
    this.kind = kind;
    this.sourceId = sourceId;
  }
}
