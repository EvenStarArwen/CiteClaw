/**
 * design-data/index.ts — importable surface over the VERBATIM demo data modules.
 *
 * The `.js` files in this directory are byte-identical copies of the modules
 * the KnowledgeLab iPad Demo loads (see PROVENANCE.json for sha256s). They are
 * never edited. This file is the only thing that knows how to load them:
 *
 *   - six are plain ES modules (`export const …`) and are dynamically imported
 *     so the shell stays small — topic-data.js alone is ~640 kB;
 *   - two (run-mock.js, paper-row.js) are IIFEs that assign a global and
 *     dispatch a `…-ready` event. `loadRunMock()` / `loadPaperRow()` hide that.
 *
 * Screens should NOT import from here directly — they consume the DataSource
 * in ../data. This module is the design-data provider's private back end.
 */

import type { DesignTopic, DesignPaper } from './topic-data.js';
import type { DesignEdge } from './graph-data.js';
import type { DesignCommunity } from './community-data.js';
import type { DesignCitationGroup } from './citation-context.js';

export type { DesignTopic, DesignPaper } from './topic-data.js';
export type { DesignEdge } from './graph-data.js';
export type { DesignCommunity } from './community-data.js';
export type { DesignCitationGroup, DesignCitationItem } from './citation-context.js';

export const loadTopicData = (): Promise<{ TOPICS: DesignTopic[]; PAPERS: DesignPaper[] }> =>
  import('./topic-data.js');

export const loadGraphData = (): Promise<{ EDGES: DesignEdge[] }> => import('./graph-data.js');

export const loadCommunityData = (): Promise<{
  LEIDEN: DesignCommunity[];
  COMMUNITY_COLORS: string[];
  PERIPHERAL_COLOR: string;
  PERIPHERAL_MAX: number;
  PERIPHERAL_ID: number;
  communityColor: (id: number) => string;
}> => import('./community-data.js');

export const loadCitationContext = (): Promise<{
  SUBJECT: number;
  GROUPS: DesignCitationGroup[];
  SUMMARY: string[];
  SUMMARY_FOOT: string;
}> => import('./citation-context.js');

export const loadReviewDraft = (): Promise<{ REVIEW_MD: string }> => import('./review-draft.js');

export const loadTopicDesc = (): Promise<{ TOPIC_DESC: Record<number, string> }> =>
  import('./topic-desc.js');

/**
 * run-mock.js is an IIFE. Importing it runs it, which sets `window.KLRunMock`
 * synchronously before the module promise resolves, so no event listener is
 * needed — but we assert rather than assume, because a silent `undefined` here
 * would surface much later as an unexplained blank panel.
 */
export async function loadRunMock(): Promise<KLRunMock> {
  await import('./run-mock.js');
  const mock = globalThis.KLRunMock;
  if (!mock) {
    throw new Error(
      'design-data: run-mock.js loaded but did not set globalThis.KLRunMock — the verbatim copy may be truncated.',
    );
  }
  return mock;
}

/**
 * Same contract as loadRunMock, for the shared import/triage module. Screens
 * that render the import flow (Home wizard step 2, Runs "Add papers", Build's
 * sidebar Import mode) all go through this one module — it is genuinely shared
 * in the demo, not copy-pasted.
 */
export async function loadImportResolver(): Promise<KLImport> {
  await import('./import-resolver.js');
  const imp = globalThis.KLImport;
  if (!imp) {
    throw new Error(
      'design-data: import-resolver.js loaded but did not set globalThis.KLImport — the verbatim copy may be truncated.',
    );
  }
  return imp;
}

/** Same contract as loadRunMock, for the shared paper-card markup module. */
export async function loadPaperRow(): Promise<KLPaperRow> {
  await import('./paper-row.js');
  const pr = globalThis.KLPaperRow;
  if (!pr) {
    throw new Error(
      'design-data: paper-row.js loaded but did not set globalThis.KLPaperRow — the verbatim copy may be truncated.',
    );
  }
  return pr;
}
