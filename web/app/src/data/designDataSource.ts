/**
 * data/designDataSource.ts — the first DataSource provider.
 *
 * Serves the verbatim demo modules in src/design-data. This is what the pilot
 * screens render from, so that "does the rewrite look like the demo?" is a
 * question about markup and CSS only, never about the data.
 *
 * The CiteClaw API adapter will be a sibling file implementing the same
 * interface. Nothing in here should leak into screen code.
 */

import {
  loadTopicData,
  loadTopicDesc,
  loadGraphData,
  loadCommunityData,
  loadCitationContext,
  loadReviewDraft,
  loadRunMock,
} from '../design-data/index';
import type {
  DataSource,
  CorpusSnapshot,
  NetworkSnapshot,
  CitationContextSnapshot,
  RunSnapshot,
  ReviewDraft,
} from './types';
import { DataSourceError } from './types';

const ID = 'design-data';

/** Memoise per snapshot so repeated mounts do not re-parse ~700 kB of data. */
function once<T>(fn: () => Promise<T>): () => Promise<T> {
  let p: Promise<T> | null = null;
  return () => (p ??= fn());
}

const corpus = once<CorpusSnapshot>(async () => {
  const [{ TOPICS, PAPERS }, { TOPIC_DESC }] = await Promise.all([
    loadTopicData(),
    loadTopicDesc(),
  ]);
  return { topics: TOPICS, papers: PAPERS, topicDescriptions: TOPIC_DESC };
});

const network = once<NetworkSnapshot>(async () => {
  const [{ EDGES }, comm] = await Promise.all([loadGraphData(), loadCommunityData()]);
  return {
    edges: EDGES,
    communities: comm.LEIDEN,
    communityColors: comm.COMMUNITY_COLORS,
    peripheralColor: comm.PERIPHERAL_COLOR,
    peripheralMax: comm.PERIPHERAL_MAX,
    peripheralId: comm.PERIPHERAL_ID,
  };
});

const citationContext = once<CitationContextSnapshot>(async () => {
  const m = await loadCitationContext();
  return {
    subject: m.SUBJECT,
    groups: m.GROUPS,
    summary: m.SUMMARY,
    summaryFoot: m.SUMMARY_FOOT,
  };
});

const run = once<RunSnapshot>(async () => {
  const m = await loadRunMock();
  return {
    pipeline: m.PIPELINE,
    filters: m.FILTERS,
    currentRun: m.RUN37,
    recommendations: m.RECO,
    library: m.RUN_LIBRARY,
    nextRunNumber: m.NEXT_RUN_NO,
  };
});

const reviewDraft = once<ReviewDraft>(async () => {
  const { REVIEW_MD } = await loadReviewDraft();
  return { markdown: REVIEW_MD };
});

export const designDataSource: DataSource = {
  id: ID,
  label: 'Design data (iPad Demo, verbatim)',

  getCorpus: corpus,
  getNetwork: network,
  getRun: run,
  getReviewDraft: reviewDraft,

  async getCitationContext(paperIndex: number) {
    const ctx = await citationContext();
    if (paperIndex !== ctx.subject) {
      // The demo shipped citing passages for exactly one paper. Say so instead
      // of returning an empty list that a screen would render as "no citations".
      throw new DataSourceError(
        'not-found',
        ID,
        `The design data only carries citation contexts for paper index ${ctx.subject}; ${paperIndex} was requested.`,
      );
    }
    return ctx;
  },
};
