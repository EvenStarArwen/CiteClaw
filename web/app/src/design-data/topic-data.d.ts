// Ambient types for the VERBATIM demo module `topic-data.js`. The .js file is a
// byte-identical copy of the design workspace file and must never be edited;
// this declaration file describes it from the outside.

/** A topic-model cluster in semantic space. `id: -1` is "Unclustered Noise". */
export interface DesignTopic {
  id: number;
  name: string;
  /** Keywords, most representative first. */
  kw: string[];
  /** Paper count in this topic. */
  n: number;
}

/** One corpus paper. Field names are the demo's own short keys. */
export interface DesignPaper {
  /** Semantic Scholar paper id (sha1-looking hex). */
  id: string;
  /** Title. */
  ti: string;
  /** Authors, " · " separated. */
  au: string;
  /** Abstract. */
  ab: string;
  /** Venue. */
  ve: string;
  /** Year. */
  yr: number;
  /** Citation count. */
  ci: number;
  /** Topic id (index into TOPICS by `id`). */
  tp: number;
  /** Semantic-space x coordinate. */
  x: number;
  /** Semantic-space y coordinate. */
  y: number;
  /** Score. */
  sc: number;
  /** Leiden community id (index into LEIDEN by `id`). */
  lc: number;
}

export const TOPICS: DesignTopic[];
export const PAPERS: DesignPaper[];
