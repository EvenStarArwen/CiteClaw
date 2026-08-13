// Ambient types for the VERBATIM demo module `community-data.js`.

/** A Leiden community over the citation graph (network space). */
export interface DesignCommunity {
  id: number;
  name: string;
  desc: string;
  kw: string[];
  /** Member count. */
  n: number;
  /** Centroid x in network space. */
  cx: number;
  /** Centroid y in network space. */
  cy: number;
}

export const LEIDEN: DesignCommunity[];
/** CSS colour expressions (they reference design tokens; do not parse as hex). */
export const COMMUNITY_COLORS: string[];
export const PERIPHERAL_COLOR: string;
/** Communities at or below this size collapse into the "Peripheral" card. */
export const PERIPHERAL_MAX: number;
/** Sentinel community id for the "Peripheral" bucket. */
export const PERIPHERAL_ID: number;
export function communityColor(id: number): string;
