/**
 * parity/registry.ts — what the /parity playground can mount.
 *
 * A rewrite agent adds ONE entry per screen it finishes. Nothing else in the
 * app needs to change: /parity/<id> then mounts it.
 *
 * Keep entries lazy. The point of the playground is to render exactly one
 * screen with nothing else on the page, and an eager import would drag other
 * screens' CSS into the document and quietly change the cascade.
 */

import { lazy } from 'react';
import type { ComponentType, LazyExoticComponent } from 'react';
import type { DemoScreen } from '../design-fonts/manifest';

export interface ParityScreen {
  /** URL segment, e.g. `build` -> /parity/build. */
  id: string;
  /** Product-facing name. */
  label: string;
  /** The demo component this is a rewrite of, for the font/CSS transcript. */
  demoScreen: DemoScreen;
  /**
   * Variant switches this screen must be mounted with. These are LOCKED — the
   * product owner signed off on the demo's defaults. Do not add alternatives.
   */
  lockedVariants?: Readonly<Record<string, string>>;
  component: LazyExoticComponent<ComponentType>;
}

export const PARITY_SCREENS: readonly ParityScreen[] = [
  {
    id: 'build',
    label: 'Build',
    demoScreen: 'Paper Card',
    // The demo file's own defaults. Never flipped; see design-reference/manifest.json.
    lockedVariants: {
      pipelineStyle: 'Flow chart (6d)',
      cardLayout: 'Inline index',
      layout: 'List',
      cardEmphasis: 'Muted',
      logoStyle: 'Terracotta tile',
      colorScheme: 'Warm paper',
      pageState: 'Has results',
    },
    component: lazy(() => import('../screens/build/BuildScreen')),
  },
];

export function findParityScreen(id: string | undefined): ParityScreen | undefined {
  return id ? PARITY_SCREENS.find((s) => s.id === id) : undefined;
}

/**
 * The two comparison viewports the product owner agreed on
 * (decisions ledger, Q15). Sizes in CSS px.
 */
export const PARITY_VIEWPORTS = {
  desktop: [1600, 900],
  ipadPro129: [1366, 1024],
} as const;

export type ParityViewportId = keyof typeof PARITY_VIEWPORTS;
