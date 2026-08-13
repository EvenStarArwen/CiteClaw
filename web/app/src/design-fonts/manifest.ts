/**
 * design-fonts/manifest.ts
 *
 * The exact stylesheet refs each KnowledgeLab iPad Demo screen carries in its
 * `<helmet>`, transcribed verbatim from the demo bundle. Nothing here is a
 * design choice — it is a transcript.
 *
 * Why it matters: support.js's helmet manager hoists every mounted screen's
 * <link> into the ONE shared document head (deduped by href), so the demo's
 * effective font environment is the union, in mount order. index.html carries
 * that union. This module exists so a rewrite agent can check what a single
 * screen was designed against, and so the parity harness can assert that the
 * union is still complete.
 *
 * Provenance: "KnowledgeLab iPad Demo.html" -> script[type="__bundler/manifest"]
 * + script[type="__bundler/ext_resources"]. Verified 2026-08-13.
 */

/** Screen names as they appear in the demo shell's `dc-import name="…"`. */
export type DemoScreen =
  | 'Login'
  | 'Home'
  | 'Paper Card'
  | 'Runs'
  | 'Explore'
  | 'Settings'
  | 'System Banners';

/** Mount order of the demo shell template (top to bottom in the template). */
export const DEMO_SCREEN_MOUNT_ORDER: readonly DemoScreen[] = [
  'Login',
  'Home',
  'Paper Card',
  'Runs',
  'Explore',
  'Settings',
  'System Banners',
];

/** The screen the product calls "Build" is the demo component named "Paper Card". */
export const BUILD_SCREEN: DemoScreen = 'Paper Card';

/** Every screen carries these two, identically. */
export const PRECONNECTS: readonly string[] = [
  'https://fonts.googleapis.com',
  'https://fonts.gstatic.com',
];

/**
 * The four distinct Google Fonts hrefs, in first-appearance (mount) order.
 * These strings must stay byte-identical to the demo's `href` attributes.
 */
export const FONT_STYLESHEETS = {
  /** Newsreader 400/500/600 + Hanken Grotesk VARIABLE 400..700. */
  newsreaderRoman_hankenVariable:
    'https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Hanken+Grotesk:wght@400..700&display=swap',
  /** Newsreader 400/500/600 + Hanken Grotesk FIXED 400;500;600. */
  newsreaderRoman_hankenFixed:
    'https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Hanken+Grotesk:wght@400;500;600&display=swap',
  /** Newsreader roman + italic + Hanken Grotesk FIXED 400;500;600. */
  newsreaderItalic_hankenFixed:
    'https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400;1,6..72,500&family=Hanken+Grotesk:wght@400;500;600&display=swap',
  /** Hanken Grotesk VARIABLE 400..700 only (no Newsreader). */
  hankenVariableOnly:
    'https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400..700&display=swap',
} as const;

/** Third-party stylesheets that are NOT fonts. Explore only. */
export const KATEX_STYLESHEET =
  'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css';

/** Per-screen transcript of the stylesheet refs found in that screen's helmet. */
export const SCREEN_STYLESHEETS: Record<DemoScreen, readonly string[]> = {
  Login: [FONT_STYLESHEETS.newsreaderRoman_hankenVariable],
  Home: [FONT_STYLESHEETS.newsreaderRoman_hankenVariable],
  'Paper Card': [FONT_STYLESHEETS.newsreaderRoman_hankenFixed],
  Runs: [FONT_STYLESHEETS.newsreaderRoman_hankenFixed],
  Explore: [FONT_STYLESHEETS.newsreaderItalic_hankenFixed, KATEX_STYLESHEET],
  Settings: [FONT_STYLESHEETS.newsreaderRoman_hankenVariable],
  'System Banners': [FONT_STYLESHEETS.hankenVariableOnly],
};

/**
 * What index.html must contain, in this order, for the union to match the demo.
 * The parity harness can compare this against the live document.
 */
export const GLOBAL_FONT_STYLESHEETS: readonly string[] = [
  FONT_STYLESHEETS.newsreaderRoman_hankenVariable,
  FONT_STYLESHEETS.newsreaderRoman_hankenFixed,
  FONT_STYLESHEETS.newsreaderItalic_hankenFixed,
  FONT_STYLESHEETS.hankenVariableOnly,
];

/** The two font stacks the demo's markup names, verbatim. */
export const FONT_STACKS = {
  serif: "'Newsreader',serif",
  sans: "'Hanken Grotesk',-apple-system,sans-serif",
  mono: 'ui-monospace,monospace',
} as const;

/**
 * True if the document already links every stylesheet in `hrefs`.
 * Used by the parity harness; cheap enough to call in a test.
 */
export function documentHasStylesheets(hrefs: readonly string[]): boolean {
  const present = new Set(
    Array.from(document.querySelectorAll<HTMLLinkElement>('link[rel~="stylesheet"]')).map(
      (el) => el.getAttribute('href') ?? '',
    ),
  );
  return hrefs.every((h) => present.has(h));
}
