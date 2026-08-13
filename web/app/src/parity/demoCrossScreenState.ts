/**
 * demoCrossScreenState — inputs a demo screen receives from its SIBLINGS.
 *
 * PARITY PLAYGROUND ONLY. Nothing in src/screens or src/components imports
 * this; it exists because the parity playground mounts one screen at a time
 * while the iPad Demo mounts all seven into one document, and a few of the
 * demo's screens are wired to each other through `document` events and globals.
 * Measuring one screen in isolation against a baseline captured with all seven
 * present means feeding it the same inputs — otherwise the harness reports a
 * behaviour difference that does not exist.
 *
 * There is exactly one such input on Build today:
 *
 *   Runs.dc.html's `klRunActivity(mode)` does
 *       const d = { state: st, label: this.rid(this.liveId()) };
 *       window.__klRunAct = d;
 *       document.dispatchEvent(new CustomEvent('kl-run-activity', {detail:d}));
 *   and EVERY page's top bar listens for it (`actPaint` in Build's
 *   componentDidMount). With the demo's locked defaults the Runs screen is in
 *   its `Running` state on run 38, so the Build screen's Runs tab shows a
 *   pulsing green `.tb-runs-dot` and the tab gets
 *   title="Run 38 is running". That is 16 px of the baseline.
 *
 * The value below is transplanted, not invented: it is what the demo's own
 * Runs screen publishes, read back out of the running demo. When the Runs
 * screen is rewritten it will publish this itself and this module should be
 * deleted, not extended.
 *
 * Do NOT move this into BuildScreen. Build does not own this state; asserting
 * a running pipeline from the Build screen would be inventing product
 * behaviour, and it would survive into the real app.
 */

/** Shape of the `kl-run-activity` detail, per Runs.dc.html. */
export interface DemoRunActivity {
  state: 'none' | 'running' | 'finished';
  label: string;
}

declare global {
  // eslint-disable-next-line no-var
  var __klRunAct: DemoRunActivity | undefined;
}

/**
 * What the demo's Runs screen has published by the time a capture settles,
 * under the locked variant defaults (runState = 'Running', run id RUN-0038).
 */
export const DEMO_RUN_ACTIVITY: DemoRunActivity = { state: 'running', label: 'Run 38' };

let applied = false;

/**
 * Publish the sibling-screen state exactly as Runs does. Idempotent, and safe
 * to call during render: the top bar reads `window.__klRunAct` from a
 * `setTimeout(…, 0)` in componentDidMount and also listens for the event, so
 * either path picks this up.
 */
export function applyDemoCrossScreenState(): void {
  if (applied || typeof document === 'undefined') return;
  applied = true;
  globalThis.__klRunAct = DEMO_RUN_ACTIVITY;
  document.dispatchEvent(
    new CustomEvent('kl-run-activity', { detail: DEMO_RUN_ACTIVITY }),
  );
}
