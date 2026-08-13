/**
 * DemoViewport — the iPad Demo shell's root box, reproduced.
 *
 * Transcribed from the demo bundle's `script[type="__bundler/template"]`:
 *
 *   <div ref="{{ rootRef }}" style="position:relative; width:{{ vpW }};
 *        height:{{ vpH }}; min-width:1024px; margin:0 auto;
 *        box-shadow:0 0 0 1px rgba(20,19,17,.08);">
 *
 * `vpW`/`vpH` come from the shell's `renderVals()`: a `viewport` prop of the
 * form "1600 × 900" pins the box to those pixel sizes, otherwise it falls back
 * to `['100%', '100dvh']` — i.e. fluid. The demo ships with no viewport prop,
 * so fluid is the shipped behaviour.
 *
 * The rotate invite is the shell's own responsive rule, also transcribed:
 *
 *   const rot = () => root.setAttribute('data-rot', root.clientWidth < 1160 ? '1' : '0');
 *   this._rotRO = new ResizeObserver(rot); this._rotRO.observe(root); rot();
 *
 * Note it observes the ROOT's client width, not the window — a 1024–1159 px
 * frame gets the invite even in landscape. Styling for `.rot` lives in
 * styles/ipad-demo-shell.css (verbatim). Do not restyle either here.
 */

import { useEffect, useRef } from 'react';
import type { ReactNode } from 'react';

/** The demo's threshold, in CSS px. Below this the rotate invite covers the shell. */
export const ROTATE_INVITE_BELOW_PX = 1160;

/** The demo's hard floor on the shell box. */
export const SHELL_MIN_WIDTH_PX = 1024;

export interface DemoViewportProps {
  /**
   * Pins the shell to a fixed size, e.g. `[1600, 900]`. Omit for the shipped
   * fluid behaviour (`100%` × `100dvh`). The parity harness uses this to hit
   * the two agreed baselines: 1600×900 and iPad Pro 12.9.
   */
  size?: readonly [number, number];
  children?: ReactNode;
}

export function DemoViewport({ size, children }: DemoViewportProps) {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const rot = () =>
      root.setAttribute('data-rot', root.clientWidth < ROTATE_INVITE_BELOW_PX ? '1' : '0');
    const ro = new ResizeObserver(rot);
    ro.observe(root);
    rot();
    return () => ro.disconnect();
  }, []);

  return (
    <div
      ref={rootRef}
      style={{
        position: 'relative',
        width: size ? `${size[0]}px` : '100%',
        height: size ? `${size[1]}px` : '100dvh',
        minWidth: `${SHELL_MIN_WIDTH_PX}px`,
        margin: '0 auto',
        boxShadow: '0 0 0 1px rgba(20,19,17,.08)',
      }}
    >
      {children}
      <RotateNotice />
    </div>
  );
}

/**
 * Verbatim transcription of the shell's `.rot` block, including its copy.
 * The wording still says "demo"; changing it is a product decision, logged in
 * web/wiring/escalations.md. Do not reword it here.
 */
function RotateNotice() {
  return (
    <div className="rot" data-screen-label="Rotate notice">
      <svg
        width="46"
        height="46"
        viewBox="0 0 24 24"
        fill="none"
        stroke="#c07a5c"
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <rect x="3.5" y="7.5" width="17" height="11" rx="2.2"></rect>
        <path d="M7.5 4.2A9 9 0 0 1 19 2.6"></path>
      </svg>
      <div
        className="rot-t"
        style={{
          fontFamily: "'Newsreader',serif",
          fontSize: '29px',
          fontWeight: 500,
          letterSpacing: '-0.01em',
        }}
      >
        Rotate to landscape
      </div>
      <div
        className="rot-s"
        style={{
          maxWidth: '330px',
          fontFamily: "'Hanken Grotesk',-apple-system,sans-serif",
          fontSize: '14.5px',
          lineHeight: 1.55,
          textWrap: 'pretty',
        }}
      >
        This demo is laid out for the iPad held wide. Turn the device and it picks up where you
        left off.
      </div>
    </div>
  );
}
