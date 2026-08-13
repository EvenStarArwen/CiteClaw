/**
 * ParityRoute — the parity playground.
 *
 * /parity            lists the screens registered in parity/registry.ts
 * /parity/<id>       mounts exactly that screen, and nothing else
 *
 * Query params (read by the harness in web/parity/, not by hand):
 *   ?viewport=desktop | ipadPro129 | fluid   default: fluid
 *   ?chrome=0                                hide the index's own chrome
 *
 * The chrome this route draws is styled inline with system fonts on purpose:
 * it must not consume design tokens, so it can never be mistaken for the
 * rewrite and can never perturb the cascade being measured.
 */

import { Suspense } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { DemoViewport } from '../components/DemoViewport';
import { useDataSource } from '../data';
import {
  PARITY_SCREENS,
  PARITY_VIEWPORTS,
  findParityScreen,
  type ParityViewportId,
} from '../parity/registry';

const chromeFont =
  '13px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

function parseViewport(v: string | null): readonly [number, number] | undefined {
  if (!v || v === 'fluid') return undefined;
  return PARITY_VIEWPORTS[v as ParityViewportId];
}

export default function ParityRoute() {
  const { screenId } = useParams();
  const [params] = useSearchParams();
  const source = useDataSource();
  const screen = findParityScreen(screenId);
  const size = parseViewport(params.get('viewport'));

  if (screenId && !screen) {
    return (
      <main style={{ font: chromeFont, padding: 24, color: '#57554e' }}>
        <h1 style={{ font: chromeFont, fontWeight: 600, marginBottom: 8 }}>
          No parity screen registered as “{screenId}”
        </h1>
        <p style={{ marginBottom: 12 }}>
          Add it to <code>src/parity/registry.ts</code>, then reload.
        </p>
        <Link to="/parity">Back to the index</Link>
      </main>
    );
  }

  if (screen) {
    const Screen = screen.component;
    return (
      <DemoViewport size={size}>
        <Suspense fallback={null}>
          <Screen />
        </Suspense>
      </DemoViewport>
    );
  }

  return (
    <main style={{ font: chromeFont, padding: 24, color: '#57554e', maxWidth: 720 }}>
      <h1 style={{ font: chromeFont, fontSize: 16, fontWeight: 600, marginBottom: 4 }}>
        Parity playground
      </h1>
      <p style={{ marginBottom: 16 }}>
        Data source: <strong>{source.label}</strong> (<code>{source.id}</code>)
      </p>

      {PARITY_SCREENS.length === 0 ? (
        <p>
          No screens registered yet. A rewrite agent registers one by adding an entry to{' '}
          <code>src/parity/registry.ts</code>; it then mounts at{' '}
          <code>/parity/&lt;id&gt;</code>.
        </p>
      ) : (
        <ul style={{ paddingLeft: 18 }}>
          {PARITY_SCREENS.map((s) => (
            <li key={s.id} style={{ marginBottom: 6 }}>
              <Link to={`/parity/${s.id}`}>{s.label}</Link>{' '}
              <span style={{ color: '#8e8b82' }}>
                — demo component “{s.demoScreen}”
              </span>
              {' · '}
              <Link to={`/parity/${s.id}?viewport=desktop`}>1600×900</Link>
              {' · '}
              <Link to={`/parity/${s.id}?viewport=ipadPro129`}>iPad Pro 12.9</Link>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
