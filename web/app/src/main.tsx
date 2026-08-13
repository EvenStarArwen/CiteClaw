/**
 * main.tsx — app entry.
 *
 * The CSS import below is load-bearing and order-sensitive; see
 * ./styles/README.md. It must stay at the top, above every other import, so
 * that screen CSS lands after it and wins at equal specificity. It is the demo
 * shell's own <style> block, verbatim.
 *
 * `explorations-tokens.css` used to be imported here as well. It is NOT — it is
 * not loaded by the iPad Demo (no screen references it), so every rule in it
 * that the demo's own stylesheets do not also carry lands on the app as a
 * silent, invented override. Measured on the Build pilot: the tokens file's
 *
 *     .cfg-pipe[data-number="Column"] .cf-idx-col { font-family: ui-monospace,
 *       monospace; … }
 *     .cf-num { font-family: ui-monospace, monospace; … }
 *
 * have no counterpart in any demo screen — the demo's copies of those two rules
 * omit font-family — so nothing overrode them and the config panel's five
 * filter-row numbers rendered in a monospace face the design never used. That
 * is 51 wrong pixels the parity harness caught, and it would have been many
 * more on the screens still to be rewritten.
 *
 * The file is deliberately kept in ./styles (it is a verbatim copy of a design
 * asset and may yet be regenerated FROM the demo) but nothing imports it.
 * Escalated as E-BLD-02 in web/wiring/escalations.md; supersedes the open
 * question in E-SCAF-01.
 */
import './styles/ipad-demo-shell.css';

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';

const container = document.getElementById('root');
if (!container) {
  throw new Error('main.tsx: #root is missing from index.html');
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
