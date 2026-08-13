/**
 * main.tsx — app entry.
 *
 * The two CSS imports below are load-bearing and order-sensitive; see
 * ./styles/README.md. They must stay at the top, above every other import, so
 * that per-screen CSS brought over by the rewrite agents lands after them and
 * wins at equal specificity.
 */
import './styles/explorations-tokens.css';
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
