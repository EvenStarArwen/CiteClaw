/**
 * App.tsx — router shell.
 *
 * Two routes only, on purpose:
 *   /        the app itself. Empty until the rewritten screens land.
 *   /parity  the parity playground: mounts ONE rewritten screen at a time,
 *            bare, so it can be diffed against the iPad Demo.
 *
 * The shell deliberately renders no chrome of its own. Anything it drew would
 * show up in every parity screenshot.
 */

import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { DataSourceProvider } from './data';

const HomeRoute = lazy(() => import('./routes/HomeRoute'));
const ParityRoute = lazy(() => import('./routes/ParityRoute'));

export function App() {
  return (
    <DataSourceProvider>
      <BrowserRouter>
        <Suspense fallback={null}>
          <Routes>
            <Route path="/" element={<HomeRoute />} />
            <Route path="/parity" element={<ParityRoute />} />
            <Route path="/parity/:screenId" element={<ParityRoute />} />
            <Route path="*" element={<HomeRoute />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </DataSourceProvider>
  );
}
