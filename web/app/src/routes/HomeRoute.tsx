/**
 * HomeRoute — the real app's entry screen. Intentionally empty.
 *
 * The rewritten Home screen replaces the body of `DemoViewport` below. The
 * viewport wrapper is NOT a placeholder: it reproduces the iPad Demo shell's
 * root box exactly (see components/DemoViewport.tsx), and the demo's
 * responsive behaviour depends on it.
 */

import { DemoViewport } from '../components/DemoViewport';

export default function HomeRoute() {
  return <DemoViewport />;
}
