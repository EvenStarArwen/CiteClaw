/**
 * data/DataSourceContext.tsx — how screens reach the DataSource.
 *
 * One provider at the app root, one hook in the screens. Swapping providers is
 * a change to the value passed here and nothing else.
 */

import { createContext, useContext } from 'react';
import type { ReactNode } from 'react';
import type { DataSource } from './types';
import { designDataSource } from './designDataSource';

const DataSourceContext = createContext<DataSource>(designDataSource);

export function DataSourceProvider({
  source = designDataSource,
  children,
}: {
  source?: DataSource;
  children: ReactNode;
}) {
  return <DataSourceContext.Provider value={source}>{children}</DataSourceContext.Provider>;
}

/** The active provider. Never import a concrete provider inside a screen. */
export function useDataSource(): DataSource {
  return useContext(DataSourceContext);
}
