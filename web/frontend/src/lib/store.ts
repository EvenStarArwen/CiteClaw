import { create } from "zustand"

/* ---- Pipeline event types (match backend event_sink.py) ---- */

export interface PipelineStepInfo {
  idx: number
  name: string
  description: string
  in_count?: number
  out_count?: number
  delta_collection?: number
  stats?: Record<string, unknown>
  status: "running" | "done"
}

export interface LiveNode {
  paper_id: string
  source: string
  /** Timestamp when added — used for entrance animation */
  addedAt: number
}

/* ---- Store ---- */

interface AppState {
  darkMode: boolean
  toggleDarkMode: () => void
  selectedPaperId: string | null
  selectPaper: (id: string | null) => void

  /* Live pipeline run state */
  pipelineRunId: string | null
  pipelineSteps: PipelineStepInfo[]
  liveNodes: LiveNode[]
  shapeTable: string | null

  stepStart: (idx: number, name: string, description: string) => void
  stepEnd: (
    idx: number,
    name: string,
    in_count: number,
    out_count: number,
    delta_collection: number,
    stats: Record<string, unknown>,
  ) => void
  paperAdded: (paper_id: string, source: string) => void
  shapeTableUpdate: (rendered: string) => void
  startRun: (runId: string) => void
  resetRun: () => void
}

export const useAppStore = create<AppState>((set) => ({
  darkMode: true,
  toggleDarkMode: () => set((s) => ({ darkMode: !s.darkMode })),
  selectedPaperId: null,
  selectPaper: (id) => set({ selectedPaperId: id }),

  /* Live pipeline defaults */
  pipelineRunId: null,
  pipelineSteps: [],
  liveNodes: [],
  shapeTable: null,

  startRun: (runId) =>
    set({ pipelineRunId: runId, pipelineSteps: [], liveNodes: [], shapeTable: null }),

  resetRun: () =>
    set({ pipelineRunId: null, pipelineSteps: [], liveNodes: [], shapeTable: null }),

  stepStart: (idx, name, description) =>
    set((s) => ({
      pipelineSteps: [
        ...s.pipelineSteps.filter((st) => st.idx !== idx),
        { idx, name, description, status: "running" as const },
      ],
    })),

  stepEnd: (idx, name, in_count, out_count, delta_collection, stats) =>
    set((s) => ({
      pipelineSteps: s.pipelineSteps.map((st) =>
        st.idx === idx
          ? { ...st, name, in_count, out_count, delta_collection, stats, status: "done" as const }
          : st,
      ),
    })),

  paperAdded: (paper_id, source) =>
    set((s) => ({
      liveNodes: [...s.liveNodes, { paper_id, source, addedAt: Date.now() }],
    })),

  shapeTableUpdate: (rendered) => set({ shapeTable: rendered }),
}))
