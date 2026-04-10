import { create } from "zustand"

interface AppState {
  darkMode: boolean
  toggleDarkMode: () => void
  selectedPaperId: string | null
  selectPaper: (id: string | null) => void
}

export const useAppStore = create<AppState>((set) => ({
  darkMode: true,
  toggleDarkMode: () => set((s) => ({ darkMode: !s.darkMode })),
  selectedPaperId: null,
  selectPaper: (id) => set({ selectedPaperId: id }),
}))
