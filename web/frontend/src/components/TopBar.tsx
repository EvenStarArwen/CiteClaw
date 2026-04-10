import { Moon, Sun } from "lucide-react"
import { useAppStore } from "../lib/store"

export function TopBar() {
  const { darkMode, toggleDarkMode } = useAppStore()

  return (
    <header className="flex items-center justify-between h-12 px-4 border-b border-gray-700 bg-gray-900">
      <div className="flex items-center gap-2">
        <span className="text-lg font-bold tracking-tight text-white">
          CiteClaw
        </span>
        <span className="text-xs text-gray-500">snowballing literature acquisition</span>
      </div>
      <button
        onClick={toggleDarkMode}
        className="p-1.5 rounded hover:bg-gray-800 text-gray-400 hover:text-white transition-colors"
        aria-label="Toggle dark mode"
      >
        {darkMode ? <Sun size={16} /> : <Moon size={16} />}
      </button>
    </header>
  )
}
