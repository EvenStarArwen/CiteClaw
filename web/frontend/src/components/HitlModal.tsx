import { useState, useCallback } from "react"
import { useAppStore } from "../lib/store"

/**
 * HitlModal — Human-in-the-Loop labelling dialog (PE-09).
 *
 * Shown when the pipeline's HumanInTheLoop step emits an `hitl_request`
 * event via WebSocket. Displays sampled papers one-by-one with yes/no
 * buttons. On submit, POSTs labels to `/api/runs/{runId}/hitl`.
 */

export interface HitlPaper {
  paper_id: string
  title: string
  venue: string
  year: number | null
  abstract: string
}

export function HitlModal() {
  const pipelineRunId = useAppStore((s) => s.pipelineRunId)
  const hitlPapers = useAppStore((s) => s.hitlPapers)
  const clearHitl = useAppStore((s) => s.clearHitl)

  const [labels, setLabels] = useState<Record<string, boolean>>({})
  const [currentIdx, setCurrentIdx] = useState(0)
  const [submitted, setSubmitted] = useState(false)

  const handleLabel = useCallback(
    (paperId: string, keep: boolean) => {
      setLabels((prev) => ({ ...prev, [paperId]: keep }))
      setCurrentIdx((i) => i + 1)
    },
    [],
  )

  const handleSubmit = useCallback(
    async (stopPipeline: boolean) => {
      if (!pipelineRunId) return
      try {
        const resp = await fetch(`/api/runs/${pipelineRunId}/hitl`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            labels,
            stop_requested: stopPipeline,
          }),
        })
        if (!resp.ok) {
          console.error("HITL submit failed:", await resp.text())
        }
      } catch (err) {
        console.error("HITL submit error:", err)
      }
      setSubmitted(true)
      setTimeout(() => {
        clearHitl()
        setLabels({})
        setCurrentIdx(0)
        setSubmitted(false)
      }, 1500)
    },
    [pipelineRunId, labels, clearHitl],
  )

  if (!hitlPapers || hitlPapers.length === 0) return null

  const allLabelled = currentIdx >= hitlPapers.length
  const current = !allLabelled ? hitlPapers[currentIdx] : null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-full max-w-lg mx-4 p-6">
        <h2 className="text-lg font-semibold text-gray-100 mb-1">
          Human-in-the-Loop Review
        </h2>
        <p className="text-sm text-gray-400 mb-4">
          Label each paper: is it in scope for your collection?
        </p>

        {/* Progress bar */}
        <div className="w-full bg-gray-800 rounded-full h-2 mb-4">
          <div
            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
            style={{
              width: `${(Object.keys(labels).length / hitlPapers.length) * 100}%`,
            }}
          />
        </div>
        <p className="text-xs text-gray-500 mb-4">
          {Object.keys(labels).length} / {hitlPapers.length} labelled
        </p>

        {submitted ? (
          <div className="text-center py-8">
            <p className="text-green-400 text-lg font-medium">
              Labels submitted
            </p>
          </div>
        ) : !allLabelled && current ? (
          <div>
            <div className="bg-gray-800 rounded-lg p-4 mb-4">
              <h3 className="text-sm font-medium text-gray-200 mb-1">
                {current.title}
              </h3>
              <p className="text-xs text-gray-400 mb-2">
                {current.venue} · {current.year ?? "?"}
              </p>
              {current.abstract && (
                <p className="text-xs text-gray-300 leading-relaxed max-h-32 overflow-y-auto">
                  {current.abstract}
                </p>
              )}
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => handleLabel(current.paper_id, false)}
                className="flex-1 py-2 px-4 rounded-lg bg-red-900/50 hover:bg-red-800/70 text-red-300 text-sm font-medium transition-colors"
              >
                Out of scope
              </button>
              <button
                onClick={() => handleLabel(current.paper_id, true)}
                className="flex-1 py-2 px-4 rounded-lg bg-green-900/50 hover:bg-green-800/70 text-green-300 text-sm font-medium transition-colors"
              >
                In scope
              </button>
            </div>
          </div>
        ) : (
          <div>
            <p className="text-sm text-gray-300 mb-4">
              All papers labelled. Continue or stop the pipeline?
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => handleSubmit(true)}
                className="flex-1 py-2 px-4 rounded-lg bg-yellow-900/50 hover:bg-yellow-800/70 text-yellow-300 text-sm font-medium transition-colors"
              >
                Stop pipeline
              </button>
              <button
                onClick={() => handleSubmit(false)}
                className="flex-1 py-2 px-4 rounded-lg bg-blue-900/50 hover:bg-blue-800/70 text-blue-300 text-sm font-medium transition-colors"
              >
                Continue
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
