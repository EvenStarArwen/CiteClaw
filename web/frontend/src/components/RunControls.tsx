import { useState } from "react"
import { useAppStore } from "../lib/store"
import type { PipelineStepInfo } from "../lib/store"

function StepRow({ step }: { step: PipelineStepInfo }) {
  const isRunning = step.status === "running"
  return (
    <div
      className={`flex items-center gap-2 py-1.5 px-2 rounded text-sm ${
        isRunning ? "bg-indigo-950/50 text-indigo-300" : "text-gray-300"
      }`}
    >
      <span className="w-5 text-center">
        {isRunning ? (
          <span className="inline-block w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
        ) : (
          <span className="text-emerald-400 text-xs">ok</span>
        )}
      </span>
      <span className="flex-1 truncate font-medium">{step.name}</span>
      {step.status === "done" && step.delta_collection != null && (
        <span
          className={`text-xs font-mono ${
            step.delta_collection > 0 ? "text-emerald-400" : "text-gray-500"
          }`}
        >
          {step.delta_collection > 0 ? "+" : ""}
          {step.delta_collection}
        </span>
      )}
    </div>
  )
}

function BudgetSummary({ steps }: { steps: PipelineStepInfo[] }) {
  const totalIn = steps.reduce((s, st) => s + (st.in_count ?? 0), 0)
  const totalAccepted = steps.reduce((s, st) => s + (st.out_count ?? 0), 0)
  const totalDelta = steps.reduce((s, st) => s + (st.delta_collection ?? 0), 0)
  const doneCount = steps.filter((s) => s.status === "done").length

  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
      <span className="text-gray-400">Steps done</span>
      <span className="text-gray-200 font-mono text-right">
        {doneCount}/{steps.length}
      </span>
      <span className="text-gray-400">Total in</span>
      <span className="text-gray-200 font-mono text-right">{totalIn.toLocaleString()}</span>
      <span className="text-gray-400">Total out</span>
      <span className="text-gray-200 font-mono text-right">{totalAccepted.toLocaleString()}</span>
      <span className="text-gray-400">Collection delta</span>
      <span className="text-emerald-400 font-mono text-right">
        {totalDelta > 0 ? "+" : ""}
        {totalDelta.toLocaleString()}
      </span>
    </div>
  )
}

export function RunControls() {
  const pipelineRunId = useAppStore((s) => s.pipelineRunId)
  const pipelineSteps = useAppStore((s) => s.pipelineSteps)
  const shapeTable = useAppStore((s) => s.shapeTable)
  const startRun = useAppStore((s) => s.startRun)
  const resetRun = useAppStore((s) => s.resetRun)
  const liveNodes = useAppStore((s) => s.liveNodes)

  const [configName, setConfigName] = useState("config.yaml")
  const [isStarting, setIsStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const isRunning = pipelineSteps.some((s) => s.status === "running")
  const isDone = pipelineSteps.length > 0 && pipelineSteps.every((s) => s.status === "done")

  async function handleStart() {
    setIsStarting(true)
    setError(null)
    try {
      const res = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config_name: configName }),
      })
      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: "Request failed" }))
        throw new Error(detail.detail ?? `HTTP ${res.status}`)
      }
      const data = await res.json()
      startRun(data.run_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error")
    } finally {
      setIsStarting(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Start Run */}
      <div>
        <label className="block text-xs text-gray-400 mb-1">Config</label>
        <input
          type="text"
          value={configName}
          onChange={(e) => setConfigName(e.target.value)}
          className="w-full px-2 py-1.5 bg-gray-800 border border-gray-700 rounded text-sm text-gray-200 focus:outline-none focus:border-indigo-500"
          placeholder="config.yaml"
          disabled={isRunning}
        />
      </div>
      <div className="flex gap-2">
        <button
          onClick={handleStart}
          disabled={isRunning || isStarting || !configName.trim()}
          className="flex-1 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 text-sm font-medium rounded transition-colors"
        >
          {isStarting ? "Starting..." : isRunning ? "Running..." : "Start Run"}
        </button>
        {(isDone || pipelineRunId) && !isRunning && (
          <button
            onClick={resetRun}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-sm rounded transition-colors"
          >
            Reset
          </button>
        )}
      </div>
      {error && <p className="text-xs text-red-400">{error}</p>}

      {/* Live Progress */}
      {pipelineRunId && (
        <>
          <div className="border-t border-gray-800 pt-3">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
                Pipeline Steps
              </h4>
              {isRunning && (
                <span className="text-xs text-indigo-400 animate-pulse">live</span>
              )}
              {isDone && <span className="text-xs text-emerald-400">done</span>}
            </div>
            <div className="space-y-0.5">
              {[...pipelineSteps]
                .sort((a, b) => a.idx - b.idx)
                .map((step) => (
                  <StepRow key={step.idx} step={step} />
                ))}
            </div>
          </div>

          {/* Budget summary */}
          {pipelineSteps.length > 0 && (
            <div className="border-t border-gray-800 pt-3">
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">
                Budget
              </h4>
              <BudgetSummary steps={pipelineSteps} />
              <div className="mt-2 text-sm">
                <span className="text-gray-400">Papers discovered</span>
                <span className="text-gray-200 font-mono float-right">
                  {liveNodes.length.toLocaleString()}
                </span>
              </div>
            </div>
          )}

          {/* Shape Table */}
          {shapeTable && (
            <div className="border-t border-gray-800 pt-3">
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">
                Shape Summary
              </h4>
              <pre className="text-xs text-gray-400 font-mono whitespace-pre-wrap overflow-x-auto bg-gray-900 rounded p-2 max-h-64 overflow-y-auto">
                {shapeTable}
              </pre>
            </div>
          )}
        </>
      )}

      {/* Idle state */}
      {!pipelineRunId && (
        <p className="text-sm text-gray-500">
          Enter a config file name and click Start Run to begin a pipeline.
        </p>
      )}
    </div>
  )
}
