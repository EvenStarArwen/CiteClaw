import { useParams } from "react-router-dom"
import { useQuery } from "@tanstack/react-query"
import { GraphView } from "../components/Graph"
import type { GraphData } from "../hooks/useSigmaGraph"

async function fetchGraph(runId: string): Promise<GraphData> {
  const res = await fetch(`/api/runs/${runId}/graph`)
  if (!res.ok) throw new Error(`Failed to fetch graph: ${res.status}`)
  return res.json()
}

export function RunView() {
  const { runId } = useParams()

  const { data, isLoading, error } = useQuery<GraphData, Error>({
    queryKey: ["graph", runId],
    queryFn: () => fetchGraph(runId!),
    enabled: !!runId,
  })

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-2 border-b border-gray-700">
        <h1 className="text-lg font-semibold tracking-tight">Run: {runId}</h1>
      </div>
      <div className="flex-1 min-h-0">
        <GraphView data={data} isLoading={isLoading} error={error ?? null} />
      </div>
    </div>
  )
}
