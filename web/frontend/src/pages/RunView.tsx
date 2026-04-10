import { useParams } from "react-router-dom"

export function RunView() {
  const { runId } = useParams()
  return (
    <div className="text-center">
      <h1 className="text-2xl font-bold tracking-tight mb-2">Run: {runId}</h1>
      <p className="text-sm text-gray-500">
        Graph visualization will render here.
      </p>
    </div>
  )
}
