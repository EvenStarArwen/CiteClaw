import { useEffect, useRef, useState } from "react"
import Graph from "graphology"

export interface GraphNode {
  paper_id: string
  title: string
  year: number | null
  venue: string | null
  citation_count: number
  cluster: string | null
  source: string | null
}

export interface GraphEdge {
  source: string
  target: string
  weight: number
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

// Deterministic color palette for clusters
const CLUSTER_COLORS = [
  "#6366f1", // indigo
  "#f59e0b", // amber
  "#10b981", // emerald
  "#ef4444", // red
  "#8b5cf6", // violet
  "#06b6d4", // cyan
  "#f97316", // orange
  "#84cc16", // lime
  "#ec4899", // pink
  "#14b8a6", // teal
]

export const SOURCE_COLORS: Record<string, string> = {
  seed: "#facc15",
  forward: "#60a5fa",
  backward: "#34d399",
  search: "#f472b6",
  semantic: "#a78bfa",
  author: "#fb923c",
  reinforced: "#f87171",
}

export function getNodeColor(node: GraphNode): string {
  if (node.cluster != null) {
    const idx = Math.abs(parseInt(node.cluster, 10)) % CLUSTER_COLORS.length
    return CLUSTER_COLORS[idx] ?? "#9ca3af"
  }
  if (node.source) {
    return SOURCE_COLORS[node.source] ?? "#9ca3af"
  }
  return "#9ca3af"
}

export function getNodeSize(citationCount: number): number {
  return 3 + Math.log2(Math.max(1, citationCount)) * 1.5
}

export function useSigmaGraph(data: GraphData | undefined) {
  const graphRef = useRef<Graph | null>(null)
  const [graph, setGraph] = useState<Graph | null>(null)

  useEffect(() => {
    if (!data) return

    const g = new Graph()

    for (const node of data.nodes) {
      const x = Math.random() * 100 - 50
      const y = Math.random() * 100 - 50
      g.addNode(node.paper_id, {
        label: node.title,
        x,
        y,
        size: getNodeSize(node.citation_count),
        color: getNodeColor(node),
        // Store metadata for click handler
        year: node.year,
        venue: node.venue,
        source: node.source,
        cluster: node.cluster,
        citation_count: node.citation_count,
      })
    }

    for (const edge of data.edges) {
      if (g.hasNode(edge.source) && g.hasNode(edge.target)) {
        const key = `${edge.source}->${edge.target}`
        if (!g.hasEdge(key)) {
          g.addEdgeWithKey(key, edge.source, edge.target, {
            weight: edge.weight,
            size: Math.max(0.5, edge.weight * 2),
            color: "#4b5563",
          })
        }
      }
    }

    graphRef.current = g
    setGraph(g)

    return () => {
      g.clear()
    }
  }, [data])

  return graph
}
