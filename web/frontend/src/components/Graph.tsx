import { useEffect, useRef } from "react"
import {
  SigmaContainer,
  useRegisterEvents,
  useSigma,
} from "@react-sigma/core"
import "@react-sigma/core/lib/style.css"
import forceAtlas2 from "graphology-layout-forceatlas2"
import type Graph from "graphology"
import { useAppStore } from "../lib/store"
import { useSigmaGraph, type GraphData } from "../hooks/useSigmaGraph"

/** Inner component that lives inside SigmaContainer and registers events */
function GraphEvents() {
  const sigma = useSigma()
  const registerEvents = useRegisterEvents()
  const selectPaper = useAppStore((s) => s.selectPaper)

  useEffect(() => {
    registerEvents({
      clickNode: (event) => {
        selectPaper(event.node)
      },
      clickStage: () => {
        selectPaper(null)
      },
    })
  }, [registerEvents, selectPaper])

  // Highlight selected node
  const selectedPaperId = useAppStore((s) => s.selectedPaperId)
  useEffect(() => {
    const graph = sigma.getGraph()
    graph.forEachNode((node, attrs) => {
      if (selectedPaperId && node !== selectedPaperId) {
        graph.setNodeAttribute(node, "color", attrs._origColor ?? attrs.color)
        graph.setNodeAttribute(node, "zIndex", 0)
      } else if (node === selectedPaperId) {
        if (!attrs._origColor) {
          graph.setNodeAttribute(node, "_origColor", attrs.color)
        }
        graph.setNodeAttribute(node, "color", "#ffffff")
        graph.setNodeAttribute(node, "zIndex", 1)
      }
    })
    sigma.refresh()
  }, [selectedPaperId, sigma])

  return null
}

/** Runs ForceAtlas 2 layout continuously at low intensity */
function FA2Layout({ graph }: { graph: Graph }) {
  const animFrameRef = useRef<number>(0)
  const sigma = useSigma()

  useEffect(() => {
    let running = true
    const settings = forceAtlas2.inferSettings(graph)
    settings.slowDown = 10
    settings.barnesHutOptimize = graph.order > 100

    function step() {
      if (!running) return
      forceAtlas2.assign(graph, { settings, iterations: 1 })
      sigma.refresh()
      animFrameRef.current = requestAnimationFrame(step)
    }

    animFrameRef.current = requestAnimationFrame(step)

    return () => {
      running = false
      cancelAnimationFrame(animFrameRef.current)
    }
  }, [graph, sigma])

  return null
}

interface GraphViewProps {
  data: GraphData | undefined
  isLoading?: boolean
  error?: Error | null
}

export function GraphView({ data, isLoading, error }: GraphViewProps) {
  const graph = useSigmaGraph(data)

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400">
        Loading graph...
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-red-400">
        Failed to load graph: {error.message}
      </div>
    )
  }

  if (!graph || graph.order === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        No graph data available.
      </div>
    )
  }

  return (
    <div className="w-full h-full">
      <SigmaContainer
        graph={graph}
        style={{ width: "100%", height: "100%" }}
        settings={{
          renderEdgeLabels: false,
          defaultEdgeType: "arrow",
          labelSize: 12,
          labelWeight: "bold",
          labelRenderedSizeThreshold: 8,
        }}
      >
        <GraphEvents />
        <FA2Layout graph={graph} />
      </SigmaContainer>
    </div>
  )
}
