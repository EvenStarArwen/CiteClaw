import { useEffect, useRef, useCallback } from "react"
import {
  SigmaContainer,
  useRegisterEvents,
  useSigma,
} from "@react-sigma/core"
import "@react-sigma/core/lib/style.css"
import forceAtlas2 from "graphology-layout-forceatlas2"
import type Graph from "graphology"
import { useAppStore } from "../lib/store"
import {
  useSigmaGraph,
  getNodeColor,
  getNodeSize,
  type GraphData,
} from "../hooks/useSigmaGraph"

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

/**
 * Animates newly added live nodes: opacity 0→1, scale 0.5→1.2→1.0 over 300ms.
 * Listens to the Zustand liveNodes array and adds missing nodes to graphology.
 */
function LiveNodeAnimator({ graph }: { graph: Graph }) {
  const sigma = useSigma()
  const liveNodes = useAppStore((s) => s.liveNodes)
  const processedRef = useRef(new Set<string>())

  const animateNode = useCallback(
    (paperId: string, baseSize: number) => {
      if (!graph.hasNode(paperId)) return

      const DURATION = 300
      const start = performance.now()

      function tick(now: number) {
        const t = Math.min(1, (now - start) / DURATION)
        // Ease-out cubic
        const ease = 1 - Math.pow(1 - t, 3)

        // Scale: 0.5 → 1.2 at t=0.6, → 1.0 at t=1.0
        let scale: number
        if (t < 0.6) {
          scale = 0.5 + (1.2 - 0.5) * (t / 0.6)
        } else {
          scale = 1.2 + (1.0 - 1.2) * ((t - 0.6) / 0.4)
        }

        if (graph.hasNode(paperId)) {
          graph.setNodeAttribute(paperId, "size", baseSize * scale)
          // Alpha channel via color opacity (sigma renders node color directly)
          const origColor = graph.getNodeAttribute(paperId, "_liveColor") as string
          const alpha = Math.round(ease * 255)
            .toString(16)
            .padStart(2, "0")
          graph.setNodeAttribute(paperId, "color", origColor + alpha)
        }

        if (t < 1) {
          requestAnimationFrame(tick)
        } else if (graph.hasNode(paperId)) {
          // Ensure final state is clean
          graph.setNodeAttribute(paperId, "size", baseSize)
          graph.setNodeAttribute(
            paperId,
            "color",
            graph.getNodeAttribute(paperId, "_liveColor") as string,
          )
          sigma.refresh()
        }
      }

      requestAnimationFrame(tick)
    },
    [graph, sigma],
  )

  useEffect(() => {
    for (const node of liveNodes) {
      if (processedRef.current.has(node.paper_id)) continue
      processedRef.current.add(node.paper_id)

      if (graph.hasNode(node.paper_id)) continue

      // Add new node with random position near the graph center
      const nodeData = {
        paper_id: node.paper_id,
        title: node.paper_id,
        year: null,
        venue: null,
        citation_count: 0,
        cluster: null,
        source: node.source,
      }
      const color = getNodeColor(nodeData)
      const size = getNodeSize(0)

      const x = Math.random() * 20 - 10
      const y = Math.random() * 20 - 10

      graph.addNode(node.paper_id, {
        label: node.paper_id,
        x,
        y,
        size: size * 0.5,
        color: color + "00", // Start transparent
        _liveColor: color,
        year: null,
        venue: null,
        source: node.source,
        cluster: null,
        citation_count: 0,
      })

      animateNode(node.paper_id, size)
    }
  }, [liveNodes, graph, animateNode])

  return null
}

/** Toast banner for step start/end events */
function StepBanner() {
  const steps = useAppStore((s) => s.pipelineSteps)
  const running = steps.filter((s) => s.status === "running")
  const latest = running[running.length - 1]

  if (!latest) return null

  return (
    <div className="absolute top-2 left-1/2 -translate-x-1/2 z-50 bg-indigo-600 text-white text-sm px-4 py-1.5 rounded-full shadow-lg animate-pulse">
      Running: {latest.name}
    </div>
  )
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
    <div className="w-full h-full relative">
      <StepBanner />
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
        <LiveNodeAnimator graph={graph} />
      </SigmaContainer>
    </div>
  )
}
