import { useEffect, useMemo, useState } from "react"
import { Search, X, ZoomIn, ZoomOut, Maximize2 } from "lucide-react"
import { SigmaContainer, useLoadGraph, useRegisterEvents, useSigma } from "@react-sigma/core"
import { MultiDirectedGraph } from "graphology"
import forceAtlas2 from "graphology-layout-forceatlas2"
import type { GraphPaper, GraphPayload } from "../lib/types"

const YEAR_COLORS = ["#adb5bd", "#97a0a9", "#818b95", "#6b7681", "#55606d", "#414b58", "#2d3744", "#1a2330"]

function yearColor(year: number | null) {
  if (!year) return YEAR_COLORS[0]
  return YEAR_COLORS[Math.max(0, Math.min(7, year - 2018))]
}

function hash(value: string) {
  let out = 2166136261
  for (let i = 0; i < value.length; i += 1) out = Math.imul(out ^ value.charCodeAt(i), 16777619)
  return out >>> 0
}

function GraphLoader({ graph, onSelect }: { graph: GraphPayload; onSelect: (paper: GraphPaper | null) => void }) {
  const loadGraph = useLoadGraph()
  const registerEvents = useRegisterEvents()

  useEffect(() => {
    const next = new MultiDirectedGraph()
    graph.nodes.forEach((paper, index) => {
      const angle = ((hash(paper.paper_id) % 360) / 180) * Math.PI
      const radius = 1 + (index % 17) / 9
      next.addNode(paper.paper_id, {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        label: paper.title,
        color: paper.seed ? "#e63946" : yearColor(paper.year),
        size: paper.seed ? 9 : Math.max(3, Math.min(10, 2.5 + Math.log10(paper.citation_count + 1) * 1.6)),
        paper,
      })
    })
    graph.edges.forEach((edge, index) => {
      if (!next.hasNode(edge.source) || !next.hasNode(edge.target)) return
      next.addEdgeWithKey(`e${index}`, edge.source, edge.target, {
        color: yearColor((next.getNodeAttribute(edge.target, "paper") as GraphPaper).year),
        size: 0.7,
      })
    })
    if (next.order > 1) {
      const iterations = next.order < 250 ? 80 : next.order < 900 ? 45 : 24
      forceAtlas2.assign(next, { iterations, settings: forceAtlas2.inferSettings(next) })
    }
    loadGraph(next)
  }, [graph, loadGraph])

  useEffect(() => registerEvents({
    clickNode: ({ node }) => onSelect((node && (graph.nodes.find((p) => p.paper_id === node))) || null),
    clickStage: () => onSelect(null),
  }), [graph.nodes, onSelect, registerEvents])
  return null
}

function CameraControls() {
  const sigma = useSigma()
  return (
    <div className="graph-controls">
      <button title="Zoom in" onClick={() => sigma.getCamera().animatedZoom()}><ZoomIn size={14} /></button>
      <button title="Zoom out" onClick={() => sigma.getCamera().animatedUnzoom()}><ZoomOut size={14} /></button>
      <button title="Reset camera" onClick={() => sigma.getCamera().animatedReset()}><Maximize2 size={14} /></button>
    </div>
  )
}

export function GraphView({ graph }: { graph: GraphPayload }) {
  const [selected, setSelected] = useState<GraphPaper | null>(null)
  const [query, setQuery] = useState("")
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return []
    return graph.nodes.filter((p) => p.title.toLowerCase().includes(q) || p.paper_id.toLowerCase().includes(q)).slice(0, 6)
  }, [graph.nodes, query])

  if (!graph.nodes.length) {
    return <div className="graph-empty"><div className="graph-empty-orbit" /><strong>Graph awaiting papers</strong><span>Nodes appear after each completed pipeline step.</span></div>
  }
  return (
    <div className="network">
      <SigmaContainer className="sigma" settings={{ renderEdgeLabels: false, labelRenderedSizeThreshold: 10, defaultEdgeColor: "#cfcabb" }}>
        <GraphLoader graph={graph} onSelect={setSelected} />
        <CameraControls />
      </SigmaContainer>
      <div className="graph-search">
        <Search size={13} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Find paper" />
        {!!query && <button onClick={() => setQuery("")}><X size={12} /></button>}
        {!!matches.length && <div className="graph-search-results">{matches.map((paper) => <button key={paper.paper_id} onClick={() => { setSelected(paper); setQuery("") }}>{paper.title}</button>)}</div>}
      </div>
      <div className="net-legend">
        <span><i className="legend-seed" />Seed</span>
        <span><i className="legend-ramp">{YEAR_COLORS.map((color) => <b key={color} style={{ background: color }} />)}</i>2018 → 2025</span>
        <span>size · citations</span>
      </div>
      <div className="net-counter"><b>{graph.nodes.length.toLocaleString()}</b> nodes <i>·</i> <b>{graph.edges.length.toLocaleString()}</b> edges</div>
      {selected && <aside className="paper-drawer">
        <button className="drawer-close" onClick={() => setSelected(null)}><X size={15} /></button>
        <div className="eyebrow">{selected.source} · depth {selected.depth}</div>
        <h2>{selected.title}</h2>
        <div className="drawer-meta">{selected.authors.slice(0, 4).join(", ") || "Authors unavailable"}</div>
        <div className="drawer-stats"><span><b>{selected.year || "—"}</b>year</span><span><b>{selected.citation_count.toLocaleString()}</b>citations</span><span><b>{selected.venue || "—"}</b>venue</span></div>
        <p>{selected.abstract || "Abstract unavailable."}</p>
        <code>{selected.paper_id}</code>
      </aside>}
    </div>
  )
}
