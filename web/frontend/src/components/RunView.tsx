import { useEffect, useMemo, useRef, useState } from "react"
import { Check, Circle, LoaderCircle, Radio, Search } from "lucide-react"
import { api } from "../lib/api"
import type { GraphPaper, GraphPayload, HitlPaper, Metrics, RunSnapshot, StepState } from "../lib/types"
import { GraphView } from "./GraphView"

const EMPTY_METRICS: Metrics = { accepted: 0, rejected: 0, seen: 0, llm_tokens: 0, llm_input_tokens: 0, llm_output_tokens: 0, llm_reasoning_tokens: 0, llm_calls: 0, cost_usd: 0, s2_requests: 0, s2_cache_hits: 0, rejection_counts: {}, elapsed_sec: 0 }

function formatCount(value: number) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(value >= 100_000 ? 0 : 1)}K`
  return value.toLocaleString()
}

function formatTime(seconds: number) {
  const h = Math.floor(seconds / 3600); const m = Math.floor((seconds % 3600) / 60); const s = seconds % 60
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
}

function HitlModal({ runId, papers, onDone }: { runId: string; papers: HitlPaper[]; onDone: () => void }) {
  const [labels, setLabels] = useState<Record<string, boolean>>({})
  const [busy, setBusy] = useState(false)
  return <div className="modal-backdrop"><div className="hitl-modal"><div className="eyebrow">HUMAN REVIEW · {papers.length} PAPERS</div><h2>Check screening decisions</h2><p>Mark each sampled paper keep or reject, then resume the pipeline.</p><div className="hitl-list">{papers.map((paper) => <div className="hitl-paper" key={paper.paper_id}><strong>{paper.title}</strong><span>{paper.year || "—"} · {paper.venue || "venue unavailable"}</span><p>{paper.abstract || "Abstract unavailable."}</p><div><button className={labels[paper.paper_id] === true ? "on keep" : ""} onClick={() => setLabels((l) => ({ ...l, [paper.paper_id]: true }))}>Keep</button><button className={labels[paper.paper_id] === false ? "on reject" : ""} onClick={() => setLabels((l) => ({ ...l, [paper.paper_id]: false }))}>Reject</button></div></div>)}</div><div className="modal-actions"><button className="btn" onClick={async () => { setBusy(true); await api.hitl(runId, labels, true); onDone() }}>Stop after review</button><button className="btn btn-primary" disabled={busy || Object.keys(labels).length !== papers.length} onClick={async () => { setBusy(true); await api.hitl(runId, labels); onDone() }}>{busy ? "Submitting…" : "Submit & resume"}</button></div></div></div>
}

export function RunView({ run, configuredSteps, onRunUpdate }: { run: RunSnapshot; configuredSteps: string[]; onRunUpdate: (run: RunSnapshot) => void }) {
  const [status, setStatus] = useState(run.status)
  const [error, setError] = useState<string | null>(run.error)
  const [metrics, setMetrics] = useState<Metrics>(run.metrics || EMPTY_METRICS)
  const [graph, setGraph] = useState<GraphPayload>({ nodes: [], edges: [] })
  const [steps, setSteps] = useState<StepState[]>(() => configuredSteps.map((name, i) => ({ idx: i + 1, name, status: "idle" })))
  const [logs, setLogs] = useState<Array<{ level: string; message: string }>>([])
  const [shape, setShape] = useState("")
  const [tab, setTab] = useState<"overview" | "rejections" | "logs" | "shape">("overview")
  const [paperQuery, setPaperQuery] = useState("")
  const [hitl, setHitl] = useState<HitlPaper[] | null>(null)
  const streamState = useRef({ runId: "", seq: -1 })

  useEffect(() => {
    if (streamState.current.runId !== run.run_id) streamState.current = { runId: run.run_id, seq: -1 }
    const protocol = location.protocol === "https:" ? "wss:" : "ws:"
    const ws = new WebSocket(`${protocol}//${location.host}/api/runs/${run.run_id}/stream`)
    ws.onmessage = (message) => {
      const event = JSON.parse(String(message.data)) as Record<string, unknown>
      const seq = Number(event.seq ?? -1)
      if (seq <= streamState.current.seq) return
      streamState.current.seq = seq
      const type = String(event.type)
      if (type === "step_start") {
        const incoming: StepState = { idx: Number(event.idx), name: String(event.name), description: String(event.description || ""), status: "running" }
        setSteps((current) => [...current.filter((s) => s.idx !== incoming.idx), incoming].sort((a, b) => a.idx - b.idx))
      } else if (type === "step_end") {
        setSteps((current) => current.map((s) => s.idx === Number(event.idx) ? { ...s, status: "done", in_count: Number(event.in_count), out_count: Number(event.out_count), delta_collection: Number(event.delta_collection), stats: event.stats as Record<string, unknown> } : s))
      } else if (type === "metrics") {
        setMetrics(event.metrics as Metrics)
      } else if (type === "graph_snapshot") {
        setGraph(event.graph as GraphPayload)
      } else if (type === "paper_added" && event.paper) {
        const paper = event.paper as GraphPaper
        setGraph((current) => current.nodes.some((p) => p.paper_id === paper.paper_id) ? current : { ...current, nodes: [...current.nodes, paper] })
      } else if (type === "log") {
        setLogs((current) => [...current.slice(-499), { level: String(event.level), message: String(event.message) }])
      } else if (type === "shape_table_update") {
        setShape(String(event.rendered_shape || ""))
      } else if (type === "hitl_request") {
        setHitl(event.papers as HitlPaper[])
      } else if (type === "run_started") {
        setStatus("running")
      } else if (type === "run_complete") {
        const finalStatus = event.status as RunSnapshot["status"]
        const finalMetrics = event.metrics as Metrics
        setStatus(finalStatus); setError((event.error as string) || null); setMetrics(finalMetrics); setGraph(event.graph as GraphPayload)
        onRunUpdate({ ...run, status: finalStatus, error: (event.error as string) || null, metrics: finalMetrics })
      }
    }
    ws.onerror = () => setLogs((current) => [...current, { level: "WARNING", message: "Live stream connection interrupted." }])
    return () => ws.close()
  }, [onRunUpdate, run])

  const papers = useMemo(() => graph.nodes.filter((paper) => !paperQuery || paper.title.toLowerCase().includes(paperQuery.toLowerCase())).sort((a, b) => b.citation_count - a.citation_count), [graph.nodes, paperQuery])
  const rejectionRows = Object.entries(metrics.rejection_counts).sort((a, b) => b[1] - a[1])
  const rejectMax = rejectionRows[0]?.[1] || 1
  const cacheTotal = metrics.s2_requests + metrics.s2_cache_hits
  const cacheRate = cacheTotal ? metrics.s2_cache_hits / cacheTotal * 100 : 0

  return <>
    <aside className="panel panel-left">
      <div className="ph"><span className="ph-title">Pipeline progress</span><span className={`run-pill ${status}`}>{status}</span></div>
      <div className="pb-scroll progress-list">{steps.map((step) => <div className={`prog-step is-${step.status}`} key={step.idx}>
        <span className="prog-badge">{step.status === "done" ? <Check size={11} /> : step.status === "running" ? <LoaderCircle className="spin" size={11} /> : <Circle size={9} />}</span>
        <div><strong>{step.name}</strong><span>{step.status === "done" ? `${step.in_count?.toLocaleString()} → ${step.out_count?.toLocaleString()} · Δ${step.delta_collection}` : step.description || "pending"}</span></div>
      </div>)}</div>
      <div className="prog-summary"><div><span>Elapsed</span><b>{formatTime(metrics.elapsed_sec)}</b></div><div><span>Output</span><b title={run.output_dir}>{run.run_id}</b></div>{error && <div className="run-error">{error}</div>}</div>
    </aside>

    <main className="main run-main"><section className="pane pane-top"><GraphView graph={graph} /></section><section className="dashboard">
      <div className="dash-head"><span className="dash-head-title">Dashboard · {steps.find((s) => s.status === "running")?.name || status}</span><div className="dash-tabs">{(["overview", "rejections", "logs", "shape"] as const).map((name) => <button className={tab === name ? "on" : ""} onClick={() => setTab(name)} key={name}>{name}</button>)}</div></div>
      <div className="dash-metrics"><div><span>Accepted</span><b>{formatCount(metrics.accepted)}</b><small>{metrics.elapsed_sec ? `${(metrics.accepted / metrics.elapsed_sec * 60).toFixed(1)} / min` : "awaiting papers"}</small></div><div><span>Rejected</span><b>{formatCount(metrics.rejected)}</b><small>{rejectionRows.length} reasons</small></div><div><span>LLM tokens</span><b>{formatCount(metrics.llm_tokens)}</b><small>{formatCount(metrics.llm_input_tokens)} in · {formatCount(metrics.llm_output_tokens)} out</small></div><div><span>LLM calls</span><b>{formatCount(metrics.llm_calls)}</b><small>{formatCount(metrics.llm_reasoning_tokens)} reasoning</small></div><div><span>Cost</span><b>${metrics.cost_usd.toFixed(4)}</b><small>live estimate</small></div><div><span>S2 cache</span><b>{cacheRate.toFixed(0)}%</b><small>{formatCount(metrics.s2_cache_hits)} / {formatCount(cacheTotal)} requests</small></div></div>
      <div className="dash-content">
        {tab === "overview" && <div className="overview-grid"><div><span className="eyebrow">RUN STATE</span><strong>{status === "running" ? "Pipeline is processing" : status === "completed" ? "Artifacts written" : status === "failed" ? "Run stopped with an error" : "Queued"}</strong><p>{metrics.seen.toLocaleString()} unique papers observed. Graph snapshots refresh after each completed step; token and request counters refresh every second.</p></div><div><span className="eyebrow">CURRENT OUTPUT</span><strong>{graph.nodes.length.toLocaleString()} nodes · {graph.edges.length.toLocaleString()} edges</strong><p>{run.output_dir}</p></div></div>}
        {tab === "rejections" && <div className="bar-list">{rejectionRows.length ? rejectionRows.map(([reason, count]) => <div key={reason}><span><code>{reason}</code><b>{count.toLocaleString()}</b></span><i><u style={{ width: `${count / rejectMax * 100}%` }} /></i></div>) : <div className="empty-copy">No rejection reasons recorded yet.</div>}</div>}
        {tab === "logs" && <div className="log-view">{logs.length ? logs.map((line, index) => <div key={index}><span className={`log-${line.level.toLowerCase()}`}>{line.level}</span>{line.message}</div>) : <div className="empty-copy">No log records yet.</div>}</div>}
        {tab === "shape" && <pre className="shape-view">{shape || "Shape summary appears when the pipeline completes."}</pre>}
      </div>
    </section></main>

    <aside className="panel panel-right"><div className="ph"><span className="ph-title">Accepted papers</span><span className="ph-count">{graph.nodes.length.toLocaleString()}</span></div><div className="accepted-search"><Search size={13} /><input placeholder="Filter accepted papers" value={paperQuery} onChange={(e) => setPaperQuery(e.target.value)} /></div><div className="pb-scroll accepted-list">{papers.slice(0, 500).map((paper) => <div className="acc-item" key={paper.paper_id}><b>{paper.citation_count.toLocaleString()}</b><div><strong>{paper.title}</strong><span>{paper.year || "—"} · {paper.venue || paper.source}</span></div><i>d{paper.depth}</i></div>)}{!papers.length && <div className="empty-copy">Accepted stream is empty.</div>}</div><div className="stream-foot"><Radio size={11} className={status === "running" ? "pulse" : ""} />{status === "running" ? "live stream" : `${status} · ${graph.nodes.length} papers`}</div></aside>
    {hitl && <HitlModal runId={run.run_id} papers={hitl} onDone={() => setHitl(null)} />}
  </>
}
