import { useCallback, useEffect, useMemo, useState } from "react"
import yaml from "js-yaml"
import { api } from "./lib/api"
import type {
  Catalog,
  ConfigObject,
  Credentials,
  CredentialStatus,
  GraphPaper,
  GraphPayload,
  HitlPaper,
  Metrics,
  RunSnapshot,
  StepState,
} from "./lib/types"
import {
  BottomBar,
  BuildBlocks,
  BuildConfig,
  BuildPipeline,
  BuildSeeds,
  CredentialsModal,
  HitlModal,
  Notice,
  RunAccepted,
  RunDashboard,
  RunNetwork,
  RunProgress,
  TopBar,
  pipelineFromConfig,
  type BlockDefinition,
} from "./components/ClaudeUI"

const EMPTY_CREDENTIALS: Credentials = { s2_api_key: "", gemini_api_key: "", openai_api_key: "" }
const EMPTY_CREDENTIAL_STATUS: CredentialStatus = { s2: false, gemini: false, openai: false }
const EMPTY_METRICS: Metrics = {
  accepted: 0,
  rejected: 0,
  seen: 0,
  llm_tokens: 0,
  llm_input_tokens: 0,
  llm_output_tokens: 0,
  llm_reasoning_tokens: 0,
  llm_calls: 0,
  cost_usd: 0,
  s2_requests: 0,
  s2_cache_hits: 0,
  rejection_counts: {},
  elapsed_sec: 0,
}

function App() {
  const [mode, setMode] = useState<"build" | "run">("build")
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [configNames, setConfigNames] = useState<string[]>([])
  const [configName, setConfigName] = useState("config.yaml")
  const [configYaml, setConfigYaml] = useState("")
  const [credentials, setCredentials] = useState<Credentials>(EMPTY_CREDENTIALS)
  const [credentialStatus, setCredentialStatus] = useState<CredentialStatus>(EMPTY_CREDENTIAL_STATUS)
  const [credentialsOpen, setCredentialsOpen] = useState(false)
  const [selectedStep, setSelectedStep] = useState(0)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<{ kind: "ok" | "error"; text: string } | null>(null)

  const [run, setRun] = useState<RunSnapshot | null>(null)
  const [metrics, setMetrics] = useState<Metrics>(EMPTY_METRICS)
  const [graph, setGraph] = useState<GraphPayload>({ nodes: [], edges: [] })
  const [steps, setSteps] = useState<StepState[]>([])
  const [logs, setLogs] = useState<Array<{ level: string; message: string }>>([])
  const [shape, setShape] = useState("")
  const [hitl, setHitl] = useState<HitlPaper[] | null>(null)
  const [freshIds, setFreshIds] = useState<Set<string>>(new Set())

  const parsed = useMemo(() => {
    if (!configYaml) return { value: null as ConfigObject | null, error: "" }
    try {
      const value = yaml.load(configYaml)
      if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Top level must be a mapping")
      return { value: value as ConfigObject, error: "" }
    } catch (error) {
      return { value: null, error: error instanceof Error ? error.message : String(error) }
    }
  }, [configYaml])

  const pipeline = useMemo(() => pipelineFromConfig(parsed.value), [parsed.value])
  const seeds = useMemo(() => {
    const raw = parsed.value?.seed_papers
    return Array.isArray(raw) ? raw.filter((seed): seed is ConfigObject => !!seed && typeof seed === "object") : []
  }, [parsed.value])
  const filterCount = useMemo(() => pipeline.reduce((total, node) => total + node.filters, 0), [pipeline])
  const selectedNode = pipeline[selectedStep] || null
  const active = run?.status === "queued" || run?.status === "running"

  const writeConfig = useCallback((next: ConfigObject) => {
    setConfigYaml(yaml.dump(next, { noRefs: true, lineWidth: 110, sortKeys: false }))
  }, [])

  const loadConfig = useCallback(async (name: string) => {
    setBusy(true)
    try {
      const result = await api.config(name)
      setConfigName(name)
      setConfigYaml(result.yaml)
      setSelectedStep(0)
      setNotice(null)
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) })
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    Promise.all([api.catalog(), api.configs(), api.credentialStatus()])
      .then(([nextCatalog, configs, status]) => {
        setCatalog(nextCatalog)
        setCredentialStatus(status)
        const names = configs.map((config) => config.name)
        setConfigNames(names)
        const initial = names.includes("config.yaml") ? "config.yaml" : names[0]
        if (initial) void loadConfig(initial)
      })
      .catch((error) => setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) }))
  }, [loadConfig])

  useEffect(() => {
    if (!run?.run_id) return
    const protocol = location.protocol === "https:" ? "wss:" : "ws:"
    const socket = new WebSocket(`${protocol}//${location.host}/api/runs/${run.run_id}/stream`)
    let lastSequence = -1
    socket.onmessage = (message) => {
      const event = JSON.parse(String(message.data)) as Record<string, unknown>
      const sequence = Number(event.seq ?? -1)
      if (sequence <= lastSequence) return
      lastSequence = sequence
      const type = String(event.type)
      if (type === "run_started") {
        setRun((current) => current ? { ...current, status: "running" } : current)
      } else if (type === "step_start") {
        const incoming: StepState = {
          idx: Number(event.idx),
          name: String(event.name),
          description: String(event.description || ""),
          status: "running",
        }
        setSteps((current) => [...current.filter((step) => step.idx !== incoming.idx), incoming].sort((a, b) => a.idx - b.idx))
      } else if (type === "step_end") {
        setSteps((current) => current.map((step) => step.idx === Number(event.idx) ? {
          ...step,
          status: "done",
          in_count: Number(event.in_count),
          out_count: Number(event.out_count),
          delta_collection: Number(event.delta_collection),
          stats: event.stats as Record<string, unknown>,
        } : step))
      } else if (type === "metrics") {
        setMetrics(event.metrics as Metrics)
      } else if (type === "graph_snapshot") {
        setGraph(event.graph as GraphPayload)
      } else if (type === "paper_added" && event.paper) {
        const paper = event.paper as GraphPaper
        setGraph((current) => current.nodes.some((node) => node.paper_id === paper.paper_id)
          ? current
          : { ...current, nodes: [...current.nodes, paper] })
        setFreshIds((current) => new Set(current).add(paper.paper_id))
        window.setTimeout(() => setFreshIds((current) => {
          const next = new Set(current)
          next.delete(paper.paper_id)
          return next
        }), 1800)
      } else if (type === "log") {
        setLogs((current) => [...current.slice(-499), { level: String(event.level), message: String(event.message) }])
      } else if (type === "shape_table_update") {
        setShape(String(event.rendered_shape || ""))
      } else if (type === "hitl_request") {
        setHitl(event.papers as HitlPaper[])
      } else if (type === "run_complete") {
        const status = event.status as RunSnapshot["status"]
        const finalMetrics = event.metrics as Metrics
        setMetrics(finalMetrics)
        setGraph(event.graph as GraphPayload)
        setRun((current) => current ? {
          ...current,
          status,
          error: (event.error as string) || null,
          metrics: finalMetrics,
        } : current)
      }
    }
    socket.onerror = () => setLogs((current) => [...current, {
      level: "WARNING",
      message: "Live stream connection interrupted.",
    }])
    return () => socket.close()
  }, [run?.run_id])

  const updateSeeds = (nextSeeds: ConfigObject[]) => {
    if (!parsed.value) return
    const next = structuredClone(parsed.value)
    next.seed_papers = nextSeeds
    writeConfig(next)
  }

  const updateStep = (index: number, key: string, value: unknown) => {
    if (!parsed.value) return
    const next = structuredClone(parsed.value)
    const nextPipeline = Array.isArray(next.pipeline) ? [...next.pipeline] as ConfigObject[] : []
    nextPipeline[index] = { ...nextPipeline[index], [key]: value }
    next.pipeline = nextPipeline
    writeConfig(next)
  }

  const duplicateStep = (index: number) => {
    if (!parsed.value) return
    const next = structuredClone(parsed.value)
    const nextPipeline = Array.isArray(next.pipeline) ? [...next.pipeline] as ConfigObject[] : []
    nextPipeline.splice(index + 1, 0, structuredClone(nextPipeline[index]))
    next.pipeline = nextPipeline
    writeConfig(next)
    setSelectedStep(index + 1)
  }

  const deleteStep = (index: number) => {
    if (!parsed.value) return
    const next = structuredClone(parsed.value)
    const nextPipeline = Array.isArray(next.pipeline) ? [...next.pipeline] as ConfigObject[] : []
    nextPipeline.splice(index, 1)
    next.pipeline = nextPipeline
    writeConfig(next)
    setSelectedStep(Math.max(0, Math.min(index, nextPipeline.length - 1)))
  }

  const addBlock = (definition: BlockDefinition) => {
    if (!parsed.value) return
    const next = structuredClone(parsed.value)
    const nextPipeline = Array.isArray(next.pipeline) ? [...next.pipeline] as ConfigObject[] : []
    nextPipeline.push(structuredClone(definition.defaultStep))
    next.pipeline = nextPipeline
    writeConfig(next)
    setSelectedStep(nextPipeline.length - 1)
  }

  const updateRoot = (key: string, value: unknown) => {
    if (!parsed.value) return
    const next = structuredClone(parsed.value)
    next[key] = value
    writeConfig(next)
  }

  const validate = async () => {
    setBusy(true)
    try {
      const result = await api.validate(configYaml)
      setNotice({
        kind: "ok",
        text: `Valid · ${String(result.summary.pipeline_steps)} pipeline steps · ${String(result.summary.seeds)} seeds`,
      })
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) })
    } finally {
      setBusy(false)
    }
  }

  const save = async () => {
    setBusy(true)
    try {
      await api.save(configName, configYaml)
      setNotice({ kind: "ok", text: `${configName} saved locally` })
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) })
    } finally {
      setBusy(false)
    }
  }

  const launch = async () => {
    setBusy(true)
    setNotice(null)
    setMetrics(EMPTY_METRICS)
    setGraph({ nodes: [], edges: [] })
    setLogs([])
    setShape("")
    setSteps(pipeline.map((node, index) => ({ idx: index + 1, name: node.step, status: "idle" })))
    try {
      const created = await api.run(configName, configYaml, credentials)
      setRun(created)
      setMode("run")
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      setNotice({ kind: "error", text })
      if (text.toLowerCase().includes("api key")) setCredentialsOpen(true)
    } finally {
      setBusy(false)
    }
  }

  return <div className="app">
    <TopBar
      mode={mode}
      setMode={setMode}
      configName={configName}
      configNames={configNames}
      onChooseConfig={loadConfig}
      pipelineCount={pipeline.length}
      filterCount={filterCount}
      metrics={metrics}
      run={run}
      busy={busy}
      canRun={!!configYaml && !parsed.error && !active}
      onCredentials={() => setCredentialsOpen(true)}
      onValidate={() => void validate()}
      onSave={() => void save()}
      onReset={() => void loadConfig(configName)}
      onRun={() => void launch()}
    />

    {mode === "build" ? <>
      <BuildSeeds seeds={seeds} onChange={updateSeeds} />
      <div className="main">
        <BuildPipeline pipeline={pipeline} selectedIndex={selectedStep} onSelect={setSelectedStep} />
        <BuildConfig
          node={selectedNode}
          configYaml={configYaml}
          parseError={parsed.error}
          onYamlChange={setConfigYaml}
          onUpdateField={(key, value) => updateStep(selectedStep, key, value)}
          onDuplicate={() => duplicateStep(selectedStep)}
          onDelete={() => deleteStep(selectedStep)}
        />
      </div>
      <BuildBlocks onAdd={addBlock} />
    </> : <>
      <RunProgress steps={steps} run={run} metrics={metrics} />
      <div className="main split-60-40">
        <RunNetwork graph={graph} metrics={metrics} />
        <RunDashboard metrics={metrics} steps={steps} run={run} logs={logs} shape={shape} />
      </div>
      <RunAccepted papers={graph.nodes} freshIds={freshIds} running={active} />
    </>}

    <BottomBar
      mode={mode}
      running={active}
      configName={configName}
      pipelineLen={pipeline.length}
      seedCount={seeds.length}
      steps={steps}
      metrics={metrics}
      run={run}
    />

    {credentialsOpen && catalog && <CredentialsModal
      catalog={catalog}
      credentials={credentials}
      credentialStatus={credentialStatus}
      selectedModel={String(parsed.value?.screening_model || "")}
      reasoningEffort={String(parsed.value?.reasoning_effort || "minimal")}
      onCredentialsChange={setCredentials}
      onModelChange={(value) => updateRoot("screening_model", value)}
      onReasoningChange={(value) => updateRoot("reasoning_effort", value)}
      onClose={() => setCredentialsOpen(false)}
    />}
    {hitl && run && <HitlModal runId={run.run_id} papers={hitl} onDone={() => setHitl(null)} />}
    {notice && <Notice notice={notice} onClose={() => setNotice(null)} />}
  </div>
}

export default App
