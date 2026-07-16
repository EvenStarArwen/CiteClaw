import { useCallback, useEffect, useMemo, useState } from "react"
import { AlertTriangle, Check, Eye, Play, Save, ShieldCheck } from "lucide-react"
import yaml from "js-yaml"
import { api } from "./lib/api"
import type { Catalog, ConfigObject, Credentials, CredentialStatus, RunSnapshot } from "./lib/types"
import { BrandMark } from "./components/Brand"
import { BuildView } from "./components/BuildView"
import { RunView } from "./components/RunView"

const EMPTY_CREDENTIALS: Credentials = { s2_api_key: "", gemini_api_key: "", openai_api_key: "" }
const EMPTY_STATUS: CredentialStatus = { s2: false, gemini: false, openai: false }

function App() {
  const [mode, setMode] = useState<"build" | "run">("build")
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [configNames, setConfigNames] = useState<string[]>([])
  const [configName, setConfigName] = useState("config.yaml")
  const [configYaml, setConfigYaml] = useState("")
  const [credentials, setCredentials] = useState<Credentials>(EMPTY_CREDENTIALS)
  const [credentialStatus, setCredentialStatus] = useState<CredentialStatus>(EMPTY_STATUS)
  const [run, setRun] = useState<RunSnapshot | null>(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<{ kind: "ok" | "error"; text: string } | null>(null)

  const parsed = useMemo(() => {
    if (!configYaml) return { value: null as ConfigObject | null, error: "" }
    try {
      const value = yaml.load(configYaml)
      if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Top level must be a mapping")
      return { value: value as ConfigObject, error: "" }
    } catch (error) { return { value: null, error: error instanceof Error ? error.message : String(error) } }
  }, [configYaml])

  const loadConfig = useCallback(async (name: string) => {
    setBusy(true)
    try { const result = await api.config(name); setConfigName(name); setConfigYaml(result.yaml); setNotice(null) }
    catch (error) { setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) }) }
    finally { setBusy(false) }
  }, [])

  useEffect(() => {
    Promise.all([api.catalog(), api.configs(), api.credentialStatus()]).then(([cat, configs, status]) => {
      setCatalog(cat); setCredentialStatus(status)
      const names = configs.map((item) => item.name); setConfigNames(names)
      const initial = names.includes("config.yaml") ? "config.yaml" : names[0]
      if (initial) void loadConfig(initial)
    }).catch((error) => setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) }))
  }, [loadConfig])

  const updateConfig = useCallback((next: ConfigObject) => {
    setConfigYaml(yaml.dump(next, { noRefs: true, lineWidth: 110, sortKeys: false }))
  }, [])

  const validate = async () => {
    setBusy(true)
    try { const result = await api.validate(configYaml); setNotice({ kind: "ok", text: `Valid · ${result.summary.pipeline_steps} pipeline steps · ${result.summary.seeds} seeds` }) }
    catch (error) { setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) }) }
    finally { setBusy(false) }
  }
  const save = async () => {
    setBusy(true)
    try { await api.save(configName, configYaml); setNotice({ kind: "ok", text: `${configName} saved locally` }) }
    catch (error) { setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) }) }
    finally { setBusy(false) }
  }
  const launch = async () => {
    setBusy(true); setNotice(null)
    try { const created = await api.run(configName, configYaml, credentials); setRun(created); setMode("run") }
    catch (error) { setNotice({ kind: "error", text: error instanceof Error ? error.message : String(error) }) }
    finally { setBusy(false) }
  }

  const configuredSteps = useMemo(() => Array.isArray(parsed.value?.pipeline) ? (parsed.value!.pipeline as ConfigObject[]).map((step) => String(step.step || "Unnamed step")) : [], [parsed.value])
  const active = run?.status === "queued" || run?.status === "running"
  const progress = run?.metrics ? Math.min(100, configuredSteps.length ? (run.metrics.accepted / Math.max(1, Number(parsed.value?.max_papers_total || 1))) * 100 : 0) : 0

  return <div className="app">
    <header className="topbar">
      <div className="tb-brand"><BrandMark /><span>citeclaw</span><i>/</i><code>{mode === "build" ? configName : run?.run_id || "run"}</code></div>
      <div className="mode-toggle"><button className={mode === "build" ? "on" : ""} onClick={() => setMode("build")}>Build</button><button className={mode === "run" ? "on" : ""} disabled={!run} onClick={() => run && setMode("run")}>Run</button></div>
      <div className="tb-spacer" />
      <div className="tb-security"><ShieldCheck size={13} />loopback only</div>
      {mode === "build" ? <div className="tb-actions"><button className="btn btn-ghost" disabled={busy || !configYaml} onClick={validate}><Check size={13} />Validate</button><button className="btn" disabled={busy || !configYaml} onClick={save}><Save size={13} />Save</button><button className="btn btn-primary" disabled={busy || !!parsed.error || !configYaml || active} onClick={launch}><Play size={13} fill="currentColor" />{busy ? "Working…" : "Run pipeline"}</button></div> : <div className="tb-actions"><button className="btn" onClick={() => setMode("build")}><Eye size={13} />Configuration</button>{run && <span className={`top-status ${run.status}`}>{run.status}</span>}</div>}
    </header>

    {mode === "build" ? <BuildView configNames={configNames} configName={configName} config={parsed.value} configYaml={configYaml} parseError={parsed.error} catalog={catalog} credentials={credentials} credentialStatus={credentialStatus} onChooseConfig={loadConfig} onConfigChange={updateConfig} onYamlChange={setConfigYaml} onCredentialsChange={setCredentials} /> : run ? <RunView run={run} configuredSteps={configuredSteps} onRunUpdate={setRun} /> : null}

    <footer className="statusbar"><span className={`sb-dot ${active ? "run" : "idle"}`} /><span>{active ? "running" : "ready"}</span><i>·</i><span>{parsed.value ? `${configuredSteps.length} steps · ${Array.isArray(parsed.value.seed_papers) ? parsed.value.seed_papers.length : 0} seeds` : "loading config"}</span>{run && <><i>·</i><span>{run.metrics.accepted.toLocaleString()} accepted</span><div className="sb-progress"><u style={{ width: `${progress}%` }} /></div></>}<div className="tb-spacer" /><span>local session · keys not persisted</span></footer>
    {notice && <div className={`notice ${notice.kind}`}>{notice.kind === "error" ? <AlertTriangle size={15} /> : <Check size={15} />}<span>{notice.text}</span><button onClick={() => setNotice(null)}>×</button></div>}
  </div>
}

export default App
