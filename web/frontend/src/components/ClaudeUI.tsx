import { useEffect, useMemo, useState, type ComponentType, type CSSProperties, type ReactNode } from "react"
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Copy,
  Database,
  DollarSign,
  Download,
  FileCheck2,
  Filter,
  Flag,
  GitBranch,
  Inbox,
  KeyRound,
  LoaderCircle,
  Maximize2,
  Minus,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Save,
  Search,
  SlidersHorizontal,
  Square,
  Star,
  Trash2,
  X,
  type LucideProps,
} from "lucide-react"
import { MultiDirectedGraph } from "graphology"
import forceAtlas2 from "graphology-layout-forceatlas2"
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
} from "../lib/types"
import { api } from "../lib/api"

const ICONS: Record<string, ComponentType<LucideProps>> = {
  "arrow-left": ArrowLeft,
  "arrow-right": ArrowRight,
  check: Check,
  copy: Copy,
  database: Database,
  "dollar-sign": DollarSign,
  download: Download,
  "file-check": FileCheck2,
  filter: Filter,
  flag: Flag,
  "git-branch": GitBranch,
  inbox: Inbox,
  key: KeyRound,
  loader: LoaderCircle,
  "maximize-2": Maximize2,
  minus: Minus,
  pause: Pause,
  play: Play,
  plus: Plus,
  "rotate-ccw": RotateCcw,
  save: Save,
  search: Search,
  "sliders-horizontal": SlidersHorizontal,
  square: Square,
  star: Star,
  "trash-2": Trash2,
  x: X,
}

function Icon({ name, size = 14, className = "", style }: { name: string; size?: number; className?: string; style?: CSSProperties }) {
  const Glyph = ICONS[name] || Square
  return <span className={`lc ${className}`} style={{ width: size, height: size, ...style }}><Glyph size={size} strokeWidth={1.75} /></span>
}

function formatCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(value >= 100_000 ? 0 : 1)}K`
  return value.toLocaleString()
}

function formatTime(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remainder = seconds % 60
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
}

const STEP_META: Record<string, { kind: string; name: string; prefix: string; icon: string }> = {
  LoadSeeds: { kind: "seed", name: "Seed set", prefix: "SED", icon: "flag" },
  ResolveSeeds: { kind: "seed", name: "Resolve seeds", prefix: "RES", icon: "search" },
  ExpandForward: { kind: "fwd", name: "Forward screener", prefix: "FWD", icon: "arrow-right" },
  ExpandBackward: { kind: "bwd", name: "Backward screener", prefix: "BWD", icon: "arrow-left" },
  ExpandBySearch: { kind: "fwd", name: "Search expansion", prefix: "SEA", icon: "search" },
  ExpandBySemantics: { kind: "fwd", name: "Semantic expansion", prefix: "SEM", icon: "arrow-right" },
  ExpandByAuthor: { kind: "fwd", name: "Author expansion", prefix: "AUT", icon: "arrow-right" },
  ExpandByPDF: { kind: "fwd", name: "PDF expansion", prefix: "PDF", icon: "arrow-right" },
  Rerank: { kind: "rerank", name: "Diversified rerank", prefix: "RRK", icon: "sliders-horizontal" },
  ReScreen: { kind: "rsc", name: "Rescreener", prefix: "RSC", icon: "filter" },
  Cluster: { kind: "rerank", name: "Cluster", prefix: "CLU", icon: "sliders-horizontal" },
  MergeDuplicates: { kind: "rerank", name: "Merge duplicates", prefix: "MRG", icon: "copy" },
  HumanInTheLoop: { kind: "rsc", name: "Human review", prefix: "HIT", icon: "file-check" },
  Parallel: { kind: "fwd", name: "Parallel", prefix: "PAR", icon: "git-branch" },
  Finalize: { kind: "sink", name: "Accepted sink", prefix: "OUT", icon: "inbox" },
}

export interface PipelineNode {
  id: string
  step: string
  kind: string
  name: string
  localId: string
  icon: string
  filters: number
  raw: ConfigObject
}

export function pipelineFromConfig(config: ConfigObject | null): PipelineNode[] {
  const rawPipeline = config?.pipeline
  if (!Array.isArray(rawPipeline)) return []
  return rawPipeline.filter((step): step is ConfigObject => !!step && typeof step === "object").map((raw, index) => {
    const step = String(raw.step || "Step")
    const meta = STEP_META[step] || { kind: "rsc", name: step, prefix: "STP", icon: "square" }
    const filters = ["screener", "filter", "blocks", "routes"].reduce((count, key) => count + (key in raw ? 1 : 0), 0)
    return {
      id: `step-${index}`,
      step,
      kind: meta.kind,
      name: meta.name,
      localId: `${meta.prefix}-${String(index + 1).padStart(2, "0")}`,
      icon: meta.icon,
      filters,
      raw,
    }
  })
}

export interface BlockDefinition {
  cat: string
  kind: string
  name: string
  hint: string
  icon: string
  defaultStep: ConfigObject
}

const BLOCKS: BlockDefinition[] = [
  { cat: "Sources", kind: "seed", name: "Seed set", hint: "Load configured papers", icon: "flag", defaultStep: { step: "LoadSeeds" } },
  { cat: "Expanders", kind: "fwd", name: "Forward screener", hint: "Follows incoming citations", icon: "arrow-right", defaultStep: { step: "ExpandForward", screener: "default" } },
  { cat: "Expanders", kind: "bwd", name: "Backward screener", hint: "Walks references", icon: "arrow-left", defaultStep: { step: "ExpandBackward", screener: "default" } },
  { cat: "Rerankers", kind: "rerank", name: "Diversified rerank", hint: "Top-K diversity control", icon: "sliders-horizontal", defaultStep: { step: "Rerank", top_k: 500 } },
  { cat: "Rerankers", kind: "rsc", name: "Rescreener", hint: "Re-screen collection", icon: "filter", defaultStep: { step: "ReScreen", screener: "default" } },
  { cat: "Sinks", kind: "sink", name: "Accepted sink", hint: "Final export target", icon: "inbox", defaultStep: { step: "Finalize" } },
]

export function TopBar(props: {
  mode: "build" | "run"
  setMode: (mode: "build" | "run") => void
  configName: string
  configNames: string[]
  onChooseConfig: (name: string) => void
  pipelineCount: number
  filterCount: number
  metrics: Metrics
  run: RunSnapshot | null
  busy: boolean
  canRun: boolean
  onCredentials: () => void
  onValidate: () => void
  onSave: () => void
  onReset: () => void
  onRun: () => void
}) {
  const running = props.run?.status === "queued" || props.run?.status === "running"
  return <header className="topbar">
    <div className="tb-brand">
      <svg width="20" height="20" viewBox="0 0 48 46" fill="none" aria-hidden="true"><path fill="#863bff" d="M25.946 44.938c-.664.845-2.021.375-2.021-.698V33.937a2.26 2.26 0 0 0-2.262-2.262H10.287c-.92 0-1.456-1.04-.92-1.788l7.48-10.471c1.07-1.497 0-3.578-1.842-3.578H1.237c-.92 0-1.456-1.04-.92-1.788L10.013.474c.214-.297.556-.474.92-.474h28.894c.92 0 1.456 1.04.92 1.788l-7.48 10.471c-1.07 1.498 0 3.579 1.842 3.579h11.377c.943 0 1.473 1.088.89 1.83L25.947 44.94z" /></svg>
      <span className="tb-brand-name">CiteClaw</span><span className="tb-brand-sep">/</span>
      {props.mode === "build" ? <select className="tb-brand-run config-select" value={props.configName} onChange={(event) => props.onChooseConfig(event.target.value)}>{props.configNames.map((name) => <option key={name}>{name}</option>)}</select> : <span className="tb-brand-run">{props.run?.run_id || "no_run"}</span>}
    </div>
    <div className="mode-toggle" role="tablist">
      <button className={`mode-btn${props.mode === "build" ? " on" : ""}`} onClick={() => props.setMode("build")}>Build</button>
      <button className={`mode-btn${props.mode === "run" ? " on" : ""}`} disabled={!props.run} onClick={() => props.run && props.setMode("run")}>Run</button>
    </div>
    <div className="tb-spacer" />
    <div className="tb-meta">
      {props.mode === "build" ? <><span className="tb-meta-num">{props.pipelineCount}</span><span className="tb-meta-lbl">blocks</span><span className="tb-meta-dot">·</span><span className="tb-meta-num">{props.filterCount}</span><span className="tb-meta-lbl">filters</span></> : <><span className="tb-meta-num">{props.metrics.accepted.toLocaleString()}</span><span className="tb-meta-lbl">accepted</span><span className="tb-meta-dot">·</span><span className="tb-meta-num">{formatTime(props.metrics.elapsed_sec)}</span><span className="tb-meta-lbl">elapsed</span></>}
    </div>
    <div className="tb-actions">
      <button className="btn btn-ghost" onClick={props.onCredentials}><Icon name="key" size={13} /> Keys</button>
      {props.mode === "build" && <><button className="btn btn-ghost" disabled={props.busy} onClick={props.onValidate}><Icon name="file-check" size={13} /> Validate</button><button className="btn btn-ghost" disabled={props.busy} onClick={props.onSave}><Icon name="save" size={13} /> Save</button></>}
      <button className="btn btn-ghost" disabled={props.busy} onClick={props.onReset} title="Reload configuration"><Icon name="rotate-ccw" size={13} /> Reset</button>
      {running ? <button className="btn" disabled><Icon name="loader" className="spin" size={13} /> Running</button> : <button className="btn btn-primary" disabled={!props.canRun || props.busy} onClick={props.onRun}><Icon name="play" size={13} /> Run pipeline</button>}
    </div>
  </header>
}

export function BuildSeeds({ seeds, onChange }: { seeds: ConfigObject[]; onChange: (seeds: ConfigObject[]) => void }) {
  const [draft, setDraft] = useState("")
  const add = () => {
    const value = draft.trim()
    if (!value) return
    const isId = /^(DOI:|ARXIV:|[0-9a-f]{40}$)/i.test(value)
    onChange([...seeds, isId ? { paper_id: value } : { title: value }])
    setDraft("")
  }
  return <aside className="panel panel-left">
    <div className="ph"><span className="ph-title">Seeds</span><span className="ph-count">{seeds.length} selected</span></div>
    <div className="searchbox"><Icon name="search" size={12} /><input value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") add() }} placeholder="Add paper title or DOI…" /><button className="ph-btn" onClick={add} title="Add seed"><Icon name="plus" size={11} /></button></div>
    <div className="pb-scroll">
      <div className="seeds-counter"><span><span className="seeds-counter-n">{seeds.length}</span> configured · feeds seed block</span><Icon name="star" size={12} style={{ color: "var(--cc-warning)" }} /></div>
      <div className="seeds-list">{seeds.map((seed, index) => {
        const title = String(seed.title || seed.paper_id || "Untitled seed")
        return <div className="seed-card is-selected" key={`${title}-${index}`}><div className="seed-card-head"><span className="seed-star"><Icon name="star" size={13} /></span><span className="seed-title">{title}</span><button className="ph-btn seed-remove" title="Remove seed" onClick={() => onChange(seeds.filter((_, itemIndex) => itemIndex !== index))}><Icon name="x" size={11} /></button></div><div className="seed-meta"><span>{seed.paper_id ? "paper id" : "title"}</span><span className="seed-meta-sep">·</span><span>config.yaml</span></div></div>
      })}{!seeds.length && <div className="config-empty"><em>No seed papers configured.</em></div>}</div>
    </div>
  </aside>
}

function PipelineBlock({ node, selected, onClick, index }: { node: PipelineNode; selected: boolean; onClick: () => void; index: number }) {
  return <div className={`pblock style-card${selected ? " is-selected" : ""}`} onClick={onClick} data-testid={`pipeline-step-${index}`}><div className="pb-row"><span className="pb-icon"><Icon name={node.icon} size={11} /></span><span className="pb-name">{node.name}</span></div><div className="pb-meta"><span className="pb-meta-id">{node.localId}</span>{node.filters > 0 && <><span className="pb-meta-sep">·</span><span>{node.filters} filter{node.filters === 1 ? "" : "s"}</span></>}</div></div>
}

export function BuildPipeline({ pipeline, selectedIndex, onSelect }: { pipeline: PipelineNode[]; selectedIndex: number; onSelect: (index: number) => void }) {
  const [zoom, setZoom] = useState(100)
  return <section className="pane pane-top"><div className="pipe"><div className="pipe-hint">Click a block to configure · click palette blocks to append</div><div className="pipe-toolbar"><button className="btn-icon btn" onClick={() => setZoom((value) => Math.max(60, value - 10))}><Icon name="minus" size={11} /></button><span className="pipe-zoom">{zoom}%</span><button className="btn-icon btn" onClick={() => setZoom((value) => Math.min(160, value + 10))}><Icon name="plus" size={11} /></button><button className="btn-icon btn" title="Fit" onClick={() => setZoom(100)}><Icon name="maximize-2" size={11} /></button></div><div className="pipe-inner"><div className="pipe-row" style={{ transform: `scale(${zoom / 100})` }}>{pipeline.map((node, index) => <span className="pipe-fragment" key={node.id}>{index > 0 && <span className="pipe-edge"><svg viewBox="0 0 40 14" preserveAspectRatio="none"><line x1="0" y1="7" x2="32" y2="7" stroke="var(--cc-ink-2)" strokeWidth="1.25" strokeLinecap="round" /><path d="M 32 3 L 38 7 L 32 11 Z" fill="var(--cc-ink-2)" /></svg></span>}<PipelineBlock node={node} index={index} selected={selectedIndex === index} onClick={() => onSelect(index)} /></span>)}</div>{!pipeline.length && <div className="config-empty"><em>No pipeline steps configured. Add one from the palette.</em></div>}</div></div></section>
}

function ConfigField({ label, children, wide, full, hint }: { label: string; children: ReactNode; wide?: boolean; full?: boolean; hint?: string }) {
  return <div className={`field${wide ? " field-wide" : ""}${full ? " field-full" : ""}`}><span className="field-label">{label}</span>{children}{hint && <span className="field-hint">{hint}</span>}</div>
}

function JsonField({ label, value, onChange }: { label: string; value: unknown; onChange: (value: unknown) => void }) {
  const [draft, setDraft] = useState(() => JSON.stringify(value, null, 2))
  useEffect(() => setDraft(JSON.stringify(value, null, 2)), [value])
  return <ConfigField label={label} wide><textarea className="config-json" value={draft} onChange={(event) => setDraft(event.target.value)} onBlur={() => { try { onChange(JSON.parse(draft)) } catch { /* preserve invalid draft for correction */ } }} /></ConfigField>
}

export function BuildConfig(props: {
  node: PipelineNode | null
  configYaml: string
  parseError: string
  onYamlChange: (value: string) => void
  onUpdateField: (key: string, value: unknown) => void
  onDuplicate: () => void
  onDelete: () => void
}) {
  const [yamlMode, setYamlMode] = useState(false)
  return <section className="config">
    <div className="config-head"><div className="config-head-left"><div className="config-title">{yamlMode ? "Complete configuration" : props.node?.name || "Pipeline configuration"}</div><span className="config-sub">{yamlMode ? "config.yaml · authoritative" : props.node ? `${props.node.localId} · ${props.node.step}` : "no selection"}</span></div><div className="config-actions"><button className={`btn btn-ghost${yamlMode ? " is-on" : ""}`} onClick={() => setYamlMode((value) => !value)}><Icon name="file-check" size={12} /> YAML</button>{props.node && !yamlMode && <><button className="btn btn-ghost" onClick={props.onDuplicate}><Icon name="copy" size={12} /> Duplicate</button><button className="btn btn-ghost" title="Remove" onClick={props.onDelete}><Icon name="trash-2" size={12} /></button></>}</div></div>
    {yamlMode ? <div className="config-yaml-wrap">{props.parseError && <div className="yaml-error">{props.parseError}</div>}<textarea className="config-yaml" spellCheck={false} value={props.configYaml} onChange={(event) => props.onYamlChange(event.target.value)} /></div> : props.node ? <div className="config-body">{Object.entries(props.node.raw).filter(([key]) => key !== "step").map(([key, value]) => {
      const label = key.replaceAll("_", " ")
      if (value && typeof value === "object") return <JsonField key={key} label={label} value={value} onChange={(next) => props.onUpdateField(key, next)} />
      if (typeof value === "boolean") return <ConfigField key={key} label={label}><select value={String(value)} onChange={(event) => props.onUpdateField(key, event.target.value === "true")}><option value="true">true</option><option value="false">false</option></select></ConfigField>
      return <ConfigField key={key} label={label}><input type={typeof value === "number" ? "number" : "text"} value={String(value ?? "")} onChange={(event) => props.onUpdateField(key, typeof value === "number" ? Number(event.target.value) : event.target.value)} /></ConfigField>
    })}{Object.keys(props.node.raw).length === 1 && <div className="config-empty"><em>This step has no explicit parameters.</em></div>}</div> : <div className="config-empty"><em>Select a pipeline block to configure its parameters.</em></div>}
  </section>
}

export function BuildBlocks({ onAdd }: { onAdd: (definition: BlockDefinition) => void }) {
  const [query, setQuery] = useState("")
  const visible = BLOCKS.filter((block) => `${block.name} ${block.hint}`.toLowerCase().includes(query.toLowerCase()))
  const categories = ["Sources", "Expanders", "Rerankers", "Sinks"]
  return <aside className="panel panel-right"><div className="ph"><span className="ph-title">Building blocks</span><span className="ph-count">{BLOCKS.length} total</span></div><div className="searchbox"><Icon name="search" size={12} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search blocks…" /></div><div className="pb-scroll"><div className="blocks-list">{categories.map((category) => {
    const items = visible.filter((block) => block.cat === category)
    if (!items.length) return null
    return <span className="blocks-group" key={category}><span className="blocks-cat"><span>{category}</span><span className="blocks-cat-n">{items.length}</span></span>{items.map((item) => <button key={item.kind} className="block-item" title="Append to pipeline" onClick={() => onAdd(item)}><span className="block-icon"><Icon name={item.icon} size={12} /></span><span className="block-body"><span className="block-name">{item.name}</span><span className="block-hint">{item.hint}</span></span><span className="block-add"><Icon name="plus" size={13} /></span></button>)}</span>
  })}</div></div><div className="blocks-foot">Click a block to append · edit complete step parameters in the inspector or YAML</div></aside>
}

export function CredentialsModal(props: {
  catalog: Catalog
  credentials: Credentials
  credentialStatus: CredentialStatus
  selectedModel: string
  reasoningEffort: string
  onCredentialsChange: (credentials: Credentials) => void
  onModelChange: (model: string) => void
  onReasoningChange: (effort: string) => void
  onClose: () => void
}) {
  const supported = props.selectedModel === props.catalog.supported_model && props.reasoningEffort === props.catalog.supported_reasoning_effort
  const byProvider = (provider: "gemini" | "openai") => props.catalog.models.filter((model) => model.provider === provider)
  return <div className="modal-backdrop"><div className="settings-modal" role="dialog" aria-label="Provider setup"><div className="settings-head"><div><span className="ph-title">Provider setup</span><strong>Credentials & model</strong></div><button className="btn btn-icon" onClick={props.onClose}><Icon name="x" size={13} /></button></div><div className="settings-body"><div className="settings-keys"><div className="settings-eyebrow">API credentials · local memory only</div>{([
    ["Semantic Scholar", "s2_api_key", "s2"],
    ["Gemini", "gemini_api_key", "gemini"],
    ["OpenAI", "openai_api_key", "openai"],
  ] as const).map(([label, key, env]) => <label className="secret-field" key={key}><span>{label}<i className={props.credentialStatus[env] ? "env-ok" : "env-empty"}>{props.credentialStatus[env] ? "env set" : "not set"}</i></span><div><Icon name="key" size={13} /><input type="password" autoComplete="off" value={props.credentials[key]} onChange={(event) => props.onCredentialsChange({ ...props.credentials, [key]: event.target.value })} placeholder={props.credentialStatus[env] ? "Using environment value" : `${label} API key`} /></div></label>)}<p className="security-note">Keys remain in this browser tab and local backend process. They are excluded from YAML, run snapshots, and logs.</p><div className="settings-eyebrow model-heading">Screening model</div><label className="select-field"><span>Provider model</span><select value={props.selectedModel} onChange={(event) => props.onModelChange(event.target.value)}><optgroup label="Gemini">{byProvider("gemini").map((model) => <option value={model.id} key={model.id}>{model.id} · ${model.input_usd_per_m}/${model.output_usd_per_m}</option>)}</optgroup><optgroup label="OpenAI">{byProvider("openai").map((model) => <option value={model.id} key={model.id}>{model.id} · ${model.input_usd_per_m}/${model.output_usd_per_m}</option>)}</optgroup></select></label><label className="select-field"><span>Reasoning effort</span><select value={props.reasoningEffort} onChange={(event) => props.onReasoningChange(event.target.value)}><option>minimal</option><option>low</option><option>medium</option><option>high</option></select></label>{supported ? <div className="support-ok"><Icon name="check" size={13} /> Runnable in this release</div> : <div className="support-warning"><strong>Not supported in v0.1</strong><span>Select <code>{props.catalog.supported_model}</code> with <code>{props.catalog.supported_reasoning_effort}</code> reasoning.</span></div>}</div><div className="settings-catalog"><div className="settings-eyebrow">Model price catalog · USD per 1M text tokens</div><div className="catalog-date">Checked {props.catalog.pricing_updated_at}</div>{(["gemini", "openai"] as const).map((provider) => <div className="catalog-group" key={provider}><b>{provider}</b>{byProvider(provider).map((model) => <div className={`catalog-row${model.runnable ? " runnable" : ""}`} key={model.id}><code>{model.id}</code><span>${model.input_usd_per_m}<i>in</i></span><span>${model.output_usd_per_m}<i>out</i></span></div>)}</div>)}<div className="catalog-links">{Object.entries(props.catalog.sources).map(([provider, url]) => <a href={url} target="_blank" rel="noreferrer" key={provider}>{provider} pricing</a>)}</div></div></div><div className="settings-foot"><span>loopback only · keys not persisted</span><button className="btn btn-primary" onClick={props.onClose}>Done</button></div></div></div>
}

export function RunProgress({ steps, run, metrics }: { steps: StepState[]; run: RunSnapshot | null; metrics: Metrics }) {
  const done = steps.filter((step) => step.status === "done").length
  const current = steps.find((step) => step.status === "running")
  const progress = steps.length ? Math.round(((done + (current ? 0.5 : 0)) / steps.length) * 100) : 0
  return <aside className="panel panel-left"><div className="ph"><span className="ph-title">Pipeline progress</span><span className="ph-count">{done} / {steps.length}</span></div><div className="pb-scroll"><div className="progress-list">{steps.map((step, index) => <div key={`${step.idx}-${step.name}`} className={`prog-step is-${step.status === "running" ? "active" : step.status}`}><span className="prog-badge">{step.status === "done" ? "✓" : index + 1}</span><div className="prog-body"><span className="prog-name">{STEP_META[step.name]?.name || step.name}</span><span className="prog-meta">{STEP_META[step.name]?.prefix || "STP"}-{String(index + 1).padStart(2, "0")} · {step.status === "done" ? `${step.in_count?.toLocaleString()} screened · ${step.out_count?.toLocaleString()} pass` : step.description || "pending"}</span>{step.status === "running" && <div className="prog-bar-track"><div className="prog-bar-fill indeterminate" /></div>}</div></div>)}</div><div className="prog-summary"><div className="prog-summary-row"><span className="prog-summary-lbl">Overall</span><span className="prog-summary-val">{run?.status === "completed" ? 100 : progress}%</span></div><div className="prog-summary-row"><span className="prog-summary-lbl">Current</span><span className="prog-summary-val">{current ? STEP_META[current.name]?.name || current.name : run?.status || "pending"}</span></div><div className="prog-summary-row"><span className="prog-summary-lbl">Elapsed</span><span className="prog-summary-val">{formatTime(metrics.elapsed_sec)}</span></div>{run?.error && <div className="run-error">{run.error}</div>}</div></div></aside>
}

const YEAR_RAMP = ["#adb5bd", "#97a0a9", "#818b95", "#6b7681", "#55606d", "#414b58", "#2d3744", "#1a2330"]
function yearColor(year: number | null): string { return YEAR_RAMP[Math.max(0, Math.min(YEAR_RAMP.length - 1, (year || 2018) - 2018))] }
function nodeSize(paper: GraphPaper): number { return paper.seed ? 10 : Math.max(4, Math.min(11, 3 + Math.log10(Math.max(1, paper.citation_count)) * 2)) }
function hash(value: string): number { let output = 2166136261; for (let i = 0; i < value.length; i += 1) output = Math.imul(output ^ value.charCodeAt(i), 16777619); return output >>> 0 }

function layoutGraph(payload: GraphPayload) {
  if (!payload.nodes.length) return { nodes: [] as Array<GraphPaper & { x: number; y: number }>, edges: payload.edges }
  if (payload.nodes.length === 1) return { nodes: [{ ...payload.nodes[0], x: 500, y: 300 }], edges: payload.edges }
  const graph = new MultiDirectedGraph()
  payload.nodes.forEach((paper, index) => {
    const angle = ((hash(paper.paper_id) % 360) / 180) * Math.PI
    const radius = 1 + (index % 17) / 9
    graph.addNode(paper.paper_id, { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius })
  })
  payload.edges.forEach((edge, index) => {
    if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) graph.addEdgeWithKey(`edge-${index}`, edge.source, edge.target)
  })
  if (graph.order > 1) forceAtlas2.assign(graph, { iterations: graph.order < 250 ? 80 : graph.order < 900 ? 45 : 24, settings: forceAtlas2.inferSettings(graph) })
  const coordinates = payload.nodes.map((paper) => ({ paper, x: Number(graph.getNodeAttribute(paper.paper_id, "x")), y: Number(graph.getNodeAttribute(paper.paper_id, "y")) }))
  const xs = coordinates.map((item) => item.x); const ys = coordinates.map((item) => item.y)
  const minX = Math.min(...xs); const maxX = Math.max(...xs); const minY = Math.min(...ys); const maxY = Math.max(...ys)
  return { nodes: coordinates.map(({ paper, x, y }) => ({ ...paper, x: 65 + ((x - minX) / Math.max(0.001, maxX - minX)) * 870, y: 65 + ((y - minY) / Math.max(0.001, maxY - minY)) * 470 })), edges: payload.edges }
}

export function RunNetwork({ graph, metrics }: { graph: GraphPayload; metrics: Metrics }) {
  const [picked, setPicked] = useState<GraphPaper | null>(null)
  const laidOut = useMemo(() => layoutGraph(graph), [graph])
  const byId = useMemo(() => new Map(laidOut.nodes.map((node) => [node.paper_id, node])), [laidOut.nodes])
  const perMinute = metrics.elapsed_sec ? Math.round(metrics.accepted / metrics.elapsed_sec * 60) : 0
  return <section className="pane pane-top"><div className="network">{laidOut.nodes.length ? <svg viewBox="0 0 1000 600" preserveAspectRatio="xMidYMid meet"><g>{laidOut.edges.map((edge, index) => { const source = byId.get(edge.source); const target = byId.get(edge.target); return source && target ? <line key={index} x1={source.x} y1={source.y} x2={target.x} y2={target.y} stroke={yearColor(target.year)} strokeWidth="0.9" strokeOpacity="0.55" /> : null })}</g><g>{laidOut.nodes.map((node) => <g key={node.paper_id}>{node.seed && <circle cx={node.x} cy={node.y} r={nodeSize(node) + 6} fill="none" stroke="#e63946" strokeWidth="1.2" strokeOpacity="0.5" />}<circle data-testid={`graph-node-${node.paper_id}`} cx={node.x} cy={node.y} r={nodeSize(node)} fill={node.seed ? "#e63946" : yearColor(node.year)} stroke={picked?.paper_id === node.paper_id ? "#3e3548" : node.seed ? "#c42a37" : "none"} strokeWidth={picked?.paper_id === node.paper_id ? 2 : node.seed ? 1 : 0} onClick={() => setPicked(node)} className="graph-node" /></g>)}</g></svg> : <div className="network-empty"><strong>Graph awaiting papers</strong><span>Real nodes appear after each completed pipeline step.</span></div>}
    <div className="net-legend"><span className="net-legend-item"><span className="net-dot seed" />Seed</span><span className="net-legend-item"><span className="net-ramp">{YEAR_RAMP.map((color) => <span className="net-ramp-step" key={color} style={{ background: color }} />)}</span><span>2018 → 2025</span></span><span className="net-legend-item"><span className="net-edge-demo" />Edge · target year</span></div><div className="net-counter"><span><span className="net-counter-num">{graph.nodes.length.toLocaleString()}</span><span className="net-counter-lbl">nodes</span></span><span className="net-counter-sep">·</span><span><span className="net-counter-num">{graph.edges.length.toLocaleString()}</span><span className="net-counter-lbl">edges</span></span><span className="net-counter-sep">·</span><span><span className="net-counter-num">+{perMinute}</span><span className="net-counter-lbl">/ min</span></span></div>
    {picked && <div className="paper-drawer"><div className="drawer-head"><div className="drawer-eyebrow">{picked.seed ? "SEED PAPER" : `ACCEPTED · DEPTH ${picked.depth}`}</div><button className="btn btn-icon" onClick={() => setPicked(null)}><Icon name="x" size={12} /></button></div><div className="drawer-title">{picked.title}</div><div className="drawer-meta">{picked.authors.join(", ") || "Authors unavailable"} · {picked.year || "—"} · {picked.venue || "Venue unavailable"}</div><div className="drawer-stats"><div><span className="ds-num">{picked.citation_count.toLocaleString()}</span><span className="ds-lbl">cites</span></div><div><span className="ds-num">{picked.year || "—"}</span><span className="ds-lbl">year</span></div><div><span className="ds-num">{picked.depth}</span><span className="ds-lbl">depth</span></div></div><div className="drawer-abstract"><div className="drawer-label">Abstract</div><p>{picked.abstract || "Abstract unavailable."}</p></div><code className="drawer-id">{picked.paper_id}</code></div>}
  </div></section>
}

export function RunDashboard({ metrics, steps, run, logs, shape }: { metrics: Metrics; steps: StepState[]; run: RunSnapshot | null; logs: Array<{ level: string; message: string }>; shape: string }) {
  const [tab, setTab] = useState<"overview" | "rejects" | "cost" | "logs">("overview")
  const rejections = Object.entries(metrics.rejection_counts).sort((a, b) => b[1] - a[1])
  const rejectMax = rejections[0]?.[1] || 1
  const rejected = rejections.reduce((total, [, count]) => total + count, 0)
  const cacheTotal = metrics.s2_requests + metrics.s2_cache_hits
  const cacheRate = cacheTotal ? metrics.s2_cache_hits / cacheTotal * 100 : 0
  const current = steps.find((step) => step.status === "running")
  const perMinute = metrics.elapsed_sec ? metrics.accepted / metrics.elapsed_sec * 60 : 0
  return <section className="dashboard"><div className="dash-head"><span className="dash-head-title">Dashboard · {current?.name || run?.status || "pending"}</span><div className="dash-tabs">{(["overview", "rejects", "cost", "logs"] as const).map((name) => <button className={`dash-tab${tab === name ? " on" : ""}`} onClick={() => setTab(name)} key={name}>{name === "rejects" ? "Rejections" : name[0].toUpperCase() + name.slice(1)}</button>)}</div></div><div className="dash-metrics"><div className="metric"><span className="metric-label">Accepted</span><span className="metric-value">{formatCount(metrics.accepted)}</span><span className="metric-sub"><span className="metric-delta">↑ {perMinute.toFixed(1)} / min</span></span></div><div className="metric"><span className="metric-label">Rejected</span><span className="metric-value">{formatCount(metrics.rejected || rejected)}</span><span className="metric-sub">{rejections.length} reasons</span></div><div className="metric"><span className="metric-label">LLM tokens</span><span className="metric-value">{formatCount(metrics.llm_tokens)}</span><span className="metric-sub">{formatCount(metrics.llm_input_tokens)} in · {formatCount(metrics.llm_output_tokens)} out</span></div><div className="metric"><span className="metric-label">LLM calls</span><span className="metric-value">{formatCount(metrics.llm_calls)}</span><span className="metric-sub">batched dispatches</span></div><div className="metric"><span className="metric-label">Cost</span><span className="metric-value">${metrics.cost_usd.toFixed(2)}</span><span className="metric-sub">live estimate</span></div><div className="metric"><span className="metric-label">S2 cache</span><span className="metric-value">{cacheRate.toFixed(0)}%</span><span className="metric-sub">{metrics.s2_cache_hits.toLocaleString()} / {cacheTotal.toLocaleString()} reqs</span></div></div>
    {tab === "overview" && <div className="dash-bars"><div className="bars-col"><div className="bars-title"><span>Top rejection reasons</span><span className="bars-total">{rejected.toLocaleString()} total</span></div><div className="bars-list">{rejections.slice(0, 5).map(([reason, count]) => <div className="bar-row" key={reason}><div className="bar-head"><span className="bar-label mono">{reason}</span><span className="bar-val">{count.toLocaleString()}</span></div><div className="bar-track"><div className="bar-fill reject" style={{ width: `${count / rejectMax * 100}%` }} /></div></div>)}{!rejections.length && <em className="dash-empty">No rejections recorded.</em>}</div></div><div className="bars-col"><div className="bars-title"><span>Run state</span><span className="bars-total">{run?.status || "pending"}</span></div><div className="run-facts"><span><b>{metrics.seen.toLocaleString()}</b> unique papers observed</span><span><b>{steps.filter((step) => step.status === "done").length} / {steps.length}</b> pipeline steps complete</span><span><b>{formatCount(metrics.llm_reasoning_tokens)}</b> reasoning tokens</span><span><b>{graphSummary(shape)}</b> latest shape</span></div></div></div>}
    {tab === "rejects" && <div className="dashboard-tab-body bars-list">{rejections.map(([reason, count]) => <div className="bar-row" key={reason}><div className="bar-head"><span className="bar-label mono">{reason}</span><span className="bar-val">{count.toLocaleString()}</span></div><div className="bar-track"><div className="bar-fill reject" style={{ width: `${count / rejectMax * 100}%` }} /></div></div>)}{!rejections.length && <em className="dash-empty">No rejection reasons recorded.</em>}</div>}
    {tab === "cost" && <div className="dashboard-tab-body"><div className="cost-summary"><span className="metric-label">Local estimate</span><strong>${metrics.cost_usd.toFixed(4)}</strong><p>Calculated from provider token counts and the verified model pricing table.</p></div></div>}
    {tab === "logs" && <div className="dashboard-tab-body dash-log">{logs.length ? logs.map((line, index) => <div key={index}><span className={`log-${line.level.toLowerCase()}`}>{line.level}</span>{line.message}</div>) : <em>No log records yet.</em>}</div>}
  </section>
}

function graphSummary(shape: string): string {
  const line = shape.trim().split("\n").filter(Boolean).at(-1)
  if (!line) return "pending"
  return /^[-=]+$/.test(line.trim()) ? "written" : line.slice(0, 24)
}

export function RunAccepted({ papers, freshIds, running }: { papers: GraphPaper[]; freshIds: Set<string>; running: boolean }) {
  const [sort, setSort] = useState<"recent" | "year" | "cites">("recent")
  const [selected, setSelected] = useState<string | null>(null)
  const sorted = useMemo(() => {
    const copy = [...papers]
    if (sort === "year") copy.sort((a, b) => (b.year || 0) - (a.year || 0))
    if (sort === "cites") copy.sort((a, b) => b.citation_count - a.citation_count)
    return copy.slice(0, 50)
  }, [papers, sort])
  return <aside className="panel panel-right"><div className="ph"><span className="ph-title">Accepted</span><span className="ph-count"><span className="accepted-total">{papers.length.toLocaleString()}</span> total · showing {Math.min(50, papers.length)}</span></div><div className="accepted-controls"><select className="accepted-sort" value={sort} onChange={(event) => setSort(event.target.value as typeof sort)}><option value="recent">Recently accepted</option><option value="year">Newest year</option><option value="cites">Most cited</option></select><button className="ph-btn" title="Artifacts are written to the run output directory" disabled><Icon name="download" size={13} /></button><button className="ph-btn" title="Filter"><Icon name="filter" size={13} /></button></div><div className="accepted-scroll">{sorted.map((paper) => <div key={paper.paper_id} className={`acc-item${selected === paper.paper_id ? " is-selected" : ""}${freshIds.has(paper.paper_id) ? " is-fresh" : ""}`} onClick={() => setSelected(paper.paper_id)}><span className="acc-score" title="Citation count">{formatCount(paper.citation_count)}</span><div className="acc-title">{paper.title}</div><span className="acc-depth">d{paper.depth}</span><div className="acc-meta" style={{ gridColumn: "2 / 4" }}><span>{paper.authors.slice(0, 2).join(", ") || "Authors unavailable"}</span><span className="acc-meta-sep">·</span><span>{paper.year || "—"}</span><span className="acc-meta-sep">·</span><span>{paper.venue || paper.source}</span><span className="acc-meta-sep">·</span><span>{paper.citation_count.toLocaleString()} cites</span></div></div>)}{!papers.length && <div className="config-empty"><em>Accepted stream is empty.</em></div>}</div><div className="acc-foot"><span className="acc-foot-stream"><span className={`sb-dot ${running ? "sb-dot-run" : "sb-dot-idle"}`} />{running ? "Streaming" : "Stream closed"}</span><span>virtualized · {Math.min(50, papers.length)} of {papers.length.toLocaleString()}</span></div></aside>
}

export function BottomBar(props: { mode: "build" | "run"; running: boolean; configName: string; pipelineLen: number; seedCount: number; steps: StepState[]; metrics: Metrics; run: RunSnapshot | null }) {
  const cacheTotal = props.metrics.s2_requests + props.metrics.s2_cache_hits
  const cacheRate = cacheTotal ? props.metrics.s2_cache_hits / cacheTotal * 100 : 0
  const completed = props.steps.filter((step) => step.status === "done").length
  const current = props.steps.find((step) => step.status === "running")
  const progress = props.steps.length ? Math.round(((completed + (current ? 0.5 : 0)) / props.steps.length) * 100) : 0
  return <footer className="statusbar"><span className="sb-item"><span className={`sb-dot ${props.running ? "sb-dot-run" : "sb-dot-idle"}`} /><span className="sb-key">{props.running ? "Running" : props.run?.status || "Idle"}</span></span><span className="sb-sep">·</span>{props.mode === "build" ? <><span className="sb-item"><Icon name="git-branch" size={11} style={{ color: "var(--cc-ink-3)" }} /><span className="sb-key">Config</span><span className="sb-val">{props.configName}</span></span><span className="sb-sep">·</span><span className="sb-item"><span className="sb-key">Pipeline</span><span className="sb-val">{props.pipelineLen} blocks</span><span className="sb-key">·</span><span className="sb-val">{Math.max(0, props.pipelineLen - 1)} edges</span></span><span className="sb-sep">·</span><span className="sb-item"><span className="sb-key">Seeds</span><span className="sb-val">{props.seedCount} selected</span></span></> : <><span className="sb-item"><span className="sb-key">Step</span><span className="sb-val">{current ? `${current.idx} / ${props.steps.length} · ${current.name}` : `${completed} / ${props.steps.length}`}</span></span><span className="sb-sep">·</span><span className="sb-item"><span className="sb-key">Accepted</span><span className="sb-val">{props.metrics.accepted.toLocaleString()}</span></span><span className="sb-sep">·</span><span className="sb-item"><span className="sb-key">Elapsed</span><span className="sb-val">{formatTime(props.metrics.elapsed_sec)}</span></span><span className="sb-sep">·</span><div className="sb-progress"><div className="sb-progress-fill" style={{ width: `${props.run?.status === "completed" ? 100 : progress}%` }} /></div><span className="sb-val">{props.run?.status === "completed" ? 100 : progress}%</span></>}<span className="sb-spacer" /><span className="sb-item"><Icon name="database" size={11} style={{ color: "var(--cc-ink-3)" }} /><span className="sb-key">S2 cache</span><span className="sb-val">{cacheRate.toFixed(0)}%</span></span><span className="sb-sep">·</span><span className="sb-item"><Icon name="dollar-sign" size={11} style={{ color: "var(--cc-ink-3)" }} /><span className="sb-val">${props.metrics.cost_usd.toFixed(2)}</span></span><span className="sb-sep">·</span><span className="sb-item"><span className="sb-key">connected · local</span></span></footer>
}

export function HitlModal({ runId, papers, onDone }: { runId: string; papers: HitlPaper[]; onDone: () => void }) {
  const [labels, setLabels] = useState<Record<string, boolean>>({})
  const [busy, setBusy] = useState(false)
  const submit = async (stop = false) => { setBusy(true); await api.hitl(runId, labels, stop); onDone() }
  return <div className="modal-backdrop"><div className="hitl-modal"><div className="settings-eyebrow">Human review · {papers.length} papers</div><h2>Check screening decisions</h2><p>Mark each sampled paper keep or reject, then resume the pipeline.</p><div className="hitl-list">{papers.map((paper) => <div className="hitl-paper" key={paper.paper_id}><strong>{paper.title}</strong><span>{paper.year || "—"} · {paper.venue || "venue unavailable"}</span><p>{paper.abstract || "Abstract unavailable."}</p><div><button className={labels[paper.paper_id] === true ? "on keep" : ""} onClick={() => setLabels((current) => ({ ...current, [paper.paper_id]: true }))}>Keep</button><button className={labels[paper.paper_id] === false ? "on reject" : ""} onClick={() => setLabels((current) => ({ ...current, [paper.paper_id]: false }))}>Reject</button></div></div>)}</div><div className="modal-actions"><button className="btn" disabled={busy} onClick={() => void submit(true)}>Stop after review</button><button className="btn btn-primary" disabled={busy || Object.keys(labels).length !== papers.length} onClick={() => void submit()}>{busy ? "Submitting…" : "Submit & resume"}</button></div></div></div>
}

export function Notice({ notice, onClose }: { notice: { kind: "ok" | "error"; text: string }; onClose: () => void }) {
  return <div className={`notice ${notice.kind}`}><Icon name={notice.kind === "ok" ? "check" : "x"} size={14} /><span>{notice.text}</span><button onClick={onClose}>×</button></div>
}
