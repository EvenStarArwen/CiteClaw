import { useMemo, useState } from "react"
import { Braces, Check, ChevronRight, FileCode2, KeyRound, Plus, Settings2, Trash2 } from "lucide-react"
import type { Catalog, ConfigObject, Credentials, CredentialStatus } from "../lib/types"

interface Props {
  configNames: string[]
  configName: string
  config: ConfigObject | null
  configYaml: string
  parseError: string
  catalog: Catalog | null
  credentials: Credentials
  credentialStatus: CredentialStatus
  onChooseConfig: (name: string) => void
  onConfigChange: (config: ConfigObject) => void
  onYamlChange: (text: string) => void
  onCredentialsChange: (credentials: Credentials) => void
}

function asArray(value: unknown): ConfigObject[] {
  return Array.isArray(value) ? value.filter((x): x is ConfigObject => !!x && typeof x === "object") : []
}

function cloneConfig(config: ConfigObject): ConfigObject {
  return structuredClone(config)
}

function Field({ label, children, hint, wide = false }: { label: string; children: React.ReactNode; hint?: string; wide?: boolean }) {
  return <label className={`field ${wide ? "field-wide" : ""}`}><span>{label}</span>{children}{hint && <small>{hint}</small>}</label>
}

function SecretField({ label, value, envSet, onChange, placeholder }: { label: string; value: string; envSet: boolean; onChange: (value: string) => void; placeholder: string }) {
  return <label className="secret-field">
    <span>{label}<i className={envSet ? "env-ok" : "env-empty"}>{envSet ? "env set" : "not set"}</i></span>
    <div><KeyRound size={13} /><input type="password" autoComplete="off" value={value} onChange={(e) => onChange(e.target.value)} placeholder={envSet ? "Using environment value" : placeholder} /></div>
  </label>
}

export function BuildView(props: Props) {
  const [selectedStep, setSelectedStep] = useState<number | null>(null)
  const [editor, setEditor] = useState<"form" | "yaml">("form")
  const [nestedDraft, setNestedDraft] = useState<Record<string, string>>({})
  const pipeline = asArray(props.config?.pipeline)
  const seeds = asArray(props.config?.seed_papers)
  const selected = selectedStep === null ? null : pipeline[selectedStep] || null
  const selectedModel = String(props.config?.screening_model || "")
  const supported = props.catalog?.supported_model
  const unsupported = !!selectedModel && !!supported && selectedModel !== supported
  const modelsByProvider = useMemo(() => ({
    gemini: props.catalog?.models.filter((m) => m.provider === "gemini") || [],
    openai: props.catalog?.models.filter((m) => m.provider === "openai") || [],
  }), [props.catalog])

  const updateRoot = (key: string, value: unknown) => {
    if (!props.config) return
    const next = cloneConfig(props.config); next[key] = value; props.onConfigChange(next)
  }
  const updateSeeds = (nextSeeds: ConfigObject[]) => updateRoot("seed_papers", nextSeeds)
  const updateStep = (key: string, value: unknown) => {
    if (!props.config || selectedStep === null) return
    const next = cloneConfig(props.config)
    const steps = asArray(next.pipeline)
    steps[selectedStep] = { ...steps[selectedStep], [key]: value }
    next.pipeline = steps
    props.onConfigChange(next)
  }

  return <>
    <aside className="panel panel-left">
      <div className="ph"><span className="ph-title">Run configuration</span><span className="ph-count">{props.configNames.length}</span></div>
      <div className="config-picker"><FileCode2 size={14} /><select value={props.configName} onChange={(e) => props.onChooseConfig(e.target.value)}>{props.configNames.map((name) => <option key={name}>{name}</option>)}</select></div>
      <div className="ph seed-head"><span className="ph-title">Seed papers</span><button className="ph-btn" title="Add title seed" onClick={() => updateSeeds([...seeds, { title: "" }])}><Plus size={14} /></button></div>
      <div className="pb-scroll seeds-list">
        {seeds.map((seed, index) => {
          const mode = "paper_id" in seed ? "paper_id" : "title"
          return <div className="seed-editor" key={index}>
            <div><span>{mode === "paper_id" ? "S2 / DOI" : "TITLE"}</span><button onClick={() => updateSeeds(seeds.filter((_, i) => i !== index))}><Trash2 size={12} /></button></div>
            <textarea rows={2} value={String(seed[mode] || "")} placeholder={mode === "paper_id" ? "DOI:10… or S2 ID" : "Paper title"} onChange={(e) => {
              const next = [...seeds]; next[index] = { [mode]: e.target.value }; updateSeeds(next)
            }} />
            <button className="seed-mode" onClick={() => {
              const next = [...seeds]; next[index] = mode === "paper_id" ? { title: "" } : { paper_id: "" }; updateSeeds(next)
            }}>use {mode === "paper_id" ? "title" : "paper id"}</button>
          </div>
        })}
        {!seeds.length && <div className="empty-copy">No seed papers configured.</div>}
      </div>
      <div className="panel-note"><b>{seeds.length}</b> seeds · titles resolve through Semantic Scholar</div>
    </aside>

    <main className="main build-main">
      <section className="pane pane-top">
        <div className="pipe">
          <div className="pipe-hint">Select a block to edit · full schema available in YAML</div>
          <button className={`global-node ${selectedStep === null ? "is-selected" : ""}`} onClick={() => setSelectedStep(null)}><Settings2 size={15} />Global config</button>
          <div className="pipe-inner">
            {pipeline.map((step, index) => <div className="pipe-item" key={index}>
              {index > 0 && <div className="pipe-edge"><span /><ChevronRight size={14} /></div>}
              <button className={`pblock ${selectedStep === index ? "is-selected" : ""}`} onClick={() => setSelectedStep(index)}>
                <span className={`step-dot step-${String(step.step || "step").toLowerCase()}`} />
                <strong>{String(step.step || "Unnamed step")}</strong>
                <small>{String(index + 1).padStart(2, "0")} · {Object.keys(step).length - 1} params</small>
              </button>
            </div>)}
            {!pipeline.length && <div className="empty-copy">No pipeline steps. Edit the YAML to add one.</div>}
          </div>
        </div>
      </section>

      <section className="config-panel">
        <div className="config-head">
          <div><strong>{editor === "yaml" ? "Complete configuration" : selected ? String(selected.step) : "Global settings"}</strong><span>{editor === "yaml" ? "config.yaml · authoritative" : selected ? `step ${Number(selectedStep) + 1}` : "common fields"}</span></div>
          <div className="segmented"><button className={editor === "form" ? "on" : ""} onClick={() => setEditor("form")}><Settings2 size={12} />Form</button><button className={editor === "yaml" ? "on" : ""} onClick={() => setEditor("yaml")}><Braces size={12} />YAML</button></div>
        </div>
        {editor === "yaml" ? <div className="yaml-wrap">
          {props.parseError && <div className="inline-error">{props.parseError}</div>}
          <textarea className="yaml-editor" spellCheck={false} value={props.configYaml} onChange={(e) => props.onYamlChange(e.target.value)} />
        </div> : <div className="config-form">
          {!selected && <>
            <Field label="Topic description" wide><textarea rows={5} value={String(props.config?.topic_description || "")} onChange={(e) => updateRoot("topic_description", e.target.value)} /></Field>
            <Field label="Max papers"><input type="number" value={Number(props.config?.max_papers_total || 0)} onChange={(e) => updateRoot("max_papers_total", Number(e.target.value))} /></Field>
            <Field label="LLM batch size"><input type="number" value={Number(props.config?.llm_batch_size || 0)} onChange={(e) => updateRoot("llm_batch_size", Number(e.target.value))} /></Field>
            <Field label="LLM concurrency"><input type="number" value={Number(props.config?.llm_concurrency || 0)} onChange={(e) => updateRoot("llm_concurrency", Number(e.target.value))} /></Field>
            <Field label="S2 requests / sec"><input type="number" step="0.1" value={Number(props.config?.s2_requests_per_second ?? props.config?.s2_rps ?? 0.9)} onChange={(e) => updateRoot("s2_requests_per_second", Number(e.target.value))} /></Field>
            <Field label="Max LLM tokens"><input type="number" value={Number(props.config?.max_llm_tokens || 0)} onChange={(e) => updateRoot("max_llm_tokens", Number(e.target.value))} /></Field>
            <Field label="Max S2 requests"><input type="number" value={Number(props.config?.max_s2_requests || 0)} onChange={(e) => updateRoot("max_s2_requests", Number(e.target.value))} /></Field>
          </>}
          {selected && Object.entries(selected).filter(([key]) => key !== "step").map(([key, value]) => {
            const nested = typeof value === "object" && value !== null
            if (nested) {
              const draftKey = `${selectedStep}:${key}`
              const text = nestedDraft[draftKey] ?? JSON.stringify(value, null, 2)
              return <Field label={key.replaceAll("_", " ")} wide key={key} hint="JSON for nested configuration"><textarea rows={6} value={text} onChange={(e) => setNestedDraft((d) => ({ ...d, [draftKey]: e.target.value }))} onBlur={() => {
                try { updateStep(key, JSON.parse(text)); setNestedDraft((d) => { const n = { ...d }; delete n[draftKey]; return n }) } catch { /* retain draft for correction */ }
              }} /></Field>
            }
            if (typeof value === "boolean") return <Field label={key.replaceAll("_", " ")} key={key}><select value={String(value)} onChange={(e) => updateStep(key, e.target.value === "true")}><option value="true">true</option><option value="false">false</option></select></Field>
            return <Field label={key.replaceAll("_", " ")} key={key}><input type={typeof value === "number" ? "number" : "text"} value={String(value ?? "")} onChange={(e) => updateStep(key, typeof value === "number" ? Number(e.target.value) : e.target.value)} /></Field>
          })}
          {selected && Object.keys(selected).length === 1 && <div className="empty-copy">This step has no explicit parameters.</div>}
        </div>}
      </section>
    </main>

    <aside className="panel panel-right setup-panel">
      <div className="ph"><span className="ph-title">Provider setup</span><span className="ph-count">local memory</span></div>
      <div className="pb-scroll setup-scroll">
        <div className="setup-section">
          <div className="setup-title">API credentials</div>
          <SecretField label="Semantic Scholar" envSet={props.credentialStatus.s2} value={props.credentials.s2_api_key} placeholder="S2 API key · required" onChange={(value) => props.onCredentialsChange({ ...props.credentials, s2_api_key: value })} />
          <SecretField label="Gemini" envSet={props.credentialStatus.gemini} value={props.credentials.gemini_api_key} placeholder="Gemini API key · required" onChange={(value) => props.onCredentialsChange({ ...props.credentials, gemini_api_key: value })} />
          <SecretField label="OpenAI" envSet={props.credentialStatus.openai} value={props.credentials.openai_api_key} placeholder="OpenAI API key · optional" onChange={(value) => props.onCredentialsChange({ ...props.credentials, openai_api_key: value })} />
          <p className="security-note">Keys remain in this browser tab and backend process only. They are not saved to YAML or run artifacts.</p>
        </div>
        <div className="setup-section">
          <div className="setup-title">Screening model</div>
          <label className="select-field"><span>Provider model</span><select value={selectedModel} onChange={(e) => updateRoot("screening_model", e.target.value)}>
            <optgroup label="Gemini">{modelsByProvider.gemini.map((model) => <option value={model.id} key={model.id}>{model.id} · ${model.input_usd_per_m}/${model.output_usd_per_m}</option>)}</optgroup>
            <optgroup label="OpenAI">{modelsByProvider.openai.map((model) => <option value={model.id} key={model.id}>{model.id} · ${model.input_usd_per_m}/${model.output_usd_per_m}</option>)}</optgroup>
          </select></label>
          <label className="select-field"><span>Reasoning effort</span><select value={String(props.config?.reasoning_effort || "minimal")} onChange={(e) => updateRoot("reasoning_effort", e.target.value)}><option>minimal</option><option>low</option><option>medium</option><option>high</option></select></label>
          {unsupported ? <div className="support-warning"><strong>Not supported in v0.1</strong><span>Launch will stop before network access. Select <code>{supported}</code>.</span></div> : <div className="support-ok"><Check size={14} /><span>Runnable in this release</span></div>}
        </div>
        <details className="catalog-details">
          <summary>Model price catalog <span>{props.catalog?.models.length || 0}</span></summary>
          <div className="catalog-meta">Standard USD per 1M text tokens · checked {props.catalog?.pricing_updated_at}</div>
          {(["gemini", "openai"] as const).map((provider) => <div className="catalog-group" key={provider}><b>{provider}</b>{modelsByProvider[provider].map((model) => <div className={model.runnable ? "catalog-row runnable" : "catalog-row"} key={model.id}><code>{model.id}</code><span>${model.input_usd_per_m}<i>in</i></span><span>${model.output_usd_per_m}<i>out</i></span></div>)}</div>)}
          <div className="catalog-links">{Object.entries(props.catalog?.sources || {}).map(([provider, url]) => <a href={url} target="_blank" rel="noreferrer" key={provider}>{provider} pricing</a>)}</div>
        </details>
      </div>
    </aside>
  </>
}
