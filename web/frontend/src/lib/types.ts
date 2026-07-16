export type ConfigObject = Record<string, unknown>

export interface ModelPrice {
  provider: "openai" | "gemini"
  id: string
  input_usd_per_m: number
  cached_input_usd_per_m: number | null
  output_usd_per_m: number
  long_context: Record<string, number> | null
  note: string
  runnable: boolean
}

export interface Catalog {
  models: ModelPrice[]
  supported_model: string
  supported_reasoning_effort: string
  pricing_updated_at: string
  sources: Record<string, string>
  scope: string
}

export interface Credentials {
  s2_api_key: string
  gemini_api_key: string
  openai_api_key: string
}

export interface CredentialStatus {
  s2: boolean
  gemini: boolean
  openai: boolean
}

export interface Metrics {
  accepted: number
  rejected: number
  seen: number
  llm_tokens: number
  llm_input_tokens: number
  llm_output_tokens: number
  llm_reasoning_tokens: number
  llm_calls: number
  cost_usd: number
  s2_requests: number
  s2_cache_hits: number
  rejection_counts: Record<string, number>
  elapsed_sec: number
}

export interface GraphPaper {
  paper_id: string
  title: string
  abstract: string
  authors: string[]
  year: number | null
  venue: string
  citation_count: number
  depth: number
  source: string
  seed: boolean
  external_ids: Record<string, string>
}

export interface GraphPayload {
  nodes: GraphPaper[]
  edges: Array<{ source: string; target: string }>
}

export interface StepState {
  idx: number
  name: string
  description?: string
  status: "idle" | "running" | "done"
  in_count?: number
  out_count?: number
  delta_collection?: number
  stats?: Record<string, unknown>
}

export interface RunSnapshot {
  run_id: string
  config_name: string
  output_dir: string
  status: "queued" | "running" | "completed" | "failed"
  error: string | null
  metrics: Metrics
}

export interface HitlPaper {
  paper_id: string
  title: string
  venue: string
  year: number | null
  abstract: string
}
