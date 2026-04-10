/**
 * Pipeline step type definitions for the React Flow builder.
 * Mirrors the STEP_REGISTRY in src/citeclaw/steps/__init__.py.
 */

export interface StepField {
  name: string
  label: string
  type: "string" | "number" | "boolean" | "select" | "json"
  default?: unknown
  options?: string[] // for select type
  required?: boolean
  placeholder?: string
  description?: string
}

export interface StepTypeDef {
  type: string
  label: string
  category: "seed" | "expand" | "filter" | "transform" | "output" | "control"
  description: string
  fields: StepField[]
  color: string
}

export const STEP_TYPES: StepTypeDef[] = [
  {
    type: "ResolveSeeds",
    label: "Resolve Seeds",
    category: "seed",
    description: "Resolve title-only seeds via S2 search; optionally find preprint/published siblings.",
    color: "#f59e0b",
    fields: [
      { name: "include_siblings", label: "Include siblings", type: "boolean", default: false, description: "Walk DOI/ArXiv links to find preprint/published pairs" },
    ],
  },
  {
    type: "LoadSeeds",
    label: "Load Seeds",
    category: "seed",
    description: "Initialize collection from seed papers.",
    color: "#f59e0b",
    fields: [
      { name: "file", label: "File", type: "string", placeholder: "Optional JSON file path", description: "Load seeds from a JSON file instead of config" },
    ],
  },
  {
    type: "ExpandForward",
    label: "Expand Forward",
    category: "expand",
    description: "Fetch papers that cite the signal, screen, and add survivors.",
    color: "#3b82f6",
    fields: [
      { name: "max_citations", label: "Max citations", type: "number", default: 100 },
      { name: "screener", label: "Screener block", type: "string", required: true, placeholder: "e.g. forward_screener" },
    ],
  },
  {
    type: "ExpandBackward",
    label: "Expand Backward",
    category: "expand",
    description: "Fetch references of the signal, screen, and add survivors.",
    color: "#3b82f6",
    fields: [
      { name: "screener", label: "Screener block", type: "string", required: true, placeholder: "e.g. forward_screener" },
    ],
  },
  {
    type: "ExpandBySearch",
    label: "Expand By Search",
    category: "expand",
    description: "Iterative meta-LLM search agent queries S2 to find related papers.",
    color: "#8b5cf6",
    fields: [
      { name: "screener", label: "Screener block", type: "string", required: true },
      { name: "topic_description", label: "Topic override", type: "string", placeholder: "Uses global topic if empty" },
      { name: "max_anchor_papers", label: "Max anchor papers", type: "number", default: 20 },
      { name: "agent", label: "Agent config", type: "json", default: { max_iterations: 4, target_count: 200, reasoning_effort: "high" }, description: "AgentConfig overrides as JSON" },
    ],
  },
  {
    type: "ExpandBySemantics",
    label: "Expand By Semantics",
    category: "expand",
    description: "S2 SPECTER2 kNN recommendations. No LLM needed.",
    color: "#8b5cf6",
    fields: [
      { name: "screener", label: "Screener block", type: "string", required: true },
      { name: "max_anchor_papers", label: "Max anchor papers", type: "number", default: 10 },
      { name: "limit", label: "Limit", type: "number", default: 100 },
      { name: "use_rejected_as_negatives", label: "Use rejected as negatives", type: "boolean", default: false },
    ],
  },
  {
    type: "ExpandByAuthor",
    label: "Expand By Author",
    category: "expand",
    description: "Top-K authors by metric, then fetch their papers.",
    color: "#8b5cf6",
    fields: [
      { name: "screener", label: "Screener block", type: "string", required: true },
      { name: "top_k_authors", label: "Top K authors", type: "number", default: 10 },
      { name: "author_metric", label: "Author metric", type: "select", default: "h_index", options: ["h_index", "citation_count", "degree_in_collab_graph"] },
      { name: "papers_per_author", label: "Papers per author", type: "number", default: 50 },
    ],
  },
  {
    type: "Rerank",
    label: "Rerank",
    category: "filter",
    description: "Score-based top-K selection with optional cluster-aware diversity.",
    color: "#10b981",
    fields: [
      { name: "metric", label: "Metric", type: "select", default: "citation", options: ["citation", "pagerank", "recency"] },
      { name: "k", label: "K", type: "number", default: 100 },
      { name: "diversity", label: "Diversity config", type: "json", placeholder: '{"type": "walktrap", "n_communities": 3}' },
    ],
  },
  {
    type: "ReScreen",
    label: "Re-Screen",
    category: "filter",
    description: "Re-apply a screener to the entire collection, removing rejects.",
    color: "#10b981",
    fields: [
      { name: "screener", label: "Screener block", type: "string", required: true },
    ],
  },
  {
    type: "ReinforceGraph",
    label: "Reinforce Graph",
    category: "transform",
    description: "Rescue high-pagerank rejected papers via a loose screener.",
    color: "#ec4899",
    fields: [
      { name: "screener", label: "Rescue screener", type: "string", required: true },
      { name: "metric", label: "Metric", type: "select", default: "pagerank", options: ["pagerank"] },
      { name: "top_n", label: "Top N", type: "number", default: 30 },
      { name: "percentile_floor", label: "Percentile floor", type: "number", default: 0.9 },
    ],
  },
  {
    type: "Cluster",
    label: "Cluster",
    category: "transform",
    description: "Run a clustering algorithm and store results.",
    color: "#ec4899",
    fields: [
      { name: "store_as", label: "Store as", type: "string", required: true, placeholder: "e.g. forward_topics" },
      { name: "algorithm", label: "Algorithm config", type: "json", default: { type: "walktrap", n_communities: 5 } },
      { name: "naming", label: "Naming config", type: "json", default: { mode: "both", n_keywords: 10, n_representative: 5 } },
      { name: "drop_noise", label: "Drop noise cluster", type: "boolean", default: false },
    ],
  },
  {
    type: "MergeDuplicates",
    label: "Merge Duplicates",
    category: "transform",
    description: "Detect and merge preprint/published duplicates.",
    color: "#ec4899",
    fields: [
      { name: "title_threshold", label: "Title threshold", type: "number", default: 0.95 },
      { name: "semantic_threshold", label: "Semantic threshold", type: "number", default: 0.98 },
      { name: "year_window", label: "Year window", type: "number", default: 1 },
      { name: "use_embeddings", label: "Use embeddings", type: "boolean", default: true },
    ],
  },
  {
    type: "HumanInTheLoop",
    label: "Human in the Loop",
    category: "control",
    description: "Interactive screener-quality check with balanced sampling.",
    color: "#6366f1",
    fields: [
      { name: "enabled", label: "Enabled", type: "boolean", default: false },
      { name: "k", label: "Sample size", type: "number", default: 10 },
      { name: "include_accepted", label: "Include accepted", type: "boolean", default: true },
      { name: "include_rejected", label: "Include rejected", type: "boolean", default: true },
      { name: "balance_by_filter", label: "Balance by filter", type: "boolean", default: true },
    ],
  },
  {
    type: "Parallel",
    label: "Parallel",
    category: "control",
    description: "Broadcast signal to N branches and union outputs.",
    color: "#6366f1",
    fields: [],
  },
  {
    type: "Finalize",
    label: "Finalize",
    category: "output",
    description: "Write collection JSON, BibTeX, and GraphML.",
    color: "#ef4444",
    fields: [],
  },
]

export const STEP_TYPE_MAP = Object.fromEntries(
  STEP_TYPES.map((s) => [s.type, s])
) as Record<string, StepTypeDef>

export const CATEGORIES = [
  { key: "seed", label: "Seeds" },
  { key: "expand", label: "Expand" },
  { key: "filter", label: "Filter / Rerank" },
  { key: "transform", label: "Transform" },
  { key: "control", label: "Control Flow" },
  { key: "output", label: "Output" },
] as const
