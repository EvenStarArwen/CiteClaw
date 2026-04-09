# CiteClaw

**Transparent, high-fidelity literature acquisition for autonomous scientific discovery.**

CiteClaw is a composable snowballing agent that constructs systematic research corpora
from a handful of seed papers. It traverses the global citation graph with a circuit-style
pipeline of rule-based, similarity-based, and LLM-based filter blocks — tiering cheap
checks before expensive LLM screening to deliver survey-grade collections at remarkably
low cost, with full provenance at every step.

In a recent benchmark mapping the evolution of AI4Biology, CiteClaw screened **18,000+
candidates** in 30 minutes and extracted **400+ milestone papers** (2% acceptance rate)
with high precision and recall — at a total cost of **$0.70** using Gemini 3 Flash-Lite.
[Visualization](https://drive.google.com/file/d/14nNVAOmLy8FvEwKcRBwOVRu7tSpZeyFL/view?usp=sharing).

---

## Why CiteClaw

Autonomous scientific discovery with LLM agents hinges on rigorous understanding of
frontier literature. Today, building survey-grade research corpora is a punishing manual
bottleneck, while existing LLM-based tools trade rigor for efficiency, producing opaque,
hallucination-prone knowledge bases that scientists cannot trust for high-stakes work.

CiteClaw is built to be the trustworthy substrate for the next generation of autonomous
research — from hypothesis generation and experiment planning to fully self-driving labs.

Key design principles:

- **Circuit-style composition** — `nn.Sequential`-inspired pipeline of pure signal
  transformers, with a `Parallel` branch primitive for independent exploration.
- **Tiered filtering** — YearFilter → CitationFilter → SimilarityFilter → LLMFilter,
  so expensive LLM calls only see pre-screened candidates.
- **Full provenance** — every paper's acceptance path is tracked; run state, shape
  summaries, and rejection counts are persisted to disk.
- **Community-aware reranking** — graph- or embedding-based clustering can drive
  diversity-aware top-K selection.
- **S2 SPECTER2 out of the box** — semantic similarity and topic modeling read
  Semantic Scholar's precomputed vectors; no embedding backend setup required.
- **Cost-conscious** — batched LLM dispatch, read-through SQLite cache, and
  cheap-first filter ordering push per-paper cost to cents.

---

## Installation

```bash
pip install -e .

# optional: UMAP+HDBSCAN topic modeling
pip install -e '.[topic_model]'

# optional: dev tooling (ruff, mypy, pytest)
pip install -e '.[dev]'
```

Requires Python 3.11+.

### Environment

Set your API keys in a `.env` file (or export them directly):

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
SEMANTIC_SCHOLAR_API_KEY=...   # optional, improves S2 rate limits
```

---

## Quick start

1. Write a `config.yaml` describing reusable filter blocks and a linear list of
   pipeline steps. The repository ships with a worked biology / ML example at
   `config.yaml` — copy it and adapt the topic description, seed papers, and
   screening prompts to your own domain.

2. Run CiteClaw:

```bash
python -m citeclaw -c config.yaml
```

3. Continue a prior run (adds another expansion generation on top of the existing
   collection):

```bash
python -m citeclaw -c config.yaml --continue-from data/
```

4. Annotate the resulting citation graph with LLM-generated node summaries:

```bash
python -m citeclaw annotate data/citation_network.graphml -c config.yaml
```

CLI flags: `--topic`, `--seed`, `--data-dir`, `--max-papers`, `--model`, `-v`,
`--continue-from`.

---

## Architecture at a glance

Two compositional layers, mirroring `nn.Sequential` plus a parallel branch.

| Layer            | Operates on                 | Composition primitives                                                    |
| ---------------- | --------------------------- | ------------------------------------------------------------------------- |
| Pipeline (steps) | `signal: list[PaperRecord]` | top-level Sequential, `Parallel`                                          |
| Filter blocks    | one paper → bool            | `Sequential` (AND), `Any` (OR), `Not` (invert), `Route` (if/elif/else)    |

Every step is a pure signal transformer:

```python
class BaseStep(Protocol):
    name: str
    def run(self, signal: list[PaperRecord], ctx: Context) -> StepResult: ...
```

`ctx.collection` is the cumulative union of every paper ever accepted; only `ReScreen`
removes from it. `Rerank` is **non-destructive** — it filters the signal but never
touches `ctx.collection`. This invariant lets `Parallel` work: one branch can
rerank-then-forward while another sees the original input untouched.

### Pipeline steps

| Step                | Purpose                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------- |
| `LoadSeeds`         | Initialise `ctx.collection` from `seed_papers`. Emits the seeds as the first signal.                 |
| `ResolveSeeds`      | Convert mixed `{title: ...}` / `{paper_id: ...}` seed entries to canonical S2 IDs via `search_match`; optionally pull preprint↔published siblings via `external_ids`. |
| `ExpandForward`     | For every paper in the signal, fetch citers, screen them, add the survivors to collection + signal.  |
| `ExpandBackward`    | Same as ExpandForward but follows references instead of citers.                                       |
| `ExpandBySearch`    | **Expansion family**: iterative meta-LLM search agent designs targeted database queries from a topic + anchor papers, refines per turn, and screens the hits. |
| `ExpandBySemantics` | **Expansion family**: anchor on the input signal, fetch SPECTER2 nearest neighbours via S2 Recommendations API, and screen the hits. |
| `ExpandByAuthor`    | **Expansion family**: rank authors in the input signal by h-index / citation count / collab-graph degree, pull each top-K author's papers, and screen them. |
| `Rerank`            | Score-based top-K (with optional cluster-aware diversity). Non-destructive — only filters the signal. |
| `ReScreen`          | Apply a screener block to the entire `ctx.collection` (minus seeds), removing rejected papers.        |
| `Cluster`           | Run a clusterer over the signal once, store the `ClusterResult` in `ctx.clusters[<store_as>]`.        |
| `MergeDuplicates`   | Detect and merge preprint↔published duplicates via DOI/ArXiv ID + title sim + SPECTER2 cosine.        |
| `Parallel`          | Broadcast the signal to N branches, run each independently, union outputs by `paper_id`.             |
| `Finalize`          | Write `literature_collection.json` / `.bib`, `citation_network.graphml`, `run_state.json`.            |

**Expansion family.** `ExpandBySearch`, `ExpandBySemantics`, and `ExpandByAuthor` compose at the same level as `ExpandForward` / `ExpandBackward` — users mix all five freely in YAML pipelines. Each ExpandBy* step is anchored on its input *signal* (not an upstream citation edge), so the source-less `FilterContext` they pass to the screener carries `source=None`. All built-in atoms and measures tolerate this case; insert a `Rerank` (with diversity) before any ExpandBy* step to control which papers the agent uses as anchors.

### Filter blocks

| Type                | Purpose                                                                                |
| ------------------- | -------------------------------------------------------------------------------------- |
| `Sequential`        | AND of `layers:` (plural). Short-circuits on first reject.                             |
| `Any`               | OR of `layers:` (plural). Short-circuits on first pass.                                |
| `Not`               | Invert a single child block specified as `layer:` (singular).                          |
| `Route`             | if/elif/else dispatch over `routes:`.                                                  |
| `SimilarityFilter`  | Max of normalized scores from `measures:` list (RefSim / CitSim / SemanticSim).        |
| `YearFilter`        | Pass if `year` is in `[min, max]`.                                                     |
| `CitationFilter`    | Pass if citation count is high enough relative to `beta` and paper age.                |
| `LLMFilter`         | Batched LLM screening; `scope:` is `title` / `title_abstract` / `venue`. Single-prompt or Boolean formula mode. |

### Clusterers

| Type          | Mechanism                                                                              |
| ------------- | -------------------------------------------------------------------------------------- |
| `walktrap`    | igraph `community_walktrap` over the citation graph; targets a fixed `n_communities`.  |
| `louvain`     | igraph `community_multilevel` (modularity maximisation); auto-determines count.        |
| `topic_model` | UMAP + HDBSCAN over S2 SPECTER2 embeddings. BERTopic-inspired, no bertopic dep.        |

Cluster naming (label / summary / keywords / representative papers) is filled in by an
algorithm-agnostic post-processor using c-TF-IDF and/or LLM calls — the same code labels
a walktrap community and a topic_model topic identically.

---

## Example config

```yaml
screening_model: "gemini-2.5-flash-lite"
data_dir: "data_bio"
topic_description: "..."
max_papers_total: 500
seed_papers:
  - paper_id: "DOI:10.1038/s41586-021-03819-2"

blocks:
  year_layer:   {type: YearFilter, min: 2018, max: 2026}
  cit_base:     {type: CitationFilter, beta: 30}

  similarity:
    type: SimilarityFilter
    threshold: 0.025
    measures:
      - {type: RefSim}
      - {type: CitSim, pass_if_cited_at_least: 200}
      - {type: SemanticSim, embedder: s2}

  topic_llm:
    type: LLMFilter
    scope: title_abstract
    formula: "(q_ml | q_stats) & !q_survey"
    queries:
      q_ml:     "the paper proposes a new ML/DL method"
      q_stats:  "the paper proposes a new statistical method"
      q_survey: "the paper is a pure survey or review"

  forward_screener:
    type: Sequential
    layers: [year_layer, cit_base, similarity, topic_llm]

pipeline:
  - step: LoadSeeds
  - step: ExpandForward
    max_citations: 30
    screener: forward_screener
  - step: ExpandBackward
    screener: forward_screener
  - step: Cluster
    store_as: topics
    algorithm: {type: topic_model, min_cluster_size: 5}
    naming: {mode: both, n_keywords: 10, n_representative: 5}
  - step: Rerank
    metric: pagerank
    k: 100
    diversity: {cluster: topics}
  - step: Finalize
```

See `config.yaml` for a full production configuration with `Route`, `Parallel`,
and `Not` in use.

---

## Output artifacts

Every run writes into `data_dir/`:

- `literature_collection.json` (+ `.expN` for continuation runs)
- `literature_collection.bib`
- `citation_network.graphml` — rich node/edge attributes, cluster assignments, and
  LLM-generated cluster labels (ready for Gephi color-coding by topic)
- `collaboration_network.graphml` — undirected author co-authorship graph
- `run_state.json` — full run state for continuation
- `cache.db` — SQLite read-through cache for S2 metadata
- `shape_summary.txt` — PyTorch-summary-style pipeline shape table

---

## Module layout

```
src/citeclaw/
  config.py            globals + raw blocks/pipeline + lazy build
  context.py           Context dataclass
  pipeline.py          run_pipeline(ctx) — iterates the built Step list
  __main__.py          CLI entry: `python -m citeclaw [-c …] [annotate …]`
  cache.py             SQLite read-through cache for S2 metadata
  filters/             Filter Protocol, builder, runner, blocks, atoms, measures
  screening/           BooleanFormula DSL + batched concurrent LLM dispatch
  cluster/             walktrap / louvain / topic_model + representation
  search/              apply_local_query — pure regex + range predicate filter
  agents/              iterative_search — meta-LLM agent for ExpandBySearch
  steps/               LoadSeeds, ResolveSeeds, ExpandForward/Backward,
                       ExpandBySearch/Semantics/Author, Rerank, ReScreen,
                       Cluster, MergeDuplicates, Parallel, Finalize
  rerank/              metrics + cluster_diverse_top_k
  clients/             s2, llm (OpenAI / Gemini / Stub), embeddings
  output/              json / bibtex / graphml writers
  prompts/             screening, annotation, topic_naming, search_refine
```

---

## Development

```bash
pytest                 # run the full test suite
ruff check src tests   # lint
mypy src               # type-check
```

Tests that exercise optional `topic_model` extras are skipped if the extras aren't
installed.
