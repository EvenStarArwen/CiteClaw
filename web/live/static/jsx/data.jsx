/* eslint-disable */
// Sample domain data for the CiteClaw WebUI prototype.

// Six landmark AI-scientist / self-driving-lab papers, all starred so the
// default pipeline can run out of the box. `id` is the real Semantic Scholar
// paperId (only starred seeds are submitted, keyed on it).
const SEED_PAPERS = [
  { id: "6fe3779fe5f2e9402abdd08ad8db41a0f13a99eb", title: "Autonomous chemical research with large language models",
    authors: "Boiko et al.", year: 2023, venue: "Nature", cites: 1094, starred: true },
  { id: "390c607723e9767a9a873ded3bc1f66808215e8e", title: "Accelerating scientific discovery with Co-Scientist",
    authors: "Gottweis et al.", year: 2025, venue: "Nature", cites: 359, starred: true },
  { id: "ddaac3c134274b3f40f8bcbad705d80b00af3198", title: "A multi-agent system for automating scientific discovery",
    authors: "Ghareeb et al.", year: 2026, venue: "Nature", cites: 52, starred: true },
  { id: "8b8854e236ad0504672670d9ae643d6418ce4685", title: "An AI system to help scientists write expert-level empirical software",
    authors: "Aygün et al.", year: 2025, venue: "arXiv", cites: 39, starred: true },
  { id: "7e55d8701785818776323b4147cb13354c820469", title: "PaperQA: Retrieval-Augmented Generative Agent for Scientific Research",
    authors: "Lála et al.", year: 2023, venue: "arXiv", cites: 196, starred: true },
  { id: "9e57dda195973c4b6c81386b1cc44595ecfd4697", title: "AutoSurvey: Large Language Models Can Automatically Write Surveys",
    authors: "Wang et al.", year: 2024, venue: "NeurIPS", cites: 126, starred: true },
];

const BLOCK_CATALOG = [
  { cat: "Sources", n: 1, items: [
    { kind: "seed",   name: "Seed set",          hint: "Semantic Scholar query",  icon: "flag" },
  ]},
  { cat: "Expanders", n: 2, items: [
    { kind: "fwd",    name: "Forward screener",  hint: "Follows outgoing cites",  icon: "arrow-right" },
    { kind: "bwd",    name: "Backward screener", hint: "Walks references",        icon: "arrow-left" },
  ]},
  { cat: "Rerankers", n: 2, items: [
    { kind: "rerank", name: "Diversified rerank", hint: "MMR with λ control",     icon: "sliders-horizontal" },
    { kind: "rsc",    name: "Rescreener",         hint: "LLM accept / reject",    icon: "filter" },
  ]},
  { cat: "Sinks", n: 1, items: [
    { kind: "sink",   name: "Accepted sink",     hint: "Final export target",     icon: "inbox" },
  ]},
];

// The default pipeline: an AI-scientist / self-driving-labs snowball over the
// six seed papers above. Forward + backward snowball across three rounds; the
// shared screener is a simple sequence — a 2023–2026 year window, a citation
// bar (β=30, scaled by age, applied to the current year too), an abstract
// keyword prefilter, then a title LLM screen and an abstract LLM screen.
// The screener is embedded per expansion step (fresh ids per copy); the
// repeated forward/backward/rerank steps in rounds 2–3 are LINKED copies of
// the round-1 originals, so editing one screener updates the whole pipeline.
// ExpandBySearch is omitted (the CLI agent is a placeholder for now — the step
// is still selectable from the add-step menu); Cluster+Rerank pairs are
// collapsed into diversified reranks with inline louvain.

// One fresh copy of the shared screener tree; `p` prefixes every node id so
// each embedding is independent.
function _sdlScreener(p) {
  const ABS_EXPR =
    "(discover* OR scientist* OR scientific OR self-driving OR research OR lab* OR " +
    "autonomous OR automat* OR design* OR reasoning) AND " +
    '(agent* OR "large language model*" OR LLM OR AI OR "artificial intelligence")';
  const TITLE_PROMPT =
    "The paper is about self-driving / autonomous laboratories or AI scientists or " +
    "AI agents for automating/aiding scientific research / scientific discovery.";
  const ABS_PROMPT =
    "The paper introduces a new self-driving / autonomous laboratory or AI scientist " +
    "system or AI agentic system for automating/aiding scientific research / scientific discovery.";
  return {
    id: p + "root", kind: "Sequential", children: [
      { id: p + "yr",  kind: "YearFilter",     params: { min: 2023, max: 2026 } },
      { id: p + "cit", kind: "CitationFilter", params: { beta: 30, exemption_years: -1, curve: "linear" } },
      { id: p + "kw",  kind: "KeywordFilter",  params: { scope: "abstract", match: "whole_word", expression: ABS_EXPR } },
      { id: p + "tl",  kind: "LLMFilter",      params: { scope: "title",          formula: "q1", queries: { q1: TITLE_PROMPT }, model: "", effort: "" } },
      { id: p + "al",  kind: "LLMFilter",      params: { scope: "title_abstract", formula: "q1", queries: { q1: ABS_PROMPT },   model: "", effort: "" } },
    ],
  };
}

const INITIAL_PIPELINE = [
  { id: "n1", kind: "seed", name: "Seed set", localId: "SED-01",
    config: { query: "", years: "2019-2025", maxSeeds: 42 },
    screener: null },
  // Round 1: forward + backward snowball from the seeds, both screened.
  { id: "par1", kind: "parallel", branches: [
    [ { id: "n2", kind: "fwd", name: "Forward screener", localId: "FWD-02",
        config: { maxCitations: 200 }, screener: _sdlScreener("f") } ],
    [ { id: "n3", kind: "bwd", name: "Backward screener", localId: "BWD-03",
        config: {}, screener: _sdlScreener("g") } ],
  ] },
  // Round 2: cluster-diverse pagerank top-20 anchors a second forward hop;
  // the backward branch walks everything again. fwd/bwd are linked copies.
  { id: "par2", kind: "parallel", branches: [
    [ { id: "n4", kind: "rerank", name: "Diversified rerank", localId: "RRK-04",
        config: { metric: "pagerank", targetN: 20, diversity: "louvain" }, screener: null },
      { id: "n5", kind: "fwd", name: "Forward screener", localId: "FWD-05",
        config: {}, screener: null, syncOf: "n2", synced: true } ],
    [ { id: "n6", kind: "bwd", name: "Backward screener", localId: "BWD-06",
        config: {}, screener: null, syncOf: "n3", synced: true } ],
  ] },
  // Round 3: same shape again — every step a linked copy.
  { id: "par3", kind: "parallel", branches: [
    [ { id: "n7", kind: "rerank", name: "Diversified rerank", localId: "RRK-07",
        config: {}, screener: null, syncOf: "n4", synced: true },
      { id: "n8", kind: "fwd", name: "Forward screener", localId: "FWD-08",
        config: {}, screener: null, syncOf: "n2", synced: true } ],
    [ { id: "n9", kind: "bwd", name: "Backward screener", localId: "BWD-09",
        config: {}, screener: null, syncOf: "n3", synced: true } ],
  ] },
];

// ~60 accepted papers so we can show "50 most recent" virtualization comfortably.
function mkAccepted() {
  const titles = [
    "Physics-informed neural networks for atmospheric dynamics",
    "Graph neural networks for global precipitation nowcasting",
    "Diffusion models for probabilistic weather ensembles",
    "Transformer architectures for seasonal climate prediction",
    "Downscaling climate projections with generative models",
    "Data assimilation with learned observation operators",
    "Neural operators for the shallow-water equations",
    "Self-supervised pretraining on ERA5 reanalysis data",
    "Benchmarking data-driven forecasts against ECMWF IFS",
    "Spherical CNNs for global weather fields",
    "Uncertainty quantification in neural weather models",
    "Learning subgrid parameterizations for climate models",
    "Attention maps reveal atmospheric teleconnection patterns",
    "Extreme-event forecasting with machine-learning emulators",
    "Zero-shot nowcasting with foundation weather models",
    "Emulating radiative transfer with neural networks",
    "Fusing satellite and radar data for nowcasting",
    "Ensemble calibration via conditional normalizing flows",
    "Long-range forecasting with deep state-space models",
    "Hybrid physics-ML general circulation modeling",
  ];
  const authors = [
    "Rao & Watson", "Chowdhury et al.", "Lin, Wang, Liu", "Kovac & Patel",
    "Fernández-García et al.", "Okonkwo, Reed", "Harrington, Zhu",
    "Bustamante et al.", "Nilsson, Ohira", "Varga et al.",
    "Kapoor & Stanton", "Yang, Pereira", "Ghosh et al.", "Anand, Mori",
  ];
  const venues = ["arXiv", "npj Clim Atmos Sci", "Science", "Geophys Res Lett", "NeurIPS", "ICML", "Mon Weather Rev", "Nat Comms"];
  const out = [];
  for (let i = 0; i < 60; i++) {
    const score = 0.62 + Math.random() * 0.36;
    const t = titles[i % titles.length];
    const a = authors[i % authors.length];
    const v = venues[i % venues.length];
    const y = 2019 + (i % 7);
    const depth = 1 + (i % 3);
    out.push({
      id: "p" + i,
      title: t,
      authors: a, year: y, venue: v,
      score, depth,
      cites: Math.floor(10 + Math.random() * 900),
      addedAt: Date.now() - i * 4200,
    });
  }
  return out.sort((a, b) => b.score - a.score);
}

// Run-mode data starts empty in the live app — it is populated by the
// backend event stream (see live-store.jsx). mkAccepted/mkNetwork above are
// retained but unused.
const ACCEPTED_PAPERS = [];

// Run progress starts empty; the backend derives real steps at run start.
const PROGRESS_STEPS = [];

// Network viz — seeded positions + real paper IDs so clicking a node picks a row.
// Coordinates here are *initial* positions in [0..1]²; the canvas renderer runs
// a light force-directed relaxation on top before drawing.
function mkNetwork() {
  const rng = (() => { let s = 17; return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; }; })();
  const papers = window.ACCEPTED_PAPERS || [];
  const nodes = [];
  const N_SEEDS = 3;
  const N_SAT = 48;

  // 3 seeds — triangle spread across the canvas, widely apart
  const seedPos = [
    { x: 0.28, y: 0.38 },
    { x: 0.68, y: 0.32 },
    { x: 0.52, y: 0.72 },
  ];
  seedPos.forEach((pos, i) => {
    const p = papers[i] || { id: "sp" + i, year: 2022, cites: 500 };
    nodes.push({
      id: "s" + i, paperId: p.id, seed: true,
      x: pos.x, y: pos.y,
      year: p.year, r: 8, cites: p.cites || 400,
    });
  });
  // satellites — cluster around ONE of the three seeds, not all three
  // This creates 3 visible subclusters rather than an omni-linked hairball
  const assign = [];
  for (let i = 0; i < N_SAT; i++) assign.push(i % N_SEEDS);
  // shuffle
  for (let i = assign.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [assign[i], assign[j]] = [assign[j], assign[i]];
  }
  for (let i = 0; i < N_SAT; i++) {
    const p = papers[(i + N_SEEDS) % papers.length] || { id: "n" + i, year: 2020, cites: 30 };
    const parent = seedPos[assign[i]];
    const ring = 0.08 + rng() * 0.14;
    const ang = rng() * Math.PI * 2;
    nodes.push({
      id: "n" + i, paperId: p.id, seed: false,
      parent: assign[i],
      x: parent.x + Math.cos(ang) * ring,
      y: parent.y + Math.sin(ang) * ring,
      year: p.year,
      r: 3.5 + Math.min(4.5, Math.log10(1 + (p.cites || 1)) * 1.2),
      cites: p.cites || 30,
    });
  }
  // edges: each satellite connects to ITS parent seed + occasional sibling
  const edges = [];
  for (let i = N_SEEDS; i < nodes.length; i++) {
    edges.push({ a: i, b: nodes[i].parent });
    if (rng() < 0.18) {
      // link to a sibling in the same cluster
      const siblings = [];
      for (let j = N_SEEDS; j < nodes.length; j++) {
        if (j !== i && nodes[j].parent === nodes[i].parent) siblings.push(j);
      }
      if (siblings.length) {
        edges.push({ a: i, b: siblings[Math.floor(rng() * siblings.length)] });
      }
    }
    // rare cross-cluster bridge
    if (rng() < 0.08) {
      const other = N_SEEDS + Math.floor(rng() * (nodes.length - N_SEEDS));
      if (nodes[other].parent !== nodes[i].parent) edges.push({ a: i, b: other });
    }
  }
  return { nodes, edges };
}
const NETWORK = { nodes: [], edges: [] };

Object.assign(window, {
  SEED_PAPERS, BLOCK_CATALOG, INITIAL_PIPELINE,
  ACCEPTED_PAPERS, PROGRESS_STEPS, NETWORK,
});
