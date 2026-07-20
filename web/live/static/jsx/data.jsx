/* eslint-disable */
// Sample domain data for the CiteClaw WebUI prototype.

const SEED_PAPERS = [
  { id: "s1", title: "GraphCast: learning skillful medium-range global weather forecasting",
    authors: "Lam et al.", year: 2023, venue: "Science", cites: 1893, starred: true },
  { id: "s2", title: "Accurate medium-range global weather forecasting with 3D neural networks",
    authors: "Bi et al.", year: 2023, venue: "Nature", cites: 2471, starred: true },
  { id: "s3", title: "FourCastNet: a global data-driven high-resolution weather model",
    authors: "Pathak et al.", year: 2022, venue: "arXiv", cites: 1204, starred: true },
  { id: "s4", title: "ClimaX: a foundation model for weather and climate",
    authors: "Nguyen et al.", year: 2023, venue: "ICML", cites: 642, starred: false },
  { id: "s5", title: "Aurora: a foundation model of the atmosphere",
    authors: "Bodnar et al.", year: 2024, venue: "arXiv", cites: 189, starred: false },
  { id: "s6", title: "Neural general circulation models for weather and climate",
    authors: "Kochkov et al.", year: 2024, venue: "Nature", cites: 274, starred: false },
  { id: "s7", title: "GenCast: diffusion-based ensemble forecasting for medium-range weather",
    authors: "Price et al.", year: 2024, venue: "Nature", cites: 208, starred: false },
  { id: "s8", title: "Deep learning for twelve-hour precipitation nowcasting",
    authors: "Andrychowicz et al.", year: 2023, venue: "arXiv", cites: 331, starred: false },
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

// The default pipeline: the self-driving-labs / AI-scientist snowball
// config, translated from its original YAML. One screener ("venue_gate")
// routes every paper by venue family — top journals skip the citation bar,
// preprints face a high one (β=30), everything else a moderate one (β=10) —
// then all branches apply the same abstract-keyword prefilter + title LLM.
// The screener is embedded per expansion step (fresh ids per copy); the
// repeated forward/backward/rerank steps in rounds 2–3 are LINKED copies of
// the round-1 originals, so editing one screener updates the whole pipeline.
//
// Differences from the original YAML, on purpose:
//  · ExpandBySearch is omitted (the CLI agent is a placeholder for now —
//    the step is still selectable from the add-step menu).
//  · Cluster+Rerank({cluster: X}) pairs are collapsed into diversified
//    reranks with INLINE louvain (functionally equivalent here; only the
//    exported cluster_<name> graph attributes are lost).
//  · VenuePreset is approximated with VenueIn substrings.
//  · The unused "agent" keyword is included in the abstract formula.

// One fresh copy of the shared screener tree; `p` prefixes every node id so
// each embedding is independent.
function _sdlScreener(p) {
  const ABS_EXPR = '"self-driving" | "self driving" | autonomous | automated | ' +
    '"ai scientist" | "ai-scientist" | "co-scientist" | "large language model" | LLM | laborator | agent';
  const absKw = (id) => ({ id, kind: "AbstractKeywordFilter",
    params: { match: "substring", expression: ABS_EXPR } });
  const titleLlm = (id) => ({ id, kind: "LLMFilter",
    params: { scope: "title", formula: "q1",
      queries: { q1: "The paper is about self-driving / autonomous laboratories or AI scientists or AI agents for research / scientific discovery." },
      model: "", effort: "" } });
  const cit = (id, beta) => ({ id, kind: "CitationFilter", params: { beta, exemption_years: -1 } });
  const seq = (id, children) => ({ id, kind: "Sequential", children });
  return seq(p + "root", [
    { id: p + "gate", kind: "Route",
      routes: [
        { id: p + "gr1", if: { kind: "VenueIn", values: ["Nature", "Science", "Cell"] },
          pass_to: seq(p + "b1", [absKw(p + "b1k"), titleLlm(p + "b1l")]) },
        { id: p + "gr2", if: { kind: "VenueIn", values: ["rxiv", "SSRN", "Research Square", "preprint"] },
          pass_to: seq(p + "b2", [cit(p + "b2c", 30), absKw(p + "b2k"), titleLlm(p + "b2l")]) },
      ],
      else: seq(p + "b3", [cit(p + "b3c", 10), absKw(p + "b3k"), titleLlm(p + "b3l")]) },
  ]);
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
