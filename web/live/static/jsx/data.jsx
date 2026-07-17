/* eslint-disable */
// Sample domain data for the CiteClaw WebUI prototype.

const SEED_PAPERS = [
  { id: "s1", title: "Highly accurate protein structure prediction with AlphaFold",
    authors: "Jumper et al.", year: 2021, venue: "Nature", cites: 24891, starred: true },
  { id: "s2", title: "Accurate prediction of protein structures and interactions using a three-track neural network",
    authors: "Baek et al.", year: 2021, venue: "Science", cites: 3412, starred: true },
  { id: "s3", title: "Evolutionary-scale prediction of atomic-level protein structure",
    authors: "Lin et al.", year: 2023, venue: "Science", cites: 1807, starred: true },
  { id: "s4", title: "ColabFold: making protein folding accessible to all",
    authors: "Mirdita et al.", year: 2022, venue: "Nat Methods", cites: 2954, starred: false },
  { id: "s5", title: "Uni-Fold: an open-source platform for developing protein folding models",
    authors: "Li et al.", year: 2023, venue: "bioRxiv", cites: 87, starred: false },
  { id: "s6", title: "OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms",
    authors: "Ahdritz et al.", year: 2024, venue: "Nat Methods", cites: 312, starred: false },
  { id: "s7", title: "Improved protein structure refinement guided by deep learning based accuracy estimation",
    authors: "Hiranuma et al.", year: 2021, venue: "Nat Comms", cites: 418, starred: false },
  { id: "s8", title: "ProtTrans: toward understanding the language of life through self-supervised learning",
    authors: "Elnaggar et al.", year: 2022, venue: "IEEE TPAMI", cites: 1144, starred: false },
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

// The pipeline: 4 connected nodes. Screener nodes carry a filter TREE
// (Sequential / Parallel / Any / Not / Route + leaves), not a flat list.
const INITIAL_PIPELINE = [
  { id: "n1", kind: "seed",   name: "Seed set",         localId: "SED-01",
    config: { query: "protein structure prediction", years: "2019-2025", maxSeeds: 42 },
    screener: null },
  { id: "n2", kind: "fwd",    name: "Forward screener", localId: "FWD-02",
    config: { maxCitations: 200 },
    screener: {
      id: "f1", kind: "Sequential",
      children: [
        { id: "f1a", kind: "YearFilter",     params: { min: 2019, max: 2025 } },
        { id: "f1b", kind: "AbstractKeywordFilter",
          params: { match: "substring",
                    formula: "(ml | bio) & !erratum",
                    keywords: { ml: "machine learning", bio: "biology", erratum: "erratum" } } },
        { id: "f1c", kind: "SimilarityFilter",
          params: { threshold: 0.025,
                    measures: [
                      { kind: "RefSim" },
                      { kind: "CitSim", pass_if_cited_at_least: 200 },
                      { kind: "SemanticSim", embedder: "s2" },
                    ] } },
        { id: "f1d", kind: "LLMFilter",
          params: { scope: "title_abstract",
                    formula: "(q_ml | q_stats) & !q_survey",
                    queries: {
                      q_ml: "the paper proposes a new ML/DL method",
                      q_stats: "the paper proposes a new statistical method",
                      q_survey: "the paper is a pure survey or review",
                    } } },
      ]
    } },
  { id: "n3", kind: "bwd",    name: "Backward screener", localId: "BWD-03",
    config: {},
    screener: {
      id: "g1", kind: "Sequential",
      children: [
        { id: "g1a", kind: "YearFilter",     params: { min: 2018, max: 2025 } },
        { id: "g1b", kind: "CitationFilter", params: { beta: 30 } },
        { id: "g1c", kind: "Any",
          children: [
            { id: "g1c1", kind: "VenueKeywordFilter",
              params: { match: "starts_with",
                        formula: "nat | sci | cell",
                        keywords: { nat: "Nature", sci: "Science", cell: "Cell" } } },
            { id: "g1c2", kind: "LLMFilter",
              params: { scope: "title_abstract",
                        formula: "q_method",
                        queries: { q_method: "the paper proposes a new method" } } },
          ] },
      ]
    } },
  { id: "n4", kind: "rerank", name: "Diversified rerank", localId: "RRK-04",
    config: { metric: "citation", targetN: 500, diversity: "walktrap" },
    screener: null },
];

// ~60 accepted papers so we can show "50 most recent" virtualization comfortably.
function mkAccepted() {
  const titles = [
    "Learning rotamer-invariant representations for protein design",
    "End-to-end differentiable protein language models for binder discovery",
    "Cryptic binding pocket detection via graph attention",
    "Diffusion-based backbone generation for antibody scaffolds",
    "PairFormer: a cross-chain attention module for multimer folding",
    "Scaling inverse folding with structural tokens",
    "Sequence-to-function prediction using protein language models",
    "Distillation of MSA-free folding models onto mobile accelerators",
    "Benchmarking protein foundation models on orphan enzymes",
    "RoseTTAFold All-Atom: unified modeling of protein-ligand complexes",
    "Active learning with folding uncertainty for mutational screening",
    "GraphCodon: predicting codon usage from structural context",
    "Attention rollout reveals evolutionary couplings in AlphaFold",
    "Hallucination of de novo mini-protein binders",
    "Zero-shot prediction of protein-protein interactions with ESM-2",
    "Equivariant diffusion for docking small molecules",
    "Learning the grammar of disordered regions with language models",
    "Structure-conditioned language models for targeted design",
    "Folding with fewer templates: bootstrapped multiple-sequence augmentation",
    "Ligand-aware protein folding through co-evolutionary signals",
  ];
  const authors = [
    "Rao & Watson", "Chowdhury et al.", "Lin, Wang, Liu", "Kovac & Patel",
    "Fernández-García et al.", "Okonkwo, Reed", "Harrington, Zhu",
    "Bustamante et al.", "Nilsson, Ohira", "Varga et al.",
    "Kapoor & Stanton", "Yang, Pereira", "Ghosh et al.", "Anand, Mori",
  ];
  const venues = ["bioRxiv", "Nat Methods", "Science", "Cell", "NeurIPS", "ICML", "PNAS", "Nat Comms"];
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
