// Leiden community detection over the citation graph (network space).
// Each PAPERS row carries its Leiden community as `lc`.
//
// Communities are the CITATION-space instantiation of the same "group" object the
// topic model produces in semantic space: same card vocabulary, same drill anatomy,
// same hover/click-to-canvas behaviour. Only the provenance, the C##/T## code and
// the colour sampling differ.
export const LEIDEN = [{"id":0,"name":"Autonomous Chemistry Labs","desc":"Running synthesis experiments and mining materials literature with robotic platforms.","kw":["materials","protein","graph","chemistry","extraction","molecule","transformer","nanocrystals"],"n":57,"cx":630.28,"cy":549.58},{"id":1,"name":"Tool-Using Science Agents","desc":"Equipping models with biology and chemistry tools, then benchmarking their reasoning.","kw":["protein","gene","cell","chemical","biological","ai agents","scientific reasoning","annotation"],"n":54,"cx":725.25,"cy":543.89},{"id":2,"name":"Idea Generation & Review","desc":"Generating novel hypotheses and papers, then judging their novelty against humans.","kw":["ideas","iclr","llm-generated","peer","ideation","aspr","innovation","reviewers"],"n":52,"cx":761.55,"cy":444.9},{"id":3,"name":"ML Engineering Agents","desc":"Benchmarking agents that run data-science and machine-learning experiments end to end.","kw":["data science","ai agents","web","dr","genai","kaggle","analytical","autonomous data"],"n":49,"cx":683.92,"cy":249.28},{"id":4,"name":"Clinical AI Assistants","desc":"Building multimodal medical assistants and simulating clinicians or study subjects.","kw":["clinical","medical","reports","inductive","data science","visual","rules","social"],"n":37,"cx":671.35,"cy":641.43},{"id":5,"name":"Systematic Review Triage","desc":"Drafting Boolean searches and screening studies to speed up evidence synthesis.","kw":["screening","extraction","pubtator","sr","academic","topic","health","evidence"],"n":31,"cx":470.23,"cy":793},{"id":6,"name":"Evolved Algorithm Design","desc":"Evolving programs and heuristics to crack open math and optimization problems.","kw":["evolutionary","heuristics","combinatorial","alphaevolve","mathematical","refinement","domain knowledge","optimizers"],"n":30,"cx":469.57,"cy":272.57},{"id":7,"name":"Formal Theorem Proving","desc":"Searching Lean proofs and autoformalizing informal mathematics into checkable code.","kw":["formal","theorem","mathematical","proofs","lean","expert iteration","geometry","proving"],"n":27,"cx":131.37,"cy":493.32},{"id":8,"name":"Omics Foundation Models","desc":"Pretraining foundation models on single-cell, genome and protein sequence data.","kw":["single-cell","scgpt","protein","delivery","cells","biological","foundation model","sequencing"],"n":14,"cx":842.02,"cy":546.21},{"id":9,"name":"Biomedical Research Agent","desc":"One agent executing biomedical protocols and analyses without preset templates.","kw":["biomni","protocols","systematic benchmarking strong","artificial intelligence agent","biomedical artificial","predefined templates systematic","language model reasoning","analysis molecular cloning-without"],"n":1,"cx":828.57,"cy":506.68},{"id":10,"name":"Lab-In-The-Loop Discovery","desc":"Coupling hypothesis generation to wet-lab assays that confirmed a retinal target.","kw":["robin","automating hypothesis generation","macular degeneration","phagocytosis","candidates iterative lab-in-the-loop","confirmed vitro","hypothesis generation data","capable fully automating"],"n":1,"cx":838.35,"cy":480.76},{"id":11,"name":"Empirical Software Writing","desc":"Generating research code that beats expert baselines on measurable score metrics.","kw":["expert-level","quality metric","empirical","prevention cdc","forecasting","ideas external sources","language model combined","scientific progress"],"n":1,"cx":823.97,"cy":342.05}];

// Same 12 hues as TOPIC_COLORS in topic-viz.js, rotated by 5 (community i takes
// TOPIC_COLORS[(i*5+5) % 12]) — reads as the same palette family, a different
// partition, and C01 never lands on T01's ink.
export const COMMUNITY_COLORS = [
  'var(--v-green-dot)',
  'color-mix(in oklab, var(--v-blue) 55%, var(--fg))',
  'var(--v-rose-dot)',
  'color-mix(in oklab, var(--v-gold) 58%, var(--fg))',
  'var(--v-gold-dot)',
  'var(--v-indigo)',
  'color-mix(in oklab, var(--v-green) 55%, var(--fg))',
  'var(--v-blue-dot)',
  'color-mix(in oklab, var(--v-rose) 58%, var(--fg))',
  'var(--v-purple-dot)',
  'color-mix(in oklab, var(--v-teal) 55%, var(--fg))',
  'var(--v-teal-dot)'
];
export const PERIPHERAL_COLOR = 'color-mix(in oklab, var(--muted2) 52%, var(--canvas))';
export const communityColor = c => c < 0 ? PERIPHERAL_COLOR : COMMUNITY_COLORS[c % COMMUNITY_COLORS.length];

// ⚠ WIRING NOTE FOR CLAUDE CODE / BACKEND INTEGRATION
// Leiden (like Louvain/Infomap) routinely emits singleton and near-singleton
// communities — this run has three of size 1. Do NOT render those as ordinary
// community cards: fold every community with size <= PERIPHERAL_MAX into ONE
// synthetic "Peripheral" group (id PERIPHERAL_ID) that carries the union of their
// papers. The UI does this generically off the size field, so it stays correct for
// any run: a run with 9 singletons yields one Peripheral card of 9 papers, a run
// with none yields no Peripheral card at all. Never hard-code this run's count.
export const PERIPHERAL_MAX = 1;
export const PERIPHERAL_ID = -2;
