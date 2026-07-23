/* eslint-disable */
/* Sample paper records for the PaperCard demo. Shape mirrors what the CiteClaw
   web app passes in (a subset of a Semantic Scholar paper record). */

window.PAPER_CARD_DEMO = {
  // A well-formed paper: 2-line title, full venue + byline.
  normal: {
    id: "demo-1",
    title: "ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design",
    venue: "Neural Information Processing Systems",
    authors: "Pascal Notin et al.",
    year: 2023,
    cites: 283,
    depth: 1,
    seed: false,
  },

  // A one-line title — proves the card stays the same height as a 2-line one.
  shortTitle: {
    id: "demo-2",
    title: "Short Title Here",
    venue: "bioRxiv",
    authors: "M. Dias et al.",
    year: 2023,
    cites: 24,
    depth: 2,
    seed: false,
  },

  // Missing venue → the venue row shows "—" instead of collapsing.
  noVenue: {
    id: "demo-3",
    title: "A protein language model for exploring viral fitness landscapes",
    venue: "",
    authors: "Jumpei Ito et al.",
    year: 2025,
    cites: 39,
    depth: 3,
    seed: false,
  },

  // Long author list — clips on its own row so the year · cites line survives.
  longAuthors: {
    id: "demo-4",
    title: "Evolutionary-scale prediction of atomic level protein structure with a language model",
    venue: "Science",
    authors: "Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu et al.",
    year: 2022,
    cites: 5070,
    depth: 1,
    seed: true,
  },

  // A rejected candidate — reject mark, category chip, and the human reason.
  rejected: {
    id: "demo-5",
    title: "AutoSurvey: Large Language Models Can Automatically Write Surveys",
    venue: "Neural Information Processing Systems",
    authors: "Yidong Wang et al.",
    year: 2024,
    cites: 127,
    category: "llm__anon.layer2",
    reason: "Not a primary methods paper — reads as a survey of existing approaches.",
    seed: false,
  },

  // An off-network paper (in the collection, filtered out of the graph view).
  offnet: {
    id: "demo-6",
    title: "BioSeq-BLM: a platform for analyzing DNA, RNA and protein sequences",
    venue: "Nucleic Acids Research",
    authors: "Hong-Liang Li et al.",
    year: 2021,
    cites: 228,
    depth: 4,
    seed: false,
  },
};

// A short list used for the interactive "click to select" specimen.
window.PAPER_CARD_DEMO_LIST = [
  window.PAPER_CARD_DEMO.normal,
  window.PAPER_CARD_DEMO.shortTitle,
  window.PAPER_CARD_DEMO.longAuthors,
  window.PAPER_CARD_DEMO.noVenue,
];
