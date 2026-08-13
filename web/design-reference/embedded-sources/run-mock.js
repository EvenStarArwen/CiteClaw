/* =============================================================================
 * run-mock.js — MOCK RUN DATA + REPLAY CLOCK for the Runs page (KnowledgeLab).
 *
 * ⚠️  DEMO DATA. NOT A BACKEND CONTRACT. READ THIS BEFORE WIRING ANYTHING REAL.
 *
 *     Every number, cadence and state transition in this file exists to make a
 *     *demo* legible: a 13-minute engine run replayed in 1 minute, with counts
 *     that tick, steps that light up and a corpus that lands paper by paper.
 *     The shapes here were invented to feed the existing Runs UI, NOT derived
 *     from any API. A session wiring this app to the real backend must NOT take
 *     these decisions as the reference — ask the humans what the engine actually
 *     emits (event stream? polling? per-step deltas or absolute totals?), then
 *     map that onto the panel, replacing this module. If the real API cannot
 *     produce a field below, the field is wrong, not the API.
 *
 * WHY IT EXISTS
 *     The Runs panel used to carry its mock tables inline (step table, run list,
 *     rejection totals, a frozen "5 of 7 steps done" snapshot, hard-coded monitor
 *     numbers in the markup). One demo, no clock. This module is the single place
 *     that mock data for the Runs page lives, so more demo cases can be added by
 *     adding data, not by editing the panel. The previous inline tables are kept
 *     in archive/runs-mock-v1.md.
 *
 * THE CONTRACT (also recorded in design-system.md § Runs mock-data contract)
 *     The module owns NUMBERS and TIMING. The page owns WORDS and PIXELS.
 *     A "run source" is a plain object:
 *       { id, label, elapsed, dur, accepted, rejected, steps[], filters[],
 *         calls{}, tokens{}, history[] }
 *     A "replay" is built from a run source:
 *       makeReplay(source, cfg) -> { duration, source, frameAt(ms) -> frame }
 *     A frame is a pure function of time — no state, no side effects, so the
 *     page can seek, resume after navigating away, or freeze on Stop by simply
 *     asking for a different t. Frame shape is documented at frameAt() below.
 *
 * DATA PROVENANCE
 *     Real output of engine RUN-37 ("AI agents for scientific discovery"):
 *     7 steps, 7 seeds -> 354 accepted / 17,440 rejected, 781s wall clock.
 *     Source: RUN-37 bundle (stats.json, engine/run_state.json, accepted.csv).
 *     The 354 accepted papers themselves live in topic-data.js (field `sc` =
 *     the step that accepted the paper); citation edges in graph-data.js.
 * ========================================================================== */

(function () {
  'use strict';

  // ————————————————————————————————————————————————— canonical run: RUN-37
  // cuts = papers removed by [year, citation, keyword, LLM-title, LLM-abstract,
  // duplicate]; year..LLM-abstract sum to `rej` (duplicates are a de-dup pass,
  // counted in the filter table but not in the run's rejected total).
  var PIPELINE = [
    { code: 'SED-01', name: 'Seed Set', dot: '--v-teal-dot', seeds: true, in: 0, found: 7, kept: 7, rej: 0, cuts: [0, 0, 0, 0, 0, 0], calls: 0, tokens: 0, verb: 'Load seed papers' },
    { code: 'FWD-02', name: 'Forward Searcher', dot: '--v-gold-dot', in: 7, found: 341, kept: 45, rej: 296, cuts: [74, 175, 26, 20, 1, 0], calls: 2, tokens: 17975, verb: 'Fetch citing papers', foundLabel: 'Citing papers found', expandedLabel: 'citing papers' },
    { code: 'BWD-03', name: 'Backward Searcher', dot: '--v-gold-dot', in: 7, found: 171, kept: 8, rej: 163, cuts: [41, 97, 14, 11, 0, 0], calls: 1, tokens: 5505, verb: 'Fetch reference papers', foundLabel: 'References found', expandedLabel: 'references' },
    { code: 'FWD-04', name: 'Forward Searcher', dot: '--v-gold-dot', in: 53, found: 2634, kept: 57, rej: 2577, cuts: [644, 1526, 227, 171, 9, 0], calls: 7, tokens: 28764, verb: 'Fetch citing papers', foundLabel: 'Citing papers found', expandedLabel: 'citing papers' },
    { code: 'BWD-05', name: 'Backward Searcher', dot: '--v-gold-dot', in: 53, found: 2767, kept: 102, rej: 2665, cuts: [666, 1579, 235, 177, 8, 0], calls: 3, tokens: 55982, verb: 'Fetch reference papers', foundLabel: 'References found', expandedLabel: 'references' },
    { code: 'BWD-06', name: 'Backward Searcher', dot: '--v-gold-dot', in: 159, found: 5223, kept: 62, rej: 5161, cuts: [1290, 3057, 455, 342, 17, 0], calls: 5, tokens: 40459, verb: 'Fetch reference papers', foundLabel: 'References found', expandedLabel: 'references' },
    { code: 'FWD-07', name: 'Forward Searcher', dot: '--v-gold-dot', in: 159, found: 6651, kept: 73, rej: 6578, cuts: [1644, 3894, 580, 435, 25, 4], calls: 298, tokens: 46236, verb: 'Fetch citing papers', foundLabel: 'Citing papers found', expandedLabel: 'citing papers' }
  ];

  // Filter table: display order + which `cuts` slot feeds each row.
  var FILTERS = [
    { label: 'Citation', col: '--f2d', cut: 1 },
    { label: 'Year', col: '--f1d', cut: 0 },
    { label: 'Keyword', col: '--f3d', cut: 2 },
    { label: 'LLM title', col: '--f6d', cut: 3 },
    { label: 'LLM abstract', col: '--f7d', cut: 4 },
    { label: 'Duplicate', col: '--muted2', cut: 5 }
  ];

  // Token split per step, from the run's LLM ledger (title 18.6% / abstract 60.2%
  // / database search 0% / output 21.2% of 194,921 tokens).
  var TOKEN_MIX = { title: 0.1855, abs: 0.6023, db: 0, out: 0.2122 };

  var RUN37 = {
    id: 'RUN-37',
    label: 'AI agents for scientific discovery',
    elapsed: 781,            // engine seconds, wall clock
    dur: '13:01',
    accepted: 354,
    rejected: 17440,
    steps: PIPELINE,
    filters: FILTERS,
    tokenMix: TOKEN_MIX,
    calls: { graph: 316, reco: 0 },
    tokens: { total: 194921, budget: 250000 }
  };

  // ————————————————————————————————————————————————— run list (demo history)
  // The live row is created by the page when a run starts (rnNewLiveRun); the
  // first one takes NEXT_RUN_NO, each further run increments.
  var NEXT_RUN_NO = 38;
  var RUN_LIBRARY = [
    { id: 'RUN-37', dot: '--success', outcome: 'Completed', date: 'today', acc: 354, rej: 17440, done: 7, dur: '13:01', grp: 'Today' },
    { id: 'RUN-06', dot: '--success', outcome: 'Completed', date: 'yesterday', acc: 214, rej: 1490, done: 7, dur: '22:04', grp: 'Yesterday' },
    { id: 'RUN-05', dot: '--warning', outcome: 'Stopped', date: 'Jul 22', acc: 96, rej: 388, done: 4, dur: '09:30', grp: 'Jul 2026' },
    { id: 'RUN-04', dot: '--success', outcome: 'Completed', date: 'Jul 20', acc: 178, rej: 1104, done: 7, dur: '18:47', grp: 'Jul 2026' },
    { id: 'RUN-03', dot: '--success', outcome: 'Completed', date: 'Jul 18', acc: 132, rej: 846, done: 7, dur: '21:18', grp: 'Jul 2026' },
    { id: 'RUN-02', dot: '--error', outcome: 'Failed', date: 'Jul 15', acc: 12, rej: 74, done: 2, dur: '01:44', grp: 'Jul 2026' },
    { id: 'RUN-01', dot: '--success', outcome: 'Completed', date: 'Jul 11', acc: 88, rej: 502, done: 7, dur: '16:02', grp: 'Jul 2026' }
  ];

  // ————————————————————————————————————————————————— replay configuration
  // DEMO PACING, not engine behaviour: 1 minute for the whole run, spread
  // faithfully — each parallel wave gets time in proportion to the papers it
  // handled, so FWD-07 (6,651 fetched) dominates the way it did in reality.
  var REPLAY = {
    totalMs: 60000,
    waveBase: 400,   // constant weight per wave so the 7-seed wave stays watchable
    stepBase: 120,   // constant weight per step inside a wave
    tickMs: 320,     // how often the page asks for a frame
    // fractions of a step's duration, in order
    stageMix: [['fetch', 0.30], ['meta', 0.10], ['abs', 0.16], ['basic', 0.10], ['llmT', 0.20], ['llmA', 0.14]],
    seedMix: [['fetch', 0.50], ['meta', 0.25], ['abs', 0.25]]
  };

  var clamp01 = function (x) { return x < 0 ? 0 : x > 1 ? 1 : x; };
  var seg = function (t, s) { return clamp01((t - s.start) / Math.max(1, s.end - s.start)); };

  // Waves = consecutive steps that consume the same input, i.e. one parallel
  // block of the engine (the same rule the panel uses to draw its forks). The wave
  // states the structure; the steps inside it still PLAY one at a time.
  function waveSplit(steps) {
    var w = [];
    steps.forEach(function (s, i) {
      var prev = w[w.length - 1];
      if (prev && !s.seeds && !steps[prev[0]].seeds && steps[prev[0]].in === s.in) prev.push(i);
      else w.push([i]);
    });
    return w;
  }

  function buildTimeline(steps, cfg) {
    var waves = waveSplit(steps);
    var wW = waves.map(function (w) {
      return cfg.waveBase + Math.max.apply(null, w.map(function (i) { return steps[i].found; }));
    });
    var tot = wW.reduce(function (a, b) { return a + b; }, 0);
    var out = steps.map(function () { return null; });
    var t = 0;
    waves.forEach(function (w, wi) {
      var dur = cfg.totalMs * wW[wi] / tot;
      // Steps inside a wave run BACK TO BACK, never overlapping: execution is serial
      // (design-system.md § Run semantics — exactly one step may be active), while the
      // wave is what states the parallel structure. The wave's total time is unchanged;
      // it is split between its steps in proportion to the papers each handled.
      var sw = w.map(function (i) { return cfg.stepBase + steps[i].found; });
      var swTot = sw.reduce(function (a, b) { return a + b; }, 0);
      var at = t;
      w.forEach(function (i, wj) {
        var s = steps[i];
        var sd = dur * sw[wj] / swTot;
        var mix = s.seeds ? cfg.seedMix : cfg.stageMix;
        var a = at, stages = {};
        mix.forEach(function (m) { stages[m[0]] = { start: a, end: a + sd * m[1] }; a += sd * m[1]; });
        out[i] = { i: i, wave: wi, start: at, end: at + sd, dur: sd, stages: stages, keys: mix.map(function (m) { return m[0]; }) };
        at += sd;
      });
      t += dur;
    });
    return { duration: cfg.totalMs, steps: out, waves: waves };
  }

  /* ---------------------------------------------------------------------------
   * makeReplay(source, cfg) -> { duration, source, cfg, timeline, frameAt(ms) }
   *
   * frameAt(ms) returns a frozen snapshot — pure, cheap, repeatable:
   * {
   *   t, progress, finished,
   *   elapsed,                    engine seconds to show on the clock
   *   accepted, rejected,         cumulative corpus + screening totals
   *   stepsDone,
   *   steps: [{ code, phase:'queued'|'active'|'done', p,
   *             fetched, enrichedMeta, enrichedAbs,
   *             stages: { fetch|meta|abs|basic|llmT|llmA: {p, state, seen, pass} },
   *             kept                       papers accepted by this step so far
   *          }],
   *   filters: [{ label, col, rej, cum }],   cum = running total in display order
   *   calls:  { graph, reco, total, rate },
   *   tokens: { title, abs, db, out, total, budget, pct }
   * }
   * Papers trickle in during a step's screening window (basic -> LLM abstract),
   * which is a DEMO CHOICE: the real engine only knows a verdict once abstract
   * screening returns. Ask the backend how it streams partial results.
   * ------------------------------------------------------------------------- */
  function makeReplay(source, cfg) {
    cfg = Object.assign({}, REPLAY, cfg || {});
    var steps = source.steps;
    var tl = buildTimeline(steps, cfg);
    var mix = source.tokenMix || TOKEN_MIX;

    function frameAt(t) {
      t = Math.max(0, Math.min(tl.duration, t));
      var accTot = 0, rejTot = 0, done = 0;
      var fRej = source.filters.map(function () { return 0; });
      var calls = 0, tkT = 0, tkA = 0, tkD = 0, tkO = 0;

      var out = steps.map(function (s, i) {
        var L = tl.steps[i];
        var phase = t < L.start ? 'queued' : t >= L.end ? 'done' : 'active';
        var p = clamp01((t - L.start) / Math.max(1, L.dur));
        var g = L.stages;
        var pf = seg(t, g.fetch), pm = seg(t, g.meta), pa = seg(t, g.abs);
        var st = { fetch: { p: pf }, meta: { p: pm }, abs: { p: pa } };
        var basicOut = s.found - s.cuts[0] - s.cuts[1] - s.cuts[2];
        var tOut = basicOut - s.cuts[3];
        var aOut = tOut - s.cuts[4];
        var keptNow;

        calls += Math.round(s.calls * pf);

        if (s.seeds) {
          keptNow = Math.round(s.kept * p);
          st.fetch.seen = s.found; st.fetch.pass = Math.round(s.found * pf);
          st.meta.seen = s.found; st.meta.pass = Math.round(s.found * pm);
          st.abs.seen = s.found; st.abs.pass = Math.round(s.found * pa);
        } else {
          var pb = seg(t, g.basic), pt = seg(t, g.llmT), pl = seg(t, g.llmA);
          st.basic = { p: pb, seen: Math.round(s.found * pb), pass: Math.round(basicOut * pb) };
          st.llmT = { p: pt, seen: Math.round(basicOut * pt), pass: Math.round(tOut * pt) };
          st.llmA = { p: pl, seen: Math.round(tOut * pl), pass: Math.round(aOut * pl) };
          st.fetch.seen = s.in; st.fetch.pass = Math.round(s.found * pf);
          st.meta.seen = s.found; st.meta.pass = Math.round(s.found * pm);
          st.abs.seen = s.found; st.abs.pass = Math.round(s.found * pa);
          // rejections land with the filter that removed them
          fRej[1] += Math.round(s.cuts[0] * pb);   // Year
          fRej[0] += Math.round(s.cuts[1] * pb);   // Citation
          fRej[2] += Math.round(s.cuts[2] * pb);   // Keyword
          fRej[3] += Math.round(s.cuts[3] * pt);   // LLM title
          fRej[4] += Math.round(s.cuts[4] * pl);   // LLM abstract
          fRej[5] += phase === 'done' ? s.cuts[5] : 0;   // Duplicate (de-dup pass, at the end of the step)
          // accepted papers trickle across the screening window
          var sp = clamp01((t - g.basic.start) / Math.max(1, g.llmA.end - g.basic.start));
          keptNow = Math.round(s.kept * sp);
          tkT += s.tokens * mix.title * pt;
          tkA += s.tokens * mix.abs * pl;
          tkD += s.tokens * mix.db * pt;
          tkO += s.tokens * mix.out * (pt * 0.45 + pl * 0.55);
        }
        Object.keys(st).forEach(function (k) {
          st[k].state = st[k].p >= 1 ? 'done' : st[k].p > 0 ? 'active' : 'pending';
          st[k].p = Math.round(st[k].p * 100);
        });
        accTot += keptNow;
        if (phase === 'done') done++;
        return { code: s.code, name: s.name, dot: s.dot, seeds: !!s.seeds, phase: phase,
          p: Math.round(p * 100), keys: L.keys, stages: st, kept: keptNow };
      });

      // the run's rejected total counts screening removals, not the de-dup pass
      rejTot = fRej[0] + fRej[1] + fRej[2] + fRej[3] + fRej[4];
      var cum = 0;
      var filters = source.filters.map(function (f, i) { cum += fRej[i]; return { label: f.label, col: f.col, rej: fRej[i], cum: cum }; });
      // The on-screen clock ticks in REAL seconds: a demo compresses the WORK,
      // never the wall clock the user watches. Engine time is only used internally
      // so per-second rates stay plausible.
      var elapsed = Math.round(t / 1000);
      var engineSec = source.elapsed * (t / tl.duration);
      var tkTot = tkT + tkA + tkD + tkO;
      return {
        t: t, progress: t / tl.duration, finished: t >= tl.duration,
        elapsed: elapsed, accepted: accTot, rejected: rejTot, stepsDone: done,
        steps: out, filters: filters,
        calls: { graph: calls, reco: 0, total: calls, rate: calls / Math.max(1, engineSec) },
        tokens: { title: Math.round(tkT), abs: Math.round(tkA), db: Math.round(tkD), out: Math.round(tkO),
          total: Math.round(tkTot), budget: source.tokens.budget, pct: tkTot / source.tokens.budget }
      };
    }

    return { duration: tl.duration, source: source, cfg: cfg, timeline: tl, frameAt: frameAt };
  }

  // ————————————————————————— topology-based suggestions (Refine · Add papers)
  // Top-5 candidates per measure over the one-hop citation neighborhood of the
  // accepted corpus (46,304 candidates). Source: CiteClaw recommendation dump
  // (uploads/CiteClaw-Recommendations). `cites` = accepted-paper pids the
  // candidate cites (mapped onto topic-data.js PAPERS for canvas previews).
  // a = "Cites the corpus" (score = accepted papers cited, of 354)
  // b = "Shared bibliography" (score = ref-overlap weight / sqrt(own ref count))
  var RECO = {
   "total": 354,
   "candidates": 46304,
   "a": [
    {
     "id": "1493c93ed733df86b06733a968505054303606dd",
     "rank": 1,
     "score": 53,
     "ti": "Evolving Roles of LLMs in Scientific Innovation: Assistant, Collaborator, Scientist, and Evaluator",
     "au": "Haoxuan Zhang · Ruochi Li · Yang Zhang · Ting Xiao · Jiangping Chen · Junhua Ding · Haihua Chen",
     "ve": "arXiv.org",
     "yr": 2025,
     "ci": 13,
     "rc": 207,
     "nc": 53,
     "hits": 86,
     "share": 113,
     "best": {
      "ti": "Transforming Science with Large Language Models: A Survey on AI-assisted Scientific Discovery, Experimentation, Content Generation, and Evaluation",
      "n": 24
     },
     "pdf": 1,
     "ab": "Large language models (LLMs) are increasingly used in scientific research and discovery, supporting tasks ranging from literature retrieval and synthesis to hypothesis generation, autonomous experimentation, and research evaluation. Existing surveys often conflate scientific research with scientific discovery and typically organize systems by domain, task, or autonomy level alone. In this survey, we propose a four-role framework for understanding LLMs in scientific innovation: Assistant, Collaborator, Scientist, and Evaluator. The framework integrates three complementary dimensions: autonomy level, cognitive function, and scientific innovation, to distinguish research-oriented support from frontier-oriented discovery. We review representative methods, benchmarks, and evaluation practices for each role, examining their capabilities, limitations, and human oversight requirements. Across the literature, Assistant systems are comparatively mature in retrieval and synthesis but remain unreliable in open-ended applications; Collaborator systems expand the space of candidate hypotheses yet struggle with novelty-grounding trade-offs; Scientist systems increasingly automate research workflows but face reliability and safety bottlenecks; and Evaluator systems support review and verification while remaining weak in novelty assessment. We argue that progress in AI for science depends not only on model capability, but also on evaluation, oversight, accountability, and institutional integration.",
     "cites": [
      "002bf0720404e5dc6bf43eff64f116ec755b405f",
      "04c987757c27c9da9e9ccd4c114bbe1d142a46cf",
      "0885471c0215b3c0d31c82518066913f7f738128",
      "0e5a9fa2314d69a6666a0beb48d92b772dfa1d16",
      "110f5dc6d5bfe67138d64c261d6851c727021d1f",
      "235992701c6e70872f5292fd5818d5c5719063de",
      "25c646b3424d4bd8f969e8d37d180e124d91a86e",
      "2a87cbb8179f537dc1fa811b83e0b06323223c1f",
      "2e2b5d3589f31cdc5ca0bcd918b22794eb4fe5e4",
      "354dcdebf3f8b5feeed5c62090e0bc1f0c28db06",
      "38d45590794eda4d6d3bb1f15c293dd6911fac50",
      "394924896e24c9b086d96d0958dae07f54ff9452",
      "395c978221a21ee47c84a40a2ef11fb4d012fca1",
      "39983a80f111f7f6e793f02c5725a14bca76b32d",
      "3c0bd90ffb0c26e3fdedb45574b17b2fd5f97cee",
      "3dd5ad34012164c4ec9c571a12cc6a7561683dea",
      "45b7c7448768b51b6dbd9b76495c9cd9d110bd91",
      "48c83799530dc523ee01e6c1c40ad577d5c10a16",
      "591f5644aede42a2447e48eedb112c70e80089af",
      "594773309239331e6afcd2892136c5233383d3a5",
      "5aea5c8b536380c5ad1d42108c2c6767622318ee",
      "5b3e4513796d86ddf6ae8dfee6793af3d45cffe8",
      "5bbc93573edc73b360f76ae3ba87fe698188ab7c",
      "5dbd9f9fd231863dda17d9e3caff2edb99a113e2",
      "60c8a127e6ae8c8e21dd7edfc187ff7f0d9ae2bd",
      "6a5c45ec0e89a722388751768a70f04b8fecc905",
      "7ac14eb265086ccb7bf94d956b20ebc20144cbde",
      "7c44b7fdcec2e517799f6c54f6ba42bf1a89d2e6",
      "804b5775524552b27d5ffd6a3ab82d0f5c5f1011",
      "80a0b76dedc4c3e3d365bbaececcd44a996eb38b",
      "8c44c3fa1d3cbc9cbde4cb028f598f1999d539d7",
      "8cedeb11139eab187e43414fd7097c5d578dad7c",
      "92c82a51ad13c361d052987694cf93d6a72d5789",
      "9348b7b95982d0a675a767e92c23647aa6915a94",
      "95d19f8ede34cd712a09ae3b86bed2b338bd9a48",
      "9678af027022328f143ee2b627ee93ba313a4e7c",
      "9e57dda195973c4b6c81386b1cc44595ecfd4697",
      "b73e2b7ffbb00b03b795dc8b5e732a8859470249",
      "b8ee0b5322382807e687c95cc87b059d3f348495",
      "c62c32a776cd2b5513c1110ad7d45f9877174489",
      "d0cd8b45949b959c316a3ed75a4683d0a70b1aa9",
      "d892b1335ad37c4773f98e8282b73d1460a9f2bf",
      "dbbcdb281ed6aa646af0402172cf8a7cfda85d5c",
      "ddc1899e59a8e4fda60f5a175fef710a63abcef9",
      "de28e4cbee455fba136072a0f922c49698b0fb58",
      "e366b47c98cb922d7d5b536ca776dde8333b4bdc",
      "e4eb81ad222ba047770d5a90bdd7406c138c6126",
      "e51dff31f56847c16de3a2f2682d16109537b96e",
      "e87d34032cc2a77a7711a651296a8f51c9474929",
      "ed32e5bb11a9e5e2f03ea805782e8670a6e10efd",
      "f2c7f69177d7972510e7a026c7fa20c26905a3df",
      "fa5f9aa1cb6f97654ca8e6d279ceee1427a87e68",
      "fd30d3189b3bc3295ddad05ac1f683ce41f5e9cb"
     ]
    },
    {
     "id": "c298aea9c6b0e284a786987ccc9bce9df6b6d16a",
     "rank": 2,
     "score": 48,
     "ti": "Autonomous Agents for Scientific Discovery: Orchestrating Scientists, Language, Code, and Physics",
     "au": "Lianhao Zhou · Hongyi Ling · Cong Fu · Yepeng Huang · Michael Sun · Wendi Yu · Xiaoxuan Wang · Xiner Li · Xingyu Su · Junkai Zhang · Xiusi Chen · Chengxin Liang · Xiaofeng Qian · Heng Ji · Wei Wang · M. Zitnik · Shuiwang Ji",
     "ve": "arXiv.org",
     "yr": 2025,
     "ci": 14,
     "rc": 250,
     "nc": 48,
     "hits": 124,
     "share": 208,
     "best": {
      "ti": "Towards Scientific Intelligence: A Survey of LLM-based Scientific Agents",
      "n": 20
     },
     "pdf": 1,
     "ab": "Computing has long served as a cornerstone of scientific discovery. Recently, a paradigm shift has emerged with the rise of large language models (LLMs), introducing autonomous systems, referred to as agents, that accelerate discovery across varying levels of autonomy. These language agents provide a flexible and versatile framework that orchestrates interactions with human scientists, natural language, computer language and code, and physics. This paper presents our view and vision of LLM-based scientific agents and their growing role in transforming the scientific discovery lifecycle, from hypothesis discovery, experimental design and execution, to result analysis and refinement. We critically examine current methodologies, emphasizing key innovations, practical achievements, and outstanding limitations. Additionally, we identify open research challenges and outline promising directions for building more robust, generalizable, and adaptive scientific agents. Our analysis highlights the transformative potential of autonomous agents to accelerate scientific discovery across diverse domains.",
     "cites": [
      "002bf0720404e5dc6bf43eff64f116ec755b405f",
      "01e863776846ebd1a9a7acc4a9ca24217f953aa2",
      "0e5a9fa2314d69a6666a0beb48d92b772dfa1d16",
      "1104bec9e7a0a3d9dba341ba8005f1b7350bc876",
      "11355ce57ccdcaeed7941fc95516838d543c5cb3",
      "1601699d52b474efe80ea9a41e66b3fc9d6ac3ad",
      "1951e288ade1a453c3f1bb544f3e5690a6d177c1",
      "1c259361caa85c2d95a7d04e5e42fa98693da85b",
      "1eda67ce15645acda94d98a6a8fa1c6e75fc75c2",
      "25c646b3424d4bd8f969e8d37d180e124d91a86e",
      "2e2b5d3589f31cdc5ca0bcd918b22794eb4fe5e4",
      "354dcdebf3f8b5feeed5c62090e0bc1f0c28db06",
      "38d45590794eda4d6d3bb1f15c293dd6911fac50",
      "390c607723e9767a9a873ded3bc1f66808215e8e",
      "39983a80f111f7f6e793f02c5725a14bca76b32d",
      "3dd5ad34012164c4ec9c571a12cc6a7561683dea",
      "44d16a076c00ecada3d425203377e4ec951c4ed0",
      "454274b92a27bc5d97c1e80accc78095ae991cf0",
      "5182f0746ba2d420f943e91e65235650a896bf5a",
      "51b7b3ad7645a69e3c1c80cae69473b8bd472f67",
      "5350c0e883610e7332ff558e7d3ea18a1a2cc86e",
      "5487397061f02b34f66e2228eee303d8985a43c4",
      "58bccfcd425a9d45ce96c3b3e21c33e290b1ee2d",
      "591f5644aede42a2447e48eedb112c70e80089af",
      "5f781329fb77a7711404361597490a4f2cd18c6a",
      "67dc27725a92510066492ef5df423aad002ae486",
      "6fe3779fe5f2e9402abdd08ad8db41a0f13a99eb",
      "78a31e6b7c12ab21dd4b2e355b68daa8a45f94cc",
      "797597040167622a126c3206bbf94459238e2c19",
      "7bbe76e09d2aab075f006675355ebaa4020463a2",
      "804b5775524552b27d5ffd6a3ab82d0f5c5f1011",
      "81e8bbc9e7828e8ffbd04f6dc2adae59d965153d",
      "8284823c6d1d4b478e4536e1c8ab3a904d653ebb",
      "8a816b4d4ee63804615dba39269948fda63b1c21",
      "8cedeb11139eab187e43414fd7097c5d578dad7c",
      "95154330d74be6c79ac45e97b38ef0038e2b8e2a",
      "9678af027022328f143ee2b627ee93ba313a4e7c",
      "975b23dd7f82ea53ccf9a0c6a5d65368fb19c3d8",
      "a970be54c4df5f04c3fe65b7414e0c2879c55909",
      "bba1d4de76b8330bea5f73ddeb99da4e01c02e16",
      "bc4c630ad6caff3c7e8a0dc578690aab69a4feb9",
      "bf27c81faad0dd21bd39c45ee5e85300577fda66",
      "c62c32a776cd2b5513c1110ad7d45f9877174489",
      "cd91e012efbf737144d1855368767339ff9aca08",
      "dcfd4a696b7adfae8b95a8badecaa526164750e5",
      "dd565137ec79da38bf2d01319bba195c66733212",
      "e51dff31f56847c16de3a2f2682d16109537b96e",
      "fb970ce4383a78ad52c641bc38815d78ad737aad"
     ]
    },
    {
     "id": "cdb34c0092a767848ca1de6fa7e3a6b822585fa4",
     "rank": 3,
     "score": 43,
     "ti": "From Hypothesis to Publication: A Comprehensive Survey of AI-Driven Research Support Systems",
     "au": "Zekun Zhou · Xiaocheng Feng · Lei Huang · Xiachong Feng · Ziyun Song · Ruihan Chen · Liang Zhao · Weitao Ma · Yuxuan Gu · Baoxin Wang · Dayong Wu · Guoping Hu · Ting Liu · Bing Qin",
     "ve": "Conference on Empirical Methods in Natural Language Processing",
     "yr": 2025,
     "ci": 10,
     "rc": 293,
     "nc": 43,
     "hits": 137,
     "share": 227,
     "best": {
      "ti": "Transforming Science with Large Language Models: A Survey on AI-assisted Scientific Discovery, Experimentation, Content Generation, and Evaluation",
      "n": 44
     },
     "pdf": 1,
     "ab": "Research is a fundamental process driving the advancement of human civilization, yet it demands substantial time and effort from researchers. In recent years, the rapid development of artificial intelligence (AI) technologies has inspired researchers to explore how AI can accelerate and enhance research. To monitor relevant advancements, this paper presents a systematic review of the progress in this domain. Specifically, we organize the relevant studies into three main categories: hypothesis formulation, hypothesis validation, and manuscript publication. Hypothesis formulation involves knowledge synthesis and hypothesis generation. Hypothesis validation includes the verification of scientific claims, theorem proving, and experiment validation. Manuscript publication encompasses manuscript writing and the peer review process. Furthermore, we identify and discuss the current challenges faced in these areas, as well as potential future directions for research. Finally, we also offer a comprehensive overview of existing benchmarks and tools across various domains that support the integration of AI into the research process. We hope this paper serves as an introduction for beginners and fosters future research. Resources have been made publicly available at https://github.com/zkzhou126/AI-for-Research.",
     "cites": [
      "002bf0720404e5dc6bf43eff64f116ec755b405f",
      "0f6439b95aed189908c628f1103dbcef8541cc2e",
      "110f5dc6d5bfe67138d64c261d6851c727021d1f",
      "1601699d52b474efe80ea9a41e66b3fc9d6ac3ad",
      "281188f43de819c848a2f985144958ea9a4e0fcb",
      "2a87cbb8179f537dc1fa811b83e0b06323223c1f",
      "33161a5a9b5dcb635b5a97475e6a6209a69ada7d",
      "354dcdebf3f8b5feeed5c62090e0bc1f0c28db06",
      "3648515cc35b517cdf60331cc4870e24616f9939",
      "394924896e24c9b086d96d0958dae07f54ff9452",
      "3dd5ad34012164c4ec9c571a12cc6a7561683dea",
      "4c913d59d150fe7581386b87dfd9f90448a9adee",
      "5aea5c8b536380c5ad1d42108c2c6767622318ee",
      "5bbc93573edc73b360f76ae3ba87fe698188ab7c",
      "610684735aa3a6bad1cc28c777944dbfe959fea5",
      "62729cff7dda7614f648a84e8967076d8878a5ff",
      "63aa9598da11da04d044ecf7211b2055b7a1775c",
      "687397ab709d98476205ed36ecd3f1e01ee7dbae",
      "6a4501fefaf73261dc180ff86b52208679f3fb9c",
      "7de36d6b14aadc8cdb6ad1340b9ca64b15375bca",
      "7dfbe4882188dbac938306be93bc58b34450b87f",
      "92c82a51ad13c361d052987694cf93d6a72d5789",
      "9678af027022328f143ee2b627ee93ba313a4e7c",
      "9e57dda195973c4b6c81386b1cc44595ecfd4697",
      "b40df4b273f255b3cb5639e220c8ab7b1bdb313e",
      "b8ee0b5322382807e687c95cc87b059d3f348495",
      "c2d574f7c6a9e3bafe396ecb4ab639179d6fd92c",
      "c77d908ba29567445a9a4ad1bd4461d441cce174",
      "c890b28a001c885d1f7aa05f5d24ead9bf6ae058",
      "c97d6c8b8c0ad57da712cd4de215c64b05904cfd",
      "d0cd8b45949b959c316a3ed75a4683d0a70b1aa9",
      "d8be118ba41df62ca92e49b1f757d53404393529",
      "da17ec7b2867c3e02951c5598936b891ebe865ce",
      "dcfd4a696b7adfae8b95a8badecaa526164750e5",
      "ddc1899e59a8e4fda60f5a175fef710a63abcef9",
      "e366b47c98cb922d7d5b536ca776dde8333b4bdc",
      "e4eb81ad222ba047770d5a90bdd7406c138c6126",
      "e51dff31f56847c16de3a2f2682d16109537b96e",
      "ec64fd9b864a5b525e70f4bf03088522facc0c47",
      "ed7c053b38074003bfb947c78373f69598b960b8",
      "f2209eb5ac6747319a29b87dedabb97770be3243",
      "fa5f9aa1cb6f97654ca8e6d279ceee1427a87e68",
      "fd30d3189b3bc3295ddad05ac1f683ce41f5e9cb"
     ]
    },
    {
     "id": "7c3038d15957585bcc68eee0bd715a6991dc4798",
     "rank": 4,
     "score": 42,
     "ti": "AutoResearch AI: Towards AI-Powered Research Automation for Scientific Discovery",
     "au": "Guiyao Tie · Jiawen Shi · D. Song · Yixiao Huang · Ziji Sheng · Xueyang Zhou · Daizong Liu · Pan Zhou · Yongchao Chen · Ran Xu · Lifang He · Qingsong Wen · Manling Li · Cong Lu · Shuai Li · Pengtao Xie · Yixuan Yuan · Rui Meng · Lei Xing · Lichao Sun · Caiming Xiong · Philip S. Yu · Jianfeng Gao",
     "ve": "arXiv.org",
     "yr": 2026,
     "ci": 4,
     "rc": 158,
     "nc": 42,
     "hits": 70,
     "share": 137,
     "best": {
      "ti": "From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery",
      "n": 21
     },
     "pdf": 1,
     "ab": "Scientific research is being reshaped by AI systems that move beyond isolated assistance toward longer-horizon workflows spanning literature grounding, hypothesis generation, experimentation, validation, reporting, and revision. This shift marks a transition from task-level AI for science to workflow-level research automation. Yet current systems remain fragmented, differing in autonomy, domain scope, execution environment, validation mechanism, and human oversight, while still struggling with evidence preservation, reproducibility, weak-direction rejection, provenance tracking, cross-domain robustness, and accountable scientific closure. This survey examines these developments through AutoResearch, defined as the developmental spectrum of AI-powered scientific workflow automation. Within it, Vibe Research denotes the human-steered region of prompt-based assistance and human-verified execution, whereas emerging AI-led systems coordinate larger portions of the discovery loop without achieving robust autonomy. We analyze how research systems redistribute control, evidence, execution, validation, and accountability across workflows and organize the field around five workflow conditions: literature and research grounding; hypothesis formation and planning; experimentation and tool use; feedback, validation, and review; and reporting and knowledge communication. We further synthesize AI scientist systems, mixed-initiative co-research frameworks, benchmarks, domain deployments, and open-source infrastructures. Finally, we propose five evaluation dimensions--novelty, validity, impact, reliability, and provenance--and show that AutoResearch autonomy is domain-conditioned, being more credible in structured, executable, and rapidly verifiable settings but limited in embodied, delayed, heterogeneous, ethical, or institutionally accountable contexts.",
     "cites": [
      "0fc30d552b48b12817ba6ec4dfa9571b2a262b8d",
      "1104bec9e7a0a3d9dba341ba8005f1b7350bc876",
      "110f5dc6d5bfe67138d64c261d6851c727021d1f",
      "11355ce57ccdcaeed7941fc95516838d543c5cb3",
      "1951e288ade1a453c3f1bb544f3e5690a6d177c1",
      "274c5e690df6ca44c62430a39f3f8e4a6d9e5b79",
      "2dbec8020b9111f1c70e90b112e6318f7d02426c",
      "33161a5a9b5dcb635b5a97475e6a6209a69ada7d",
      "390c607723e9767a9a873ded3bc1f66808215e8e",
      "394924896e24c9b086d96d0958dae07f54ff9452",
      "39983a80f111f7f6e793f02c5725a14bca76b32d",
      "3cabe88c2ff476d35cd179525c7dfe474b4b75bb",
      "4155072db4186e6f5d171507f3a1225c76188949",
      "45b7c7448768b51b6dbd9b76495c9cd9d110bd91",
      "48c83799530dc523ee01e6c1c40ad577d5c10a16",
      "4c913d59d150fe7581386b87dfd9f90448a9adee",
      "4ce81182c4069867e59cb167985345d01d6ba13a",
      "51b7b3ad7645a69e3c1c80cae69473b8bd472f67",
      "52be71605c60d326381a4ddfa1fb476912709504",
      "591f5644aede42a2447e48eedb112c70e80089af",
      "594773309239331e6afcd2892136c5233383d3a5",
      "5b3e4513796d86ddf6ae8dfee6793af3d45cffe8",
      "5bbc93573edc73b360f76ae3ba87fe698188ab7c",
      "6c9288c709db52adf7df7a1e32df5683edbbdcc2",
      "6fe3779fe5f2e9402abdd08ad8db41a0f13a99eb",
      "7ac14eb265086ccb7bf94d956b20ebc20144cbde",
      "7bbe76e09d2aab075f006675355ebaa4020463a2",
      "80a0b76dedc4c3e3d365bbaececcd44a996eb38b",
      "8cedeb11139eab187e43414fd7097c5d578dad7c",
      "92c82a51ad13c361d052987694cf93d6a72d5789",
      "9678af027022328f143ee2b627ee93ba313a4e7c",
      "b8ee0b5322382807e687c95cc87b059d3f348495",
      "d0cd8b45949b959c316a3ed75a4683d0a70b1aa9",
      "d892b1335ad37c4773f98e8282b73d1460a9f2bf",
      "dbbcdb281ed6aa646af0402172cf8a7cfda85d5c",
      "dcfd4a696b7adfae8b95a8badecaa526164750e5",
      "ddaac3c134274b3f40f8bcbad705d80b00af3198",
      "e4eb81ad222ba047770d5a90bdd7406c138c6126",
      "ed32e5bb11a9e5e2f03ea805782e8670a6e10efd",
      "f2209eb5ac6747319a29b87dedabb97770be3243",
      "fa5f9aa1cb6f97654ca8e6d279ceee1427a87e68",
      "fd30d3189b3bc3295ddad05ac1f683ce41f5e9cb"
     ]
    },
    {
     "id": "d41e2dd043a7843eb7fced7306cfe3373ef3cff9",
     "rank": 5,
     "score": 42,
     "ti": "How Far Are AI Scientists from Changing the World?",
     "au": "Qiujie Xie · Yixuan Weng · Minjun Zhu · Fuchen Shen · Shulin Huang · Zhen Lin · Jiahui Zhou · Zilan Mao · Zijie Yang · Linyi Yang · Jian Wu · Yue Zhang",
     "ve": "arXiv.org",
     "yr": 2025,
     "ci": 14,
     "rc": 200,
     "nc": 42,
     "hits": 123,
     "share": 268,
     "best": {
      "ti": "Transforming Science with Large Language Models: A Survey on AI-assisted Scientific Discovery, Experimentation, Content Generation, and Evaluation",
      "n": 34
     },
     "pdf": 1,
     "ab": "The emergence of large language models (LLMs) is propelling automated scientific discovery to the next level, with LLM-based Artificial Intelligence (AI) Scientist systems now taking the lead in scientific research. Several influential works have already appeared in the field of AI Scientist systems, with AI-generated research papers having been accepted at the ICLR 2025 workshop, suggesting that a human-level AI Scientist capable of uncovering phenomena previously unknown to humans, may soon become a reality. In this survey, we focus on the central question: How far are AI scientists from changing the world and reshaping the scientific research paradigm? To answer this question, we provide a prospect-driven review that comprehensively analyzes the current achievements of AI Scientist systems, identifying key bottlenecks and the critical components required for the emergence of a scientific agent capable of producing ground-breaking discoveries that solve grand challenges. We hope this survey will contribute to a clearer understanding of limitations of current AI Scientist systems, showing where we are, what is missing, and what the ultimate goals for scientific AI should be.",
     "cites": [
      "002bf0720404e5dc6bf43eff64f116ec755b405f",
      "0e5a9fa2314d69a6666a0beb48d92b772dfa1d16",
      "1104bec9e7a0a3d9dba341ba8005f1b7350bc876",
      "110f5dc6d5bfe67138d64c261d6851c727021d1f",
      "2dbec8020b9111f1c70e90b112e6318f7d02426c",
      "33161a5a9b5dcb635b5a97475e6a6209a69ada7d",
      "390c607723e9767a9a873ded3bc1f66808215e8e",
      "394924896e24c9b086d96d0958dae07f54ff9452",
      "395c978221a21ee47c84a40a2ef11fb4d012fca1",
      "39983a80f111f7f6e793f02c5725a14bca76b32d",
      "3dd5ad34012164c4ec9c571a12cc6a7561683dea",
      "45b7c7448768b51b6dbd9b76495c9cd9d110bd91",
      "4c913d59d150fe7581386b87dfd9f90448a9adee",
      "52be71605c60d326381a4ddfa1fb476912709504",
      "591f5644aede42a2447e48eedb112c70e80089af",
      "594773309239331e6afcd2892136c5233383d3a5",
      "5aea5c8b536380c5ad1d42108c2c6767622318ee",
      "5bbc93573edc73b360f76ae3ba87fe698188ab7c",
      "60c8a127e6ae8c8e21dd7edfc187ff7f0d9ae2bd",
      "62729cff7dda7614f648a84e8967076d8878a5ff",
      "63aa9598da11da04d044ecf7211b2055b7a1775c",
      "6eeaf1e7b3be81eb1f44f80906757964e6183fec",
      "7647290e260d75dcc9f1090a9b8281574dd2afbe",
      "7bbe76e09d2aab075f006675355ebaa4020463a2",
      "7c44b7fdcec2e517799f6c54f6ba42bf1a89d2e6",
      "7e55d8701785818776323b4147cb13354c820469",
      "92c82a51ad13c361d052987694cf93d6a72d5789",
      "9348b7b95982d0a675a767e92c23647aa6915a94",
      "94ae16f47cfa8dd03a288f342a6169e291c8c889",
      "95154330d74be6c79ac45e97b38ef0038e2b8e2a",
      "9678af027022328f143ee2b627ee93ba313a4e7c",
      "9e57dda195973c4b6c81386b1cc44595ecfd4697",
      "bba1d4de76b8330bea5f73ddeb99da4e01c02e16",
      "c890b28a001c885d1f7aa05f5d24ead9bf6ae058",
      "d0cd8b45949b959c316a3ed75a4683d0a70b1aa9",
      "d32ba88571141ed0ebe7aeefbaa4ccaf8cda7be3",
      "d892b1335ad37c4773f98e8282b73d1460a9f2bf",
      "da54bda4ef0ba0f013ac639fb6a3dea1e4745ad4",
      "e366b47c98cb922d7d5b536ca776dde8333b4bdc",
      "ed32e5bb11a9e5e2f03ea805782e8670a6e10efd",
      "fa5f9aa1cb6f97654ca8e6d279ceee1427a87e68",
      "fd30d3189b3bc3295ddad05ac1f683ce41f5e9cb"
     ]
    }
   ],
   "b": [
    {
     "id": "614f2c3fec4c81eaaa68879d48746bbc89ac72ca",
     "rank": 1,
     "score": 159.118,
     "ti": "PaperClaw: Harnessing Agents for Autonomous Research and Human-in-the-Loop Refinement",
     "au": "Wei Ye · Hangchen Liu · Dongyuan Li · Renhe Jiang",
     "ve": "arXiv.org",
     "yr": 2026,
     "ci": 0,
     "rc": 85,
     "nc": 11,
     "hits": 81,
     "share": 271,
     "best": {
      "ti": "Empowering Biomedical Discovery with AI Agents",
      "n": 29
     },
     "pdf": 1,
     "ab": "Large language models have become capable reasoners and tool users that write and run code and search the literature, which makes automating the research process itself a realistic goal. We present PAPERCLAW, a harnessed multi-agent system that carries a project autonomously, from a field of study to a finished paper. PAPERCLAW curates a domain from a field's live literature, datasets, and code; brainstorms it into an idea with a pre-registered main-result contract; and drives a stoppable hypothesis map through an iterative propose, test, reflect loop that grows only from measured verdicts and halts once the evidence supports the idea, at which point it writes a venue-compliant paper. A full-lifecycle memory keeps each stage in a single living record, so a long run can be paused, inspected, and resumed without losing context. At the centre is an in-cycle research assistant with research tools and skills: it can drive the whole pipeline on its own, while the same interface lets a person step in at any stage, turning a first autonomous draft into a stronger paper through human-in-the-loop refinement. Throughout, PAPERCLAW keeps its output grounded and checkable, citing only references validated against open scholarly indexes and reporting results that genuinely ran. An evaluation with an LLM judge finds that PAPERCLAW produces strong papers both fully autonomously and with human-in-the-loop refinement.",
     "cites": [
      "002bf0720404e5dc6bf43eff64f116ec755b405f",
      "110f5dc6d5bfe67138d64c261d6851c727021d1f",
      "33161a5a9b5dcb635b5a97475e6a6209a69ada7d",
      "390c607723e9767a9a873ded3bc1f66808215e8e",
      "394924896e24c9b086d96d0958dae07f54ff9452",
      "48c83799530dc523ee01e6c1c40ad577d5c10a16",
      "51b7b3ad7645a69e3c1c80cae69473b8bd472f67",
      "5aea5c8b536380c5ad1d42108c2c6767622318ee",
      "6fe3779fe5f2e9402abdd08ad8db41a0f13a99eb",
      "b8ee0b5322382807e687c95cc87b059d3f348495",
      "e366b47c98cb922d7d5b536ca776dde8333b4bdc"
     ]
    },
    {
     "id": "6e835a15d8e1efda43766b3e0c09358934615fd6",
     "rank": 2,
     "score": 125.38,
     "ti": "OPT-BENCH: Evaluating LLM Agent on Large-Scale Search Spaces Optimization Problems",
     "au": "Xiaozhe Li · Jixuan Chen · Xinyu Fang · Shengyuan Ding · Haodong Duan · Qingwen Liu · Kai Chen",
     "ve": "arXiv.org",
     "yr": 2025,
     "ci": 13,
     "rc": 41,
     "nc": 2,
     "hits": 36,
     "share": 241,
     "best": {
      "ti": "Formal Mathematical Reasoning: A New Frontier in AI",
      "n": 12
     },
     "pdf": 1,
     "ab": "Large Language Models (LLMs) have shown remarkable capabilities in solving diverse tasks. However, their proficiency in iteratively optimizing complex solutions through learning from previous feedback remains insufficiently explored. To bridge this gap, we present OPT-BENCH, a comprehensive benchmark designed to evaluate LLM agents on large-scale search space optimization problems. OPT-BENCH includes 20 real-world machine learning tasks sourced from Kaggle and 10 classical NP problems, offering a diverse and challenging environment for assessing LLM agents on iterative reasoning and solution refinement. To enable rigorous evaluation, we introduce OPT-Agent, an end-to-end optimization framework that emulates human reasoning when tackling complex problems by generating, validating, and iteratively improving solutions through leveraging historical feedback. Through extensive experiments on 9 state-of-the-art LLMs from 6 model families, we analyze the effects of optimization iterations, temperature settings, and model architectures on solution quality and convergence. Our results demonstrate that incorporating historical context significantly enhances optimization performance across both ML and NP tasks. All datasets, code, and evaluation tools are open-sourced to promote further research in advancing LLM-driven optimization and iterative reasoning. Project page: \\href{https://github.com/OliverLeeXZ/OPT-BENCH}{https://github.com/OliverLeeXZ/OPT-BENCH}.",
     "cites": [
      "7c44b7fdcec2e517799f6c54f6ba42bf1a89d2e6",
      "b8ee0b5322382807e687c95cc87b059d3f348495"
     ]
    },
    {
     "id": "e085c991028450f0782ea9f6a341ca6247a0b5af",
     "rank": 3,
     "score": 121.147,
     "ti": "OSAgent: Copiloting Operating System with LLM-based Agent",
     "au": "Jiaming Xu · Kaibin Guo · Wuxuan Gong · Runyu Shi",
     "ve": "IEEE International Joint Conference on Neural Network",
     "yr": 2024,
     "ci": 6,
     "rc": 28,
     "nc": 1,
     "hits": 19,
     "share": 216,
     "best": {
      "ti": "Large language models empowered agent-based modeling and simulation: a survey and perspectives",
      "n": 9
     },
     "pdf": 0,
     "ab": "The large language model (LLM) has indeed made significant strides in enhancing human-computer interaction. However, it still faces challenges when dealing with more intricate tasks. Some agents have endeavored to augment LLM capabilities through external memory and tool invocation. Despite these efforts, the task completion rate remains less than satisfactory. This paper introduces a memory-enhanced LLM-based OSAgent designed to assist users in copiloting their operating systems. Specifically, we employ an external memory to store historical user requests, along with corresponding semantic vectors, intent vectors, and human-annotated task planning instructions. When receiving a user request, OSAgent first retrieves similar user requests and task planning instructions as examples to generate prompt, then LLM conducts task planning to manipulate OS tools. Extensive experiments were conducted on mobile operating systems employing seven LLMs. The results are indicative of OSAgent’s remarkable proficiency in invoking AI capabilities, third-party applications, system settings, and data resources.",
     "cites": [
      "59b3fbf146b29b581b677ec4384f14cee87997a4"
     ]
    },
    {
     "id": "58ad16a1e4435cd9e9becac470de47f1dd9e1316",
     "rank": 4,
     "score": 121.052,
     "ti": "RL-Struct: A Lightweight Reinforcement Learning Framework for Reliable Structured Output in LLMs",
     "au": "Rui Hu · Shulei Wu",
     "ve": "arXiv.org",
     "yr": 2025,
     "ci": 2,
     "rc": 101,
     "nc": 2,
     "hits": 73,
     "share": 263,
     "best": {
      "ti": "Empowering Biomedical Discovery with AI Agents",
      "n": 24
     },
     "pdf": 1,
     "ab": "The Structure Gap between probabilistic LLM generation and deterministic schema requirements hinders automated workflows. We propose RL-Struct, a lightweight framework using Gradient Regularized Policy Optimization (GRPO) with a hierarchical reward function to align LLMs with structural constraints. This approach eliminates the critic network, reducing peak VRAM by 38% compared to PPO. On complex JSON tasks, RL-Struct achieves 89.7% structural accuracy and 92.1% validity, significantly outperforming SFT and zero-shot baselines. We also report an emergent curriculum--a self-organized learning process where the model prioritizes syntax before semantics. Our model is publicly available at https://huggingface.co/Freakz3z/Qwen-JSON.",
     "cites": [
      "87875a07976c26f82705de1fc70041169e5d652b",
      "9b093787f9be3d480cd11f8ec6ca5b0e44050d6d"
     ]
    },
    {
     "id": "180560ea705f0228262a3a435ed2212c87a29657",
     "rank": 5,
     "score": 117.38,
     "ti": "AutoMaAS: Self-Evolving Multi-Agent Architecture Search for Large Language Models",
     "au": "Bo Ma · Hang Li · Zehua Hu · XiaoFan Gui · Luyao Liu · Simon Liu",
     "ve": "arXiv.org",
     "yr": 2025,
     "ci": 0,
     "rc": 74,
     "nc": 1,
     "hits": 66,
     "share": 252,
     "best": {
      "ti": "Empowering Biomedical Discovery with AI Agents",
      "n": 27
     },
     "pdf": 1,
     "ab": "Multi-agent systems powered by large language models have demonstrated remarkable capabilities across diverse domains, yet existing automated design approaches seek monolithic solutions that fail to adapt resource allocation based on query complexity and domain requirements. This paper introduces AutoMaAS, a self-evolving multi-agent architecture search framework that leverages neural architecture search principles to automatically discover optimal agent configurations through dynamic operator lifecycle management and automated machine learning techniques. Our approach incorporates four key innovations: (1) automatic operator generation, fusion, and elimination based on performance-cost analysis, (2) dynamic cost-aware optimization with real-time parameter adjustment, (3) online feedback integration for continuous architecture refinement, and (4) enhanced interpretability through decision tracing mechanisms. Extensive experiments across six benchmarks demonstrate that AutoMaAS achieves 1.0-7.1\\% performance improvement while reducing inference costs by 3-5\\% compared to state-of-the-art methods. The framework shows superior transferability across datasets and LLM backbones, establishing a new paradigm for automated multi-agent system design in the era of large language models.",
     "cites": [
      "53132ea6c107479d4557631299d3ed525109b464"
     ]
    }
   ]
  };

  window.KLRunMock = {
    PIPELINE: PIPELINE,
    FILTERS: FILTERS,
    RUN37: RUN37,
    RECO: RECO,
    RUN_LIBRARY: RUN_LIBRARY,
    NEXT_RUN_NO: NEXT_RUN_NO,
    REPLAY: REPLAY,
    makeReplay: makeReplay,
    buildTimeline: buildTimeline
  };
  try { document.dispatchEvent(new CustomEvent('kl-run-mock-ready')); } catch (e) {}
})();
