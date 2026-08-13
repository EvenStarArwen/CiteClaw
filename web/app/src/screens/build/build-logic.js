/**
 * build-logic.js — the Build screen behaviour, TRANSPLANTED VERBATIM.
 *
 * Source: web/design-reference/embedded-sources/"Paper Card.dc.html" lines
 * 2363-5874, i.e. the demo screen's entire `class Component extends DCLogic`.
 * Everything between the BEGIN/END markers below is byte-identical to that
 * source; `scripts/verify-build-logic.mjs` asserts it on every run.
 *
 * Exactly two edits are applied, both structural and both outside the marked
 * region:
 *   1. `class Component extends DCLogic {` -> `export class BuildLogic {`
 *      DCLogic (support.js) is a no-op React.Component base: its
 *      componentDidMount / componentDidUpdate / componentWillUnmount /
 *      renderVals are all empty, and this class overrides every one it uses.
 *   2. `rootRef = React.createRef();` -> a constructor that receives the React
 *      ref and the props object, so the class needs neither React nor DCLogic.
 *
 * It is deliberately .js, not .ts: a verbatim copy cannot also satisfy
 * `strict` type-checking, and rewriting it to do so would stop being a
 * transplant. Types live in build-logic.d.ts. Same pattern the scaffold
 * already uses for src/design-data/*.js.
 *
 * Props the demo shell hands this screen (KnowledgeLab iPad Demo.dc.html:80,
 * merged with the screen's own data-props defaults) are assembled in
 * BuildScreen.tsx — do not default them here.
 */

/* eslint-disable */
export class BuildLogic {
  constructor(rootRef, props) {
    this.rootRef = rootRef;
    this.props = props || {};
  }

  /* ===== BEGIN VERBATIM TRANSPLANT (Paper Card.dc.html 2365-5874) ===== */
  componentDidMount() {
    // Top-bar Runs tab activity dot (dispatched by the Runs page): pulsing = running, static = finished and not yet viewed.
    const actPaint = d => {
      const r = this.rootRef.current; if (!r) return;
      const st = (d && d.state) || 'none';
      r.querySelectorAll('.tb-runs-dot').forEach(dot => {
        const tab = dot.closest('.tb-tab'); const active = tab && tab.getAttribute('data-active') === '1';
        const show = st !== 'none' && !active;
        dot.style.display = show ? 'block' : 'none';
        dot.style.animation = st === 'running' ? 'sbPulse 1.4s ease-in-out infinite' : 'none';
        if (tab) tab.title = !show ? '' : (st === 'running' ? (d.label || 'A run') + ' is running' : (d.label || 'A run') + ' finished, not viewed yet');
      });
    };
    document.addEventListener('kl-run-activity', e => actPaint((e && e.detail) || {}));
    this._actRepaint = () => actPaint(window.__klRunAct || {});
    setTimeout(() => {
      this._actRepaint();
      const r = this.rootRef.current, apg = r && r.closest && r.closest('.pg');
      if (apg) { const mo = new MutationObserver(this._actRepaint); mo.observe(apg, { attributes: true, attributeFilter: ['data-on'] }); }
    }, 0);
    this._vpRO = new ResizeObserver(entries => { const r = entries[0].target; r.setAttribute('data-vp', r.clientWidth <= 1240 ? 'narrow' : r.clientWidth <= 1440 ? 'compact' : 'full'); this._vpPlace(); if (!r._vpReady) { r._vpReady = 1; requestAnimationFrame(() => requestAnimationFrame(() => r.setAttribute('data-vp-ready', '1'))); } requestAnimationFrame(() => r.querySelectorAll('.dr').forEach(d => d._sync && d._sync())); });
    const vpAttach = () => { const r = this.rootRef.current; if (!r || r._vpObs) return; r._vpObs = 1; this._vpRO.observe(r); };
    vpAttach(); [200, 800, 2000].forEach(ms => setTimeout(vpAttach, ms));
    const nrWire = () => {
      const root = this.rootRef.current;
      if (!root || root._nrW) return; root._nrW = 1;
      root.addEventListener('click', e => {
        const sp = e.target.closest('.cfg-src-seg button');
        if (sp) { root.setAttribute('data-srcpage', sp.dataset.srcpage); root.querySelectorAll('.cfg-src-seg button').forEach(x => x.setAttribute('data-on', x === sp ? '1' : '0')); }
      });
    };
    nrWire();
    const root = this.rootRef.current;
    if (!root) return;
    const gear = root.querySelector('.tb-settings');
    if (gear) gear.addEventListener('click', () => gear.dispatchEvent(new CustomEvent('kl-open-settings', { bubbles: true })));
    this.applyTheme(this.props.theme || 'light');
    this.setupTips(root);
    root.setAttribute('data-handle', 'Diamond');
    const tgBtn = root.querySelector('.pc-theme-toggle');
    if (tgBtn) tgBtn.addEventListener('click', () => this.applyTheme(this._theme === 'dark' ? 'light' : 'dark'));
    this.applyLayout();
    this.applyScheme('Recessed canvas');
    this.applyColor(this.props.colorScheme || 'Warm paper');
    this.applyLogo(this.props.logoStyle || 'Soft ink tile');
    this.applyTabs();
    this.applyChip();
    this.applyAddBtn();
    root.querySelectorAll('.pc').forEach(card => {
      card.addEventListener('mouseenter', () => {
        card.style.borderColor = 'var(--fg2)';
        card.style.boxShadow = '0 6px 18px rgba(var(--sh),.07)';
      });
      card.addEventListener('mouseleave', () => {
        card.style.borderColor = 'var(--border)';
        card.style.boxShadow = '0 1px 2px rgba(var(--sh),.03)';
      });
    });
    root.querySelectorAll('.pr').forEach(row => { if (!row.closest('.sidebar')) this.wireRow(row, root); });
    this.setupSidebar(root);
    this.setupAbstracts(root);
    if (this.props.overscrollBounce) this.setupRubber(root);
    this.setupConfigPanel(root);
    this.setupDownload(root);
    this.setupUserMenu(root);
    this.setupProjMenu(root);
    this.applyPage(this.props.pageState || 'Has results');
    document.addEventListener('keydown', e => {
      if (e.key !== 'Escape') return;
      root.querySelectorAll('.sidebar').forEach(sb => { if (sb.dataset.select === 'persist') return; sb.querySelectorAll('.pr').forEach(r => { r.dataset.selected = '0'; if (r.__paint) r.__paint(); }); });
      root.querySelectorAll('.cf-filter').forEach(f => { f.dataset.sel = '0'; });
      this.closeConfig();
    });
  }
  wireRow(row, root) {
    if (row.__wired) return; row.__wired = 1;
  const STAR = 'M11.48 3.5a.5.5 0 0 1 1 0l2.4 5.05 5.5.7a.5.5 0 0 1 .28.86l-4.05 3.8 1.05 5.5a.5.5 0 0 1-.74.53L12 17.7l-4.9 2.54a.5.5 0 0 1-.73-.53l1.05-5.5-4.05-3.8a.5.5 0 0 1 .28-.86l5.5-.7z';
    const rail = row.querySelector('.rail');
    const title = row.querySelector('.pr-title');
    // star lane: never on .cs-row seed clones (starless; trash overlays on hover)
    if (!row.classList.contains('cs-row')) row.style.paddingRight = '46px';
    let star = row.querySelector('.pr-star');
    if (!star) {
      // Fallback ONLY for rows that never go through prCardInner (rule 1: prCardInner
      // is the sole builder). Painted rows already carry theirs; never add a second.
      star = document.createElement('button');
      star.className = 'pr-star';
      star.setAttribute('aria-label', 'Save paper');
      star.style.cssText = 'position:absolute; right:14px; top:50%; transform:translateY(-50%); background:none; border:none; padding:6px; margin:0; cursor:pointer; display:flex; border-radius:8px; z-index:2; transition:opacity .16s ease, background .16s ease;';
      star.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--star)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="' + STAR + '"></path></svg>';
      row.appendChild(star);
    }
    const setStar = () => {
      const saved = star.dataset.saved === '1';
      const svg = star.querySelector('svg');
      svg.style.fill = saved ? 'var(--accent)' : 'none';
      svg.style.stroke = saved ? 'var(--accent)' : 'var(--star)';
      // visibility is CSS state (.pr:hover .pr-star / [data-saved]) — never set here
    };
    star.addEventListener('mouseenter', () => { star.style.background = 'var(--ghost-hover)'; });
    star.addEventListener('mouseleave', () => { star.style.background = 'none'; });
    row.__star = star; row.__setStar = setStar;
    star.addEventListener('pointerdown', () => { const s = star.querySelector('svg'); if (s) { s.style.transition = 'transform .1s ease'; s.style.transform = 'scale(.8)'; } });
    const starUp = () => { const s = star.querySelector('svg'); if (s) s.style.transform = ''; };
    star.addEventListener('pointerup', starUp);
    star.addEventListener('pointerleave', starUp);
    star.addEventListener('pointercancel', starUp);
    star.addEventListener('click', (e) => {
      e.stopPropagation();
      const sv = star.querySelector('svg');
      if (sv) sv.style.transform = '';
      if (row.closest('.sidebar')) {
        if (star.dataset.saved === '1' || row._leaving) return;
        star.dataset.saved = '1'; setStar();
        if (sv && sv.animate) sv.animate([{ transform: 'scale(.7)' }, { transform: 'scale(1.45)', offset: .45 }, { transform: 'scale(1)' }], { duration: 420, easing: 'cubic-bezier(.34,1.56,.64,1)' });
        // the row visibly leaves for the seed set instead of vanishing in place
        row._leaving = 1;
        const h = row.offsetHeight, cs = getComputedStyle(row);
        row.style.overflow = 'hidden'; row.style.pointerEvents = 'none';
        const anim = row.animate ? row.animate([
          { height: h + 'px', paddingTop: cs.paddingTop, paddingBottom: cs.paddingBottom, opacity: 1, transform: 'none' },
          { height: h + 'px', paddingTop: cs.paddingTop, paddingBottom: cs.paddingBottom, opacity: 0, transform: 'translateX(30px)', offset: .55 },
          { height: '0px', paddingTop: '0px', paddingBottom: '0px', opacity: 0, transform: 'translateX(30px)' }
        ], { duration: 440, delay: 120, easing: 'cubic-bezier(.4,0,.2,1)', fill: 'forwards' }) : null;
        const done = () => { row._leaving = 0; row.style.overflow = ''; row.style.pointerEvents = ''; this.seedAdd(row); if (anim) anim.cancel(); };
        if (anim) anim.onfinish = done; else done();
      } else {
        star.dataset.saved = star.dataset.saved === '1' ? '0' : '1';
        setStar();
        if (sv && sv.animate) sv.animate([{ transform: 'scale(1)' }, { transform: 'scale(1.5)', offset: .4 }, { transform: 'scale(1)' }], { duration: 380, easing: 'cubic-bezier(.34,1.56,.64,1)' });
      }
    });
    const card = row.dataset.rowStyle ? row.dataset.rowStyle === 'cards' : !!row.closest('[data-style="cards"]');
    const paint = () => {
      const sel = row.dataset.selected === '1';
      const hov = row.dataset.hover === '1';
      if (card) {
        row.style.borderColor = sel ? 'var(--accent)' : (hov ? 'var(--muted2)' : 'var(--border)');
        row.style.background = sel ? 'var(--card-sel)' : 'var(--card2)';
        row.style.boxShadow = sel ? '0 4px 14px rgba(var(--sh),.07)' : (hov ? '0 4px 14px rgba(var(--sh),.06)' : 'none');
      } else {
        row.style.background = sel ? 'var(--row-sel)' : (hov ? 'var(--row-hover)' : 'transparent');
        if (rail) { rail.style.opacity = sel ? '1' : '0'; rail.style.width = '4px'; }
      }
      if (title) title.style.color = 'var(--fg)';
    };
    row.__paint = paint;
    row.addEventListener('mouseenter', () => { row.dataset.hover = '1'; paint(); setStar(); });
    row.addEventListener('mouseleave', () => { row.dataset.hover = '0'; paint(); setStar(); });
    row.addEventListener('mousedown', () => { row.style.filter = 'brightness(.97)'; });
    row.addEventListener('mouseup', () => { row.style.filter = 'none'; });
    row.addEventListener('click', () => {
      const sb = row.closest('.sidebar') || row.closest('.cfg-seed-list');
      const scope = sb || root;
      scope.querySelectorAll('.pr').forEach(r => { r.dataset.selected = '0'; if (r.__paint) r.__paint(); });
      row.dataset.selected = '1';
      paint();
      if (row.__onSelect) row.__onSelect();
    });
  }
  setupSidebar(root) {
    if (!window.KLPaperRow) { document.addEventListener('kl-paper-row-ready', () => this.setupSidebar(root), { once: true }); return; }
    const mode = 'inline';
    root.querySelectorAll('.sidebar').forEach(sb => { this.expandMock(sb); this.wireSidebar(sb, mode); sb.querySelectorAll('.pr').forEach(row => this.wireRow(row, root)); });
  }
  mockPapers() {
    const raw = [
      'Data Interpreter: An LLM Agent For Data Science|Sirui Hong · Yizhang Lin · Bangbang Liu · Binhao Wu · Danyang Li · Jiaqi Chen · Jiayi Zhang · Jinlin Wang · Lingyao Zhang · Mingchen Zhuge · Taicheng Guo · Tuo Zhou · Wei Tao · Wenyi Wang · Xiangru Tang · Xiang Lu · Xinbing Liang · Yaying Fei · Yuheng Cheng · Zhibin Gou · Zongze Xu · Chenglin Wu · Li Zhang · Min Yang · Xiawu Zheng|Annual Meeting of the Association for Computational Linguistics|2024|244|Large Language Model (LLM)-based agents have shown effectiveness across many applications. However, their use in data science scenarios requiring solving long-term interconnected tasks, dynamic data adjustments and domain expertise remains challenging. Previous approaches primarily focus on individual tasks, making it difficult to assess the complete data science workflow. Moreover, they struggle to handle real-time changes in intermediate data and fail to adapt dynamically to evolving task dependencies inherent to data science problems. In this paper, we present Data Interpreter, an LLM-based agent designed to automatically solve various data science problems end-to-end. Our Data Interpreter incorporates two key modules: 1) Hierarchical Graph Modeling, which breaks down complex problems into manageable subproblems, enabling dynamic node generation and graph optimization; and 2) Programmable Node Generation, a technique that refines and verifies each subproblem to iteratively improve code generation results and robustness. Extensive experiments consistently demonstrate the superiority of Data Interpreter. On InfiAgent-DABench, it achieves a 25% performance boost, raising accuracy from 75.9% to 94.9%. For machine learning and open-ended tasks, it improves performance from 88% to 95%, and from 60% to 97%, respectively. Moreover, on the MATH dataset, Data Interpreter achieves remarkable performance with a 26% improvement compared to state-of-the-art baselines. The code is available at https://github.com/geekan/MetaGPT.',
      'Concept Induction: Analyzing Unstructured Text with High-Level Concepts Using LLooM|Michelle S. Lam · Janice Teoh · James Landay · J. Heer · Michael S. Bernstein|International Conference on Human Factors in Computing Systems|2024|128|Data analysts have long sought to turn unstructured text data into meaningful concepts. Though common, topic modeling and clustering focus on lower-level keywords and require significant interpretative work. We introduce concept induction, a computational process that instead produces high-level concepts, defined by explicit inclusion criteria, from unstructured text. For a dataset of toxic online comments, where a state-of-the-art BERTopic model outputs “women, power, female,” concept induction produces high-level concepts such as “Criticism of traditional gender roles” and “Dismissal of women’s concerns.” We present LLooM, a concept induction algorithm that leverages large language models to iteratively synthesize sampled text and propose human-interpretable concepts of increasing generality. We then instantiate LLooM in a mixed-initiative text analysis tool, enabling analysts to shift their attention from interpreting topics to engaging in theory-driven analysis. Through technical evaluations and four analysis scenarios ranging from literature review to content moderation, we find that LLooM’s concepts improve upon the prior art of topic models in terms of quality and data coverage. In expert case studies, LLooM helped researchers to uncover new insights even from familiar datasets, for example by suggesting a previously unnoticed concept of attacks on out-party stances in a political social media dataset.',
      'ScienceBoard: Evaluating Multimodal Autonomous Agents in Realistic Scientific Workflows|Qiushi Sun · Zhoumianze Liu · Chang Ma · Zichen Ding · Fangzhi Xu · Zhangyue Yin · Haiteng Zhao · Zhenyu Wu · Kanzhi Cheng · Zhaoyang Liu · Jianing Wang · Qintong Li · Xiangru Tang · Tianbao Xie · Xiachong Feng · Xiang Li · Ben Kao · Wenhai Wang · Biqing Qi · Lingpeng Kong · Zhiyong Wu|arXiv.org|2025|30|Large Language Models (LLMs) have extended their impact beyond Natural Language Processing, substantially fostering the development of interdisciplinary research. Recently, various LLM-based agents have been developed to assist scientific discovery progress across multiple aspects and domains. Among these, computer-using agents, capable of interacting with operating systems as humans do, are paving the way to automated scientific problem-solving and addressing routines in researchers’workflows. Recognizing the transformative potential of these agents, we introduce ScienceBoard, which encompasses two complementary contributions: (i) a realistic, multi-domain environment featuring dynamic and visually rich scientific workflows with integrated professional software, where agents can autonomously interact via different interfaces to accelerate complex research tasks and experiments; and (ii) a challenging benchmark of 169 high-quality, rigorously validated real-world tasks curated by humans, spanning scientific-discovery workflows in domains such as biochemistry, astronomy, and geoinformatics. Extensive evaluations of agents with state-of-the-art backbones (e.g., GPT-4o, Claude 3.7, UI-TARS) show that, despite some promising results, they still fall short of reliably assisting scientists in complex workflows, achieving only a 15% overall success rate. In-depth analysis further provides valuable insights for addressing current agent limitations and more effective design principles, paving the way to build more capable agents for scientific discovery. Our code, environment, and benchmark are at https://qiushisun.github.io/ScienceBoard-Home/.',
      'Hypothesis Generation for Materials Discovery and Design Using Goal-Driven and Constraint-Guided LLM Agents|Shrinidhi Kumbhar · Venkatesh Mishra · Kevin Coutinho · Divij Handa · A. Iquebal · Chitta Baral|North American Chapter of the Association for Computational Linguistics|2025|33|Materials discovery and design are essential for advancing technology across various industries by enabling the development of application-specific materials. Recent research has leveraged Large Language Models (LLMs) to accelerate this process. We explore the potential of LLMs to generate viable hypotheses that, once validated, can expedite materials discovery. Collaborating with materials science experts, we curated a novel dataset from recent journal publications, featuring real-world goals, constraints, and methods for designing real-world applications. Using this dataset, we test LLM-based agents that generate hypotheses for achieving given goals under specific constraints. To assess the relevance and quality of these hypotheses, we propose a novel scalable evaluation metric that emulates the process a materials scientist would use to evaluate a hypothesis critically. Our curated dataset, proposed method, and evaluation framework aim to advance future research in accelerating materials discovery and design with LLMs.',
      'CycleResearcher: Improving Automated Research via Automated Review|Yixuan Weng · Minjun Zhu · Guangsheng Bao · Hongbo Zhang · Jindong Wang · Yue Zhang · Linyi Yang|International Conference on Learning Representations|2024|132|The automation of scientific discovery has been a long-standing goal within the research community, driven by the potential to accelerate knowledge creation. While significant progress has been made using commercial large language models (LLMs) as research assistants or idea generators, the possibility of automating the entire research process with open-source LLMs remains largely unexplored. This paper explores the feasibility of using open-source post-trained LLMs as autonomous agents capable of performing the full cycle of automated research and review, from literature review and manuscript preparation to peer review and paper refinement. Our iterative preference training framework consists of CycleResearcher, which conducts research tasks, and CycleReviewer, which simulates the peer review process, providing iterative feedback via reinforcement learning. To train these models, we develop two new datasets, Review-5k and Research-14k, reflecting real-world machine learning research and peer review dynamics. Our results demonstrate that CycleReviewer achieves promising performance with a 26.89% reduction in mean absolute error (MAE) compared to individual human reviewers in predicting paper scores, indicating the potential of LLMs to effectively assist expert-level research evaluation. In research, the papers generated by the CycleResearcher model achieved a score of 5.36 in simulated peer reviews, showing some competitiveness in terms of simulated review scores compared to the preprint level of 5.24 from human experts, while still having room for improvement compared to the accepted paper level of 5.69. This work represents a significant step toward fully automated scientific inquiry, providing ethical safeguards and exploring AI-driven research capabilities. The code, dataset and model weight are released at https://wengsyx.github.io/Researcher/.',
      'ORGANA: A Robotic Assistant for Automated Chemistry Experimentation and Characterization|K. Darvish · Marta Skreta · Yuchi Zhao · Naruki Yoshikawa · Sagnik Som · Miroslav Bogdanovic · Yang Cao · Han Hao · Haoping Xu · Alán Aspuru-Guzik · Animesh Garg · F. Shkurti|Matter|2024|141|Chemistry experiments can be resource- and labor-intensive, often requiring manual tasks like polishing electrodes in electrochemistry. Traditional lab automation infrastructure faces challenges adapting to new experiments. To address this, we introduce ORGANA, an assistive robotic system that automates diverse chemistry experiments using decision-making and perception tools. It makes decisions with chemists in the loop to control robots and lab devices. ORGANA interacts with chemists using Large Language Models (LLMs) to derive experiment goals, handle disambiguation, and provide experiment logs. ORGANA plans and executes complex tasks with visual feedback, while supporting scheduling and parallel task execution. We demonstrate ORGANA’s capabilities in solubility, pH measurement, recrystallization, and electrochemistry experiments. In electrochemistry, it executes a 19-step plan in parallel to characterize quinone derivatives for flow batteries. Our user study shows ORGANA reduces frustration and physical demand by over 50%, with users saving an average of 80.3% of their time when using it.',
      'AlphaEvolve: A coding agent for scientific and algorithmic discovery|Alexander Novikov · Ngân V˜u · Marvin Eisenberger · Emilien Dupont · Po-Sen Huang · Adam Zsolt Wagner · S. Shirobokov · B. Kozlovskii · Francisco J. R. Ruiz · Abbas Mehrabian · M. P. Kumar · Abigail See · Swarat Chaudhuri · George Holland · Alex Davies · Sebastian Nowozin · Pushmeet Kohli · Matej Balog · Google DeepMind|arXiv.org|2025|630|In this white paper, we present AlphaEvolve, an evolutionary coding agent that substantially enhances capabilities of state-of-the-art LLMs on highly challenging tasks such as tackling open scientific problems or optimizing critical pieces of computational infrastructure. AlphaEvolve orchestrates an autonomous pipeline of LLMs, whose task is to improve an algorithm by making direct changes to the code. Using an evolutionary approach, continuously receiving feedback from one or more evaluators, AlphaEvolve iteratively improves the algorithm, potentially leading to new scientific and practical discoveries. We demonstrate the broad applicability of this approach by applying it to a number of important computational problems. When applied to optimizing critical components of large-scale computational stacks at Google, AlphaEvolve developed a more efficient scheduling algorithm for data centers, found a functionally equivalent simplification in the circuit design of hardware accelerators, and accelerated the training of the LLM underpinning AlphaEvolve itself. Furthermore, AlphaEvolve discovered novel, provably correct algorithms that surpass state-of-the-art solutions on a spectrum of problems in mathematics and computer science, significantly expanding the scope of prior automated discovery methods (Romera-Paredes et al., 2023). Notably, AlphaEvolve developed a search algorithm that found a procedure to multiply two $4 4$ complex-valued matrices using $48$ scalar multiplications; offering the first improvement, after 56 years, over Strassen’s algorithm in this setting. We believe AlphaEvolve and coding agents like it can have a significant impact in improving solutions of problems across many areas of science and computation.',
      'Kosmos: An AI Scientist for Autonomous Discovery|L. Mitchener · Angela Yiu · Benjamin Chang · M. Bourdenx · Tyler Nadolski · Arvis Sulovari · E. Landsness · Dániel L. Barabási · Siddharth Narayanan · Nicky Evans · S. Reddy · M. Foiani · Aizad Kamal · Leah P. Shriver · F. Cao · A. Wassie · Jon M. Laurent · Edwin Melville-Green · M. C. Ramos · Albert Bou · Kaleigh F. Roberts · Sladjana Zagorac · Timothy C. Orr · Miranda E. Orr · K. Zwezdaryk · Ali E. Ghareeb · L. McCoy · B. Gomes · Euan A. Ashley · K. Duff · T. Buonassisi · Tom Rainforth · Randall J. Bateman · Michael Skarlinski · S. Rodriques · Michaela M. Hinks · Andrew D. White|arXiv.org|2025|64|Data-driven scientific discovery requires iterative cycles of literature search, hypothesis generation, and data analysis. Substantial progress has been made towards AI agents that can automate scientific research, but all such agents remain limited in the number of actions they can take before losing coherence, thus limiting the depth of their findings. Here we present Kosmos, an AI scientist that automates data-driven discovery. Given an open-ended objective and a dataset, Kosmos runs for up to 12 hours performing cycles of parallel data analysis, literature search, and hypothesis generation before synthesizing discoveries into scientific reports. Unlike prior systems, Kosmos uses a structured world model to share information between a data analysis agent and a literature search agent. The world model enables Kosmos to coherently pursue the specified objective over 200 agent rollouts, collectively executing an average of 42,000 lines of code and reading 1,500 papers per run. Kosmos cites all statements in its reports with code or primary literature, ensuring its reasoning is traceable. Independent scientists found 79.4% of statements in Kosmos reports to be accurate, and collaborators reported that a single 20-cycle Kosmos run performed the equivalent of 6 months of their own research time on average. Furthermore, collaborators reported that the number of valuable scientific findings generated scales linearly with Kosmos cycles (tested up to 20 cycles). We highlight seven discoveries made by Kosmos that span metabolomics, materials science, neuroscience, and statistical genetics. Three discoveries independently reproduce findings from preprinted or unpublished manuscripts that were not accessed by Kosmos at runtime, while four make novel contributions to the scientific literature.',
      'SciMON: Scientific Inspiration Machines Optimized for Novelty|Qingyun Wang · Doug Downey · Heng Ji · Tom Hope|Annual Meeting of the Association for Computational Linguistics|2023|193|We explore and enhance the ability of neural language models to generate novel scientific directions grounded in literature. Work on literature-based hypothesis generation has traditionally focused on binary link prediction--severely limiting the expressivity of hypotheses. This line of work also does not focus on optimizing novelty. We take a dramatic departure with a novel setting in which models use as input background contexts (e.g., problems, experimental settings, goals), and output natural language ideas grounded in literature. We present SciMON, a modeling framework that uses retrieval of"inspirations"from past scientific papers, and explicitly optimizes for novelty by iteratively comparing to prior papers and updating idea suggestions until sufficient novelty is achieved. Comprehensive evaluations reveal that GPT-4 tends to generate ideas with overall low technical depth and novelty, while our methods partially mitigate this issue. Our work represents a first step toward evaluating and developing language models that generate new ideas derived from the scientific literature',
      'Ideas are Dimes a Dozen: Large Language Models for Idea Generation in Innovation|Karan Girotra · Lennart Meincke · C. Terwiesch · K. Ulrich|Social Science Research Network|2023|165|Large language models (LLMs) such as OpenID’s GPT series have shown remarkable capabilities in generating fluent and coherent text in various domains. We compare the ideation capabilities of ChatGPT-4, a chatbot based on a state-of-the-art LLM, with those of students at an elite university. ChatGPT-4 can generate ideas much faster and cheaper than students, and the ideas are on average of higher quality (as measured by purchase-intent surveys) and exhibit higher variance in quality. More important, the vast majority of the best ideas in the pooled sample are generated by ChatGPT and not by the students. Providing ChatGPT with a few examples of highly rated ideas further increases its performance. We discuss the implications of these findings for the management of innovation.',
      'Large language models to accelerate organic chemistry synthesis|Yu Zhang · Yang Han · Shuai Chen · Ruijie Yu · Xin Zhao · Xianbin Liu · Kaipeng Zeng · Mengdi Yu · Jidong Tian · Feng Zhu · Xiaokang Yang · Yaohui Jin · Yanyan Xu|Nature Machine Intelligence|2025|44|Chemical synthesis, as a foundational methodology in the creation of transformative molecules, exerts substantial influence across diverse sectors from life sciences to materials and energy. Current chemical synthesis practices emphasize laborious and costly trial-and-error workflows, underscoring the urgent needs for advanced AI assistants. Recently, large language models, typified by GPT-4, have been introduced as an efficient tool to facilitate scientific research. Here we present Chemma, a fully fine-tuned large language model with 1.28 million pairs of questions and answers about reactions, as an assistant to accelerate organic chemistry synthesis. Chemma surpasses the best-known results in multiple chemical tasks, for example, single-step retrosynthesis and yield prediction, which highlights the potential of general artificial intelligence for organic chemistry. By predicting yields across the experimental reaction space, Chemma significantly improves the reaction exploration capability of Bayesian optimization. More importantly, integrated in an active learning framework, Chemma exhibits advanced potentials of autonomously experimental exploration and optimization in open reaction spaces. For an unreported Suzuki–Miyaura cross-coupling reaction of cyclic aminoboronates and aryl halides for the synthesis of α-aryl N-heterocycles, the human–artificial intelligence collaboration successfully explored a suitable ligand (tri(1-adamantyl)phosphine) and solvent (1,4-dioxane) within only 15 runs, achieving an isolated yield of 67%. These results reveal that, without quantum-chemical calculations, Chemma can comprehend and extract chemical insights from reaction data, in a manner akin to human experts. This work opens avenues for accelerating organic chemistry synthesis with adapted large language models. Large language models (LLMs) can be useful tools for science, but they often lack expert understanding of complex domains that they were not trained on. Zhang and colleagues fine-tuned a LLaMA-2-7b-based LLM with questions on organic chemistry reactions.',
      'AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench|Edan Toledo · Karen Hambardzumyan · Martin Josifoski · Rishi Hazra · N. Baldwin · Alexis Audran-Reiss · Michael Kuchnik · Despoina Magka · Minqi Jiang · A. Lupidi · Andrei Lupu · R. Raileanu · Kelvin Niu · Tatiana Shavrina · Jean-Christophe Gagnon-Audet · Michael Shvartsman · Shagun Sodhani · Alexander H. Miller · Abhishek Charnalia · Derek Dunfield · Carole-Jean Wu · Pontus Stenetorp · Nicola Cancedda · J. Foerster · Yoram Bachrach|arXiv.org|2025|41|AI research agents are demonstrating great potential to accelerate scientific progress by automating the design, implementation, and training of machine learning models. We focus on methods for improving agents’performance on MLE-bench, a challenging benchmark where agents compete in Kaggle competitions to solve real-world machine learning problems. We formalize AI research agents as search policies that navigate a space of candidate solutions, iteratively modifying them using operators. By designing and systematically varying different operator sets and search policies (Greedy, MCTS, Evolutionary), we show that their interplay is critical for achieving high performance. Our best pairing of search strategy and operator set achieves a state-of-the-art result on MLE-bench lite, increasing the success rate of achieving a Kaggle medal from 39.6% to 47.7%. Our investigation underscores the importance of jointly considering the search strategy, operator design, and evaluation methodology in advancing automated machine learning.',
      'Can GPT-4 Perform Neural Architecture Search?|Mingkai Zheng · Xiu Su · Shan You · Fei Wang · Chen Qian · Chang Xu · Samuel Albanie|arXiv.org|2023|101|We investigate the potential of GPT-4 to perform Neural Architecture Search (NAS) -- the task of designing effective neural architectures. Our proposed approach, GPT-4 Enhanced Neural archItectUre Search (GENIUS), leverages the generative capabilities of GPT-4 as a black-box optimiser to quickly navigate the architecture search space, pinpoint promising candidates, and iteratively refine these candidates to improve performance. We assess GENIUS across several benchmarks, comparing it with existing state-of-the-art NAS techniques to illustrate its effectiveness. Rather than targeting state-of-the-art performance, our objective is to highlight GPT-4’s potential to assist research on a challenging technical problem through a simple prompting scheme that requires relatively limited domain expertise{Code available at .}. More broadly, we believe our preliminary results point to future research that harnesses general purpose language models for diverse optimisation tasks. We also highlight important limitations to our study, and note implications for AI safety.',
      'AIDE: AI-Driven Exploration in the Space of Code|Z. Jiang · Dominik Schmidt · Dhruv Srikanth · Dixing Xu · Ian Kaplan · Deniss Jacenko · Yuxiang Wu|arXiv.org|2025|152|Machine learning, the foundation of modern artificial intelligence, has driven innovations that have fundamentally transformed the world. Yet, behind advancements lies a complex and often tedious process requiring labor and compute intensive iteration and experimentation. Engineers and scientists developing machine learning models spend much of their time on trial-and-error tasks instead of conceptualizing innovative solutions or research hypotheses. To address this challenge, we introduce AI-Driven Exploration (AIDE), a machine learning engineering agent powered by large language models (LLMs). AIDE frames machine learning engineering as a code optimization problem, and formulates trial-and-error as a tree search in the space of potential solutions. By strategically reusing and refining promising solutions, AIDE effectively trades computational resources for enhanced performance, achieving state-of-the-art results on multiple machine learning engineering benchmarks, including our Kaggle evaluations, OpenAI MLE-Bench and METRs RE-Bench.',
      'DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data|Huajian Xin · Daya Guo · Zhihong Shao · Z. Ren · Qihao Zhu · Bo Liu · C. Ruan · Wenda Li · Xiaodan Liang|arXiv.org|2024|234|Proof assistants like Lean have revolutionized mathematical proof verification, ensuring high accuracy and reliability. Although large language models (LLMs) show promise in mathematical reasoning, their advancement in formal theorem proving is hindered by a lack of training data. To address this issue, we introduce an approach to generate extensive Lean 4 proof data derived from high-school and undergraduate-level mathematical competition problems. This approach involves translating natural language problems into formal statements, filtering out low-quality statements, and generating proofs to create synthetic data. After fine-tuning the DeepSeekMath 7B model on this synthetic dataset, which comprises 8 million formal statements with proofs, our model achieved whole-proof generation accuracies of 46.3% with 64 samples and 52% cumulatively on the Lean 4 miniF2F test, surpassing the baseline GPT-4 at 23.0% with 64 samples and a tree search reinforcement learning method at 41.0%. Additionally, our model successfully proved 5 out of 148 problems in the Lean 4 Formalized International Mathematical Olympiad (FIMO) benchmark, while GPT-4 failed to prove any. These results demonstrate the potential of leveraging large-scale synthetic data to enhance theorem-proving capabilities in LLMs. Both the synthetic dataset and the model will be made available to facilitate further research in this promising field.',
      'PubMed and beyond: biomedical literature search in the age of artificial intelligence|Qiao Jin · Robert Leaman · Zhiyong Lu|EBioMedicine|2023|105|Summary Biomedical research yields vast information, much of which is only accessible through the literature. Consequently, literature search is crucial for healthcare and biomedicine. Recent improvements in artificial intelligence (AI) have expanded functionality beyond keywords, but they might be unfamiliar to clinicians and researchers. In response, we present an overview of over 30 literature search tools tailored to common biomedical use cases, aiming at helping readers efficiently fulfill their information needs. We first discuss recent improvements and continued challenges of the widely used PubMed. Then, we describe AI-based literature search tools catering to five specific information needs: 1. Evidence-based medicine. 2. Precision medicine and genomics. 3. Searching by meaning, including questions. 4. Finding related articles with literature recommendation. 5. Discovering hidden associations through literature mining. Finally, we discuss the impacts of recent developments of large language models such as ChatGPT on biomedical information seeking.',
      'Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning|Anja Surina · Amin Mansouri · Lars Quaedvlieg · Amal Seddas · Maryna Viazovska · Emmanuel Abbe · Caglar Gulcehre|arXiv.org|2025|45|Discovering efficient algorithms for solving complex problems has been an outstanding challenge in mathematics and computer science, requiring substantial human expertise over the years. Recent advancements in evolutionary search with large language models (LLMs) have shown promise in accelerating the discovery of algorithms across various domains, particularly in mathematics and optimization. However, existing approaches treat the LLM as a static generator, missing the opportunity to update the model with the signal obtained from evolutionary exploration. In this work, we propose to augment LLM-based evolutionary search by continuously refining the search operator - the LLM - through reinforcement learning (RL) fine-tuning. Our method leverages evolutionary search as an exploration strategy to discover improved algorithms, while RL optimizes the LLM policy based on these discoveries. Our experiments on combinatorial optimization tasks demonstrate that integrating RL with evolutionary search accelerates the discovery of superior algorithms, showcasing the potential of RL-enhanced evolutionary strategies for algorithm design.',
      'MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering|Jun Shern Chan · Neil Chowdhury · Oliver Jaffe · James Aung · Dane Sherburn · E. Mays · Giulio Starace · Kevin Liu · Leon Maksin · Tejal Patwardhan · Lilian Weng · Aleksander Mkadry|arXiv.org|2024|290|We introduce MLE-bench, a benchmark for measuring how well AI agents perform at machine learning engineering. To this end, we curate 75 ML engineering-related competitions from Kaggle, creating a diverse set of challenging tasks that test real-world ML engineering skills such as training models, preparing datasets, and running experiments. We establish human baselines for each competition using Kaggle’s publicly available leaderboards. We use open-source agent scaffolds to evaluate several frontier language models on our benchmark, finding that the best-performing setup--OpenAI’s o1-preview with AIDE scaffolding--achieves at least the level of a Kaggle bronze medal in 16.9% of competitions. In addition to our main results, we investigate various forms of resource scaling for AI agents and the impact of contamination from pre-training. We open-source our benchmark code (github.com/openai/mle-bench/) to facilitate future research in understanding the ML engineering capabilities of AI agents.',
      'MicroVQA: A Multimodal Reasoning Benchmark for Microscopy-Based Scientific Research|James Burgess · Jeffrey J. Nirschl · Laura Bravo-S’anchez · Alejandro Lozano · S. Gupte · Jesús G. Galaz-Montoya · Yuhui Zhang · Yuchang Su · Disha Bhowmik · Zachary Coman · S. Hasan · Alexandra Johannesson · William D. Leineweber · Malvika G. Nair · Ridhi Yarlagadda · Connor R Zuraski · Wah Chiu · S. Cohen · Jan N. Hansen · Manuel D. Leonetti · Chad Liu · Emma Lundberg · S. Yeung-Levy|Computer Vision and Pattern Recognition|2025|31|Scientific research demands sophisticated reasoning over multimodal data, a challenge especially prevalent in biology. Despite recent advances in multimodal large language models (MLLMs) for AI-assisted research, existing multimodal reasoning benchmarks only target up to college-level difficulty, while research-level benchmarks emphasize lower-level perception, falling short of the complex multimodal reasoning needed for scientific discovery. To bridge this gap, we introduce MicroVQA, a visual-question answering (VQA) benchmark designed to assess three reasoning capabilities vital in research workflows: expert image understanding, hypothesis generation, and experiment proposal. MicroVQA consists of 1,042 multiple-choice questions (MCQs) curated by biology experts across diverse microscopy modalities, ensuring VQA samples represent real scientific practice. In constructing the benchmark, we find that standard MCQ generation methods induce language shortcuts, motivating a new two-stage pipeline: an optimized LLM prompt structures question-answer pairs into MCQs; then, an agent-based ‘RefineBot’ updates them to remove shortcuts. Benchmarking on state-of-the-art MLLMs reveal a peak performance of 53%; models with smaller LLMs only slightly underperform top models, suggesting that language-based reasoning is less challenging than multimodal reasoning; and tuning with scientific articles enhances performance. Expert analysis of chain-of-thought responses shows that perception errors are the most frequent, followed by knowledge errors and then overgeneralization errors. These insights highlight the challenges in multimodal scientific reasoning, showing MicroVQA is a valuable resource advancing AI-driven biomedical research. MicroVQA is available here, project here.',
      'AI-driven multi-omics integration for multi-scale predictive modeling of genotype-environment-phenotype relationships|You Wu · Lei Xie|Computational and Structural Biotechnology Journal|2024|136|Despite the wealth of single-cell multi-omics data, it remains challenging to predict the consequences of novel genetic and chemical perturbations in the human body. It requires knowledge of molecular interactions at all biological levels, encompassing disease models and humans. Current machine learning methods primarily establish statistical correlations between genotypes and phenotypes but struggle to identify physiologically significant causal factors, limiting their predictive power. Key challenges in predictive modeling include scarcity of labeled data, generalization across different domains, and disentangling causation from correlation. In light of recent advances in multi-omics data integration, we propose a new artificial intelligence (AI)-powered biology-inspired multi-scale modeling framework to tackle these issues. This framework will integrate multi-omics data across biological levels, organism hierarchies, and species to predict genotype-environment-phenotype relationships under various conditions. AI models inspired by biology may identify novel molecular targets, biomarkers, pharmaceutical agents, and personalized medicines for presently unmet medical needs.',
      'Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents|Junkai Li · Siyu Wang · Mengqi Zhang · Weitao Li · Yunghwei Lai · Xin Kang · Weizhi Ma · Yang Liu|arXiv.org|2024|278|The recent rapid development of large language models (LLMs) has sparked a new wave of technological revolution in medical artificial intelligence (AI). While LLMs are designed to understand and generate text like a human, autonomous agents that utilize LLMs as their"brain"have exhibited capabilities beyond text processing such as planning, reflection, and using tools by enabling their"bodies"to interact with the environment. We introduce a simulacrum of hospital called Agent Hospital that simulates the entire process of treating illness, in which all patients, nurses, and doctors are LLM-powered autonomous agents. Within the simulacrum, doctor agents are able to evolve by treating a large number of patient agents without the need to label training data manually. After treating tens of thousands of patient agents in the simulacrum (human doctors may take several years in the real world), the evolved doctor agents outperform state-of-the-art medical agent methods on the MedQA benchmark comprising US Medical Licensing Examination (USMLE) test questions. Our methods of simulacrum construction and agent evolution have the potential in benefiting a broad range of applications beyond medical AI.',
      'Generating dermatopathology reports from gigapixel whole slide images with HistoGPT|M. Tran · P. Schmidle · R. Guo · S. Wagner · V. Koch · V. Lupperger · Brenna Novotny · Dennis H. Murphree · H. Hardway · Marina D’Amato · Judith Lefkes · D. Geijs · A. Feuchtinger · A. Böhner · R. Kaczmarczyk · T. Biedermann · Avital L. Amir · A. Mooyaart · Francesco Ciompi · G. Litjens · Chen Wang · N. Comfere · K. Eyerich · S. A. Braun · Carsten Marr · T. Peng|Nature Communications|2025|40|Histopathology is the reference standard for diagnosing the presence and nature of many diseases, including cancer. However, analyzing tissue samples under a microscope and summarizing the findings in a comprehensive pathology report is time-consuming, labor-intensive, and non-standardized. To address this problem, we present HistoGPT, a vision language model that generates pathology reports from a patient’s multiple full-resolution histology images. It is trained on 15,129 whole slide images from 6705 dermatology patients with corresponding pathology reports. The generated reports match the quality of human-written reports for common and homogeneous malignancies, as confirmed by natural language processing metrics and domain expert analysis. We evaluate HistoGPT in an international, multi-center clinical study and show that it can accurately predict tumor subtypes, tumor thickness, and tumor margins in a zero-shot fashion. Our model demonstrates the potential of artificial intelligence to assist pathologists in evaluating, reporting, and understanding routine dermatopathology cases. Machine learning models represent an opportunity for the automatic generation of histopathology reports. Here, the authors develop HistoGPT, a vision language model that can generate reports from multiple gigapixel-sized whole slide images and also predict tumour thickness, subtypes, and margins, among other diseases.',
      'QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training|Wei Dai · Peilin Chen · C. Ekbote · Paul Pu Liang|arXiv.org|2025|38|Clinical decision-making routinely demands reasoning over heterogeneous data, yet existing multimodal language models (MLLMs) remain largely vision-centric and fail to generalize across clinical specialties. To bridge this gap, we introduce QoQ-Med-7B/32B, the first open generalist clinical foundation model that jointly reasons across medical images, time-series signals, and text reports. QoQ-Med is trained with Domain-aware Relative Policy Optimization (DRPO), a novel reinforcement-learning objective that hierarchically scales normalized rewards according to domain rarity and modality difficulty, mitigating performance imbalance caused by skewed clinical data distributions. Trained on 2.61 million instruction tuning pairs spanning 9 clinical domains, we show that DRPO training boosts diagnostic performance by 43% in macro-F1 on average across all visual domains as compared to other critic-free training methods like GRPO. Furthermore, with QoQ-Med trained on intensive segmentation data, it is able to highlight salient regions related to the diagnosis, with an IoU 10x higher than open models while reaching the performance of OpenAI o4-mini. To foster reproducibility and downstream research, we release (i) the full model weights, (ii) the modular training pipeline, and (iii) all intermediate reasoning traces at https://github.com/DDVD233/QoQ_Med.',
      'BLADE: Benchmarking Language Model Agents for Data-Driven Science|Ken Gu · Ruoxi Shang · Ruien Jiang · Keying Kuang · Richard Lin · Donghe Lyu · Yue Mao · Youran Pan · Teng Wu · Jiaqian Yu · Yikun Zhang · Tianmai M. Zhang · Lanyi Zhu · Mike A. Merrill · J. Heer · Tim Althoff|Conference on Empirical Methods in Natural Language Processing|2024|62|Data-driven scientific discovery requires the iterative integration of scientific domain knowledge, statistical expertise, and an understanding of data semantics to make nuanced analytical decisions, e.g., about which variables, transformations, and statistical models to consider. LM-based agents equipped with planning, memory, and code execution capabilities have the potential to support data-driven science. However, evaluating agents on such open-ended tasks is challenging due to multiple valid approaches, partially correct steps, and different ways to express the same decisions. To address these challenges, we present BLADE, a benchmark to automatically evaluate agents’multifaceted approaches to open-ended research questions. BLADE consists of 12 datasets and research questions drawn from existing scientific literature, with ground truth collected from independent analyses by expert data scientists and researchers. To automatically evaluate agent responses, we developed corresponding computational methods to match different representations of analyses to this ground truth. Though language models possess considerable world knowledge, our evaluation shows that they are often limited to basic analyses. However, agents capable of interacting with the underlying data demonstrate improved, but still non-optimal, diversity in their analytical decision making. Our work enables the evaluation of agents for data-driven science and provides researchers deeper insights into agents’analysis approaches.',
      'DeepAnalyze: Agentic Large Language Models for Autonomous Data Science|Shaolei Zhang · Ju Fan · Meihao Fan · Guoliang Li · Xiaoyong Du|arXiv.org|2025|37|Autonomous data science, from raw data sources to analyst-grade deep research reports, has been a long-standing challenge, and is now becoming feasible with the emergence of powerful large language models (LLMs). Recent workflow-based data agents have shown promising results on specific data tasks but remain fundamentally limited in achieving fully autonomous data science due to their reliance on predefined workflows. In this paper, we introduce DeepAnalyze-8B, the first agentic LLM designed for autonomous data science, capable of automatically completing the end-toend pipeline from data sources to analyst-grade deep research reports. To tackle high-complexity data science tasks, we propose a curriculum-based agentic training paradigm that emulates the learning trajectory of human data scientists, enabling LLMs to progressively acquire and integrate multiple capabilities in real-world environments. We also introduce a data-grounded trajectory synthesis framework that constructs high-quality training data. Through agentic training, DeepAnalyze learns to perform a broad spectrum of data tasks, ranging from data question answering and specialized analytical tasks to open-ended data research. Experiments demonstrate that, with only 8B parameters, DeepAnalyze outperforms previous workflow-based agents built on most advanced proprietary LLMs. The model, code, and training data of DeepAnalyze are open-sourced, paving the way toward autonomous data science.',
      'LLMs Accelerate Annotation for Medical Information Extraction|Akshay Goel · Almog Gueta · Omry Gilon · Chang Liu · Sofia Erell · Lan Nguyen · Xiaohong Hao · Bolous Jaber · Shashir Reddy · Rupesh Kartha · Jean L. Steiner · Itay Laish · Amir Feder|ML4H@NeurIPS|2023|193|The unstructured nature of clinical notes within electronic health records often conceals vital patient-related information, making it challenging to access or interpret. To uncover this hidden information, specialized Natural Language Processing (NLP) models are required. However, training these models necessitates large amounts of labeled data, a process that is both time-consuming and costly when relying solely on human experts for annotation. In this paper, we propose an approach that combines Large Language Models (LLMs) with human expertise to create an efficient method for generating ground truth labels for medical text annotation. By utilizing LLMs in conjunction with human annotators, we significantly reduce the human annotation burden, enabling the rapid creation of labeled datasets. We rigorously evaluate our method on a medical information extraction task, demonstrating that our approach not only substantially cuts down on human intervention but also maintains high accuracy. The results highlight the potential of using LLMs to improve the utilization of unstructured clinical data, allowing for the swift deployment of tailored NLP solutions in healthcare.',
      'Enhancing Knowledge Graph Construction Using Large Language Models|Milena Trajanoska · Riste Stojanov · D. Trajanov|arXiv.org|2023|96|The growing trend of Large Language Models (LLM) development has attracted significant attention, with models for various applications emerging consistently. However, the combined application of Large Language Models with semantic technologies for reasoning and inference is still a challenging task. This paper analyzes how the current advances in foundational LLM, like ChatGPT, can be compared with the specialized pretrained models, like REBEL, for joint entity and relation extraction. To evaluate this approach, we conducted several experiments using sustainability-related text as our use case. We created pipelines for the automatic creation of Knowledge Graphs from raw texts, and our findings indicate that using advanced LLM models can improve the accuracy of the process of creating these graphs from unstructured text. Furthermore, we explored the potential of automatic ontology creation using foundation LLM models, which resulted in even more relevant and accurate knowledge graphs.',
      'A scoping review of using Large Language Models (LLMs) to investigate Electronic Health Records (EHRs)|Lingyao Li · Jiayan Zhou · Zhenxiang Gao · Wenyue Hua · Lizhou Fan · Huizi Yu · Loni Hagen · Yonfeng Zhang · T. Assimes · Libby Hemphill · Siyuan Ma|arXiv.org|2024|83|Electronic Health Records (EHRs) play an important role in the healthcare system. However, their complexity and vast volume pose significant challenges to data interpretation and analysis. Recent advancements in Artificial Intelligence (AI), particularly the development of Large Language Models (LLMs), open up new opportunities for researchers in this domain. Although prior studies have demonstrated their potential in language understanding and processing in the context of EHRs, a comprehensive scoping review is lacking. This study aims to bridge this research gap by conducting a scoping review based on 329 related papers collected from OpenAlex. We first performed a bibliometric analysis to examine paper trends, model applications, and collaboration networks. Next, we manually reviewed and categorized each paper into one of the seven identified topics: named entity recognition, information extraction, text similarity, text summarization, text classification, dialogue system, and diagnosis and prediction. For each topic, we discussed the unique capabilities of LLMs, such as their ability to understand context, capture semantic relations, and generate human-like text. Finally, we highlighted several implications for researchers from the perspectives of data resources, prompt engineering, fine-tuning, performance measures, and ethical concerns. In conclusion, this study provides valuable insights into the potential of LLMs to transform EHR research and discusses their applications and ethical considerations.'
    ];
    return raw.map(s => { const p = s.split('|'); return { title: p[0], authors: p[1], venue: p[2], year: +p[3], cites: +p[4], abs: p[5] }; });
  }
  venueClass(name) {
    const n = (name || '').toLowerCase();
    if (/arxiv|biorxiv|medrxiv|preprint|ssrn|openreview/.test(n)) return 'preprint';
    if (/neurips|nips|iclr|icml|acl|emnlp|naacl|cvpr|iccv|eccv|focs|stoc|soda|aaai|ijcai|kdd|sigir|chi|uist|conference|proceedings|workshop|symposium/.test(n)) return 'conference';
    return 'journal';
  }
  venueColors() { return ['var(--muted2)', 'var(--muted2)']; }
  paintVenue() { /* venue tone is fixed in paper-row.js contract */ }
  _paintVenueLegacy(row) {
    const name = (row.dataset.venue || '').trim();
    const [dot, fg] = this.venueColors();
    const spans = Array.from(row.querySelectorAll(':scope > div > span'));
    const dEl = spans.find(x => !x.children.length && !x.textContent.trim() && (x.style.borderRadius === '50%' || /border-radius:\s*50%/.test(x.getAttribute('style') || '')));
    if (dEl) dEl.style.background = dot;
    const lEl = spans.find(x => x.textContent.trim().toLowerCase() === name.toLowerCase());
    if (lEl) lEl.style.color = fg;
  }
  expandMock(sidebar) {
    if (sidebar._expanded) { sidebar.querySelectorAll('.pr').forEach(r => this.paintVenue(r)); return; }
    sidebar._expanded = true;
    const rows = Array.from(sidebar.querySelectorAll('.pr'));
    const tpl = rows[0];
    if (!tpl) return;
    const list = tpl.parentElement;
    let anchor = rows[rows.length - 1];
    this.mockPapers().forEach(p => {
      const c = tpl.cloneNode(true);
      c.dataset.venue = p.venue; c.dataset.year = p.year; c.dataset.cites = p.cites;
      c.dataset.url = 'https://arxiv.org'; c.dataset.abstract = p.abs;
      const t = c.querySelector('.pr-title'); if (t) t.textContent = p.title;
      const a = c.querySelector('.pr-authors'); if (a) a.textContent = p.authors;
      const spans = Array.from(c.querySelectorAll('span'));
      const tv = (tpl.dataset.venue || '').trim().toLowerCase();
      const vEl = spans.find(s => !s.children.length && s.textContent.trim().toLowerCase() === tv);
      if (vEl) vEl.textContent = p.venue;
      const cites = p.cites >= 1000 ? (p.cites / 1000).toFixed(1).replace(/\.0$/, '') + 'k' : String(p.cites);
      const mEl = spans.find(s => s.querySelector('b') && /cites/.test(s.textContent));
      if (mEl) mEl.innerHTML = '<span>' + p.year + '</span><span style="color:var(--dot-sep);"></span><span style="color:var(--fg2);"><b style="color:var(--fg);">' + cites + '</b> cites</span>';
      list.insertBefore(c, anchor.nextSibling);
      anchor = c;
    });
    sidebar.querySelectorAll('.pr').forEach(r => this.prPaint(r));
    if (!window.KLPaperRow) document.addEventListener('kl-paper-row-ready', () => sidebar.querySelectorAll('.pr').forEach(r => this.prPaint(r)), { once: true });
  }
  prPaint(r) {
    if (!window.KLPaperRow || r.dataset.prPainted === '1') return;
    const p = { title: (r.querySelector('.pr-title') || {}).textContent || '', authors: (r.querySelector('.pr-authors') || {}).textContent || '', venue: r.dataset.venue || '', year: r.dataset.year || '', cites: +r.dataset.cites || 0 };
    // one owner for the star: the repaint emits it, so any earlier JS-made copy goes
    const keep = Array.from(r.children).filter(c => (c.tagName === 'BUTTON' || c.dataset.keep === '1') && !c.classList.contains('pr-star'));
    r.querySelectorAll('.pr-star').forEach(s => s.remove());
    r.innerHTML = window.KLPaperRow.prCardInner(p, { railClass: 'rail', star: true, emphasis: this.props.cardEmphasis || 'Venue ink' });
    keep.forEach(c => r.appendChild(c));
    r.dataset.prPainted = '1';
  }
  paintPager(pager, pg, PAGE, total, count, api) {
    if (total <= PAGE) { pager.style.display = 'none'; pager.innerHTML = ''; return; }
    const S = pg.style || 'Numbered bar';
    const pages = Math.max(1, Math.ceil(total / PAGE));
    const flow = S === 'Load more' || S === 'Infinite scroll';
    const from = flow ? 1 : (pg.page - 1) * PAGE + 1;
    const range = from + '–' + (from + count - 1) + ' of ' + total;
    const base = 'flex:none; padding:9px 14px; border-top:1px solid var(--divider); background:var(--dock-bg); display:flex; align-items:center;';
    const BTN = 'display:inline-flex; align-items:center; justify-content:center; min-width:27px; height:27px; padding:0 6px; border-radius:8px; border:1px solid transparent; background:none; font-family:inherit; font-size:12.5px; font-variant-numeric:tabular-nums; color:var(--muted); cursor:pointer;';
    const ON = 'background:var(--fg); border-color:var(--fg); color:var(--dock-bg); font-weight:600;';
    const ARR = BTN + ' border-color:var(--border); color:var(--fg2); padding:0;';
    const MUTED = 'font-size:11.5px; color:var(--muted); font-variant-numeric:tabular-nums;';
    const arrow = d => '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M' + (d < 0 ? '14 7l-5 5 5 5' : '10 7l5 5-5 5') + '"></path></svg>';
    pager.style.display = '';
    if (S === 'Load more') {
      const done = count >= total, rem = Math.min(PAGE, total - count);
      pager.setAttribute('style', base + ' flex-direction:column; gap:8px; padding:11px 14px 12px;');
      pager.innerHTML = '<div style="width:100%; height:3px; border-radius:2px; background:var(--border); overflow:hidden;"><div style="height:100%; width:' + Math.round(count / total * 100) + '%; background:var(--fg); border-radius:2px; transition:width .3s ease;"></div></div>'
        + (done
          ? '<div style="display:flex; align-items:center; justify-content:space-between; width:100%;"><span style="' + MUTED + '">All ' + total + ' papers loaded</span><button class="sb-pgb" data-top="1" style="' + BTN + ' color:var(--fg2); font-weight:600;">↑ Top</button></div>'
          : '<button class="sb-pgmore" data-more="1" style="width:100%; padding:9px; border:1px solid var(--border); border-radius:10px; background:var(--card); font-family:inherit; font-size:13px; font-weight:600; color:var(--fg); cursor:pointer;">Load ' + rem + ' more</button><span style="' + MUTED + '">Showing ' + count + ' of ' + total + '</span>');
    } else if (S === 'Infinite scroll') {
      const done = count >= total;
      pager.setAttribute('style', base + ' justify-content:space-between; gap:8px; min-height:45px;');
      pager.innerHTML = (pg._loading
        ? '<span style="' + MUTED + ' display:flex; align-items:center; gap:7px;"><span style="width:6px; height:6px; border-radius:50%; background:var(--accent); animation:sbPulse 1s ease-in-out infinite;"></span>Loading more…</span>'
        : '<span style="' + MUTED + '">' + (done ? 'End of results  ' + total + ' papers' : range + '  scroll for more') + '</span>')
        + (count > PAGE ? '<button class="sb-pgb" data-top="1" style="' + BTN + ' color:var(--fg2); font-weight:600; padding:0 8px;">↑ Top</button>' : '<span></span>');
    } else if (S === 'Compact stepper') {
      pager.setAttribute('style', base + ' justify-content:space-between; gap:8px;');
      pager.innerHTML = '<div style="display:flex; align-items:center; gap:7px;">'
        + '<button class="sb-pgb" data-go="' + (pg.page - 1) + '" style="' + ARR + ' border-radius:50%;"' + (pg.page === 1 ? ' disabled' : '') + '>' + arrow(-1) + '</button>'
        + '<span style="font-size:12.5px; color:var(--fg2); font-variant-numeric:tabular-nums;">Page <b style="font-weight:600; color:var(--fg);">' + pg.page + '</b> of ' + pages + '</span>'
        + '<button class="sb-pgb" data-go="' + (pg.page + 1) + '" style="' + ARR + ' border-radius:50%;"' + (pg.page === pages ? ' disabled' : '') + '>' + arrow(1) + '</button></div>'
        + '<span style="' + MUTED + '">' + range + '</span>';
    } else if (S === 'Segmented rail') {
      const seg = [];
      if (pages <= 7) { for (let i = 1; i <= pages; i++) seg.push(i); }
      else { const s = Math.max(2, Math.min(pg.page - 2, pages - 5)); seg.push(1); for (let i = s; i < s + 5; i++) seg.push(i); seg.push(pages); }
      pager.setAttribute('style', base + ' flex-direction:column; gap:7px; padding:10px 14px 11px;');
      pager.innerHTML = '<div style="display:flex; width:100%; gap:2px; padding:3px; background:var(--icon-bg); border-radius:10px;">'
        + seg.map(n => '<button class="sb-pgb" data-go="' + n + '" data-on="' + (n === pg.page ? 1 : 0) + '" style="flex:1; min-width:0; height:26px; border:none; border-radius:7px; font-family:inherit; font-size:12px; font-variant-numeric:tabular-nums; cursor:pointer; ' + (n === pg.page ? 'background:var(--card); color:var(--fg); font-weight:600; box-shadow:0 1px 2px rgba(var(--sh),.16);' : 'background:none; color:var(--muted);') + '">' + n + '</button>').join('')
        + '</div><span style="' + MUTED + '">' + range + '</span>';
    } else {
      const items = [];
      const add = n => { if (n >= 1 && n <= pages && !items.includes(n)) items.push(n); };
      add(1); for (let i = pg.page - 1; i <= pg.page + 1; i++) if (i > 1 && i < pages) add(i); add(pages);
      items.sort((a, b) => a - b);
      let nums = '';
      items.forEach((n, i) => {
        if (i && n - items[i - 1] > 1) nums += '<span style="' + MUTED + ' padding:0 1px;">…</span>';
        nums += '<button class="sb-pgb" data-go="' + n + '" data-on="' + (n === pg.page ? 1 : 0) + '" style="' + BTN + (n === pg.page ? ON : '') + '">' + n + '</button>';
      });
      pager.setAttribute('style', base + ' justify-content:space-between; gap:8px;');
      pager.innerHTML = '<span style="' + MUTED + '">' + range + '</span><div style="display:flex; align-items:center; gap:3px;">'
        + '<button class="sb-pgb" data-go="' + (pg.page - 1) + '" style="' + ARR + '"' + (pg.page === 1 ? ' disabled' : '') + '>' + arrow(-1) + '</button>' + nums
        + '<button class="sb-pgb" data-go="' + (pg.page + 1) + '" style="' + ARR + '"' + (pg.page === pages ? ' disabled' : '') + '>' + arrow(1) + '</button></div>';
    }
    pager.querySelectorAll('[data-go]').forEach(b => b.addEventListener('click', () => {
      const n = +b.dataset.go;
      if (n >= 1 && n <= pages && n !== pg.page) api.goTo(n);
    }));
    const more = pager.querySelector('[data-more]');
    if (more) more.addEventListener('click', () => api.loadMore());
    const top = pager.querySelector('[data-top]');
    if (top) top.addEventListener('click', () => { if (api.listEl) api.listEl.scrollTo({ top: 0, behavior: 'smooth' }); });
  }
  wireSidebar(sidebar, mode) {
    this.sbImpWire(sidebar);
    const searchEl = sidebar.querySelector('.sb-search');
    const countEl = sidebar.querySelector('.sb-count');
    const emptyEl = sidebar.querySelector('.sb-empty');
    const errorEl = sidebar.querySelector('.sb-error');
    const rowEls = Array.from(sidebar.querySelectorAll('.pr'));
    const listEl = rowEls[0] ? rowEls[0].parentElement : sidebar;
    const isCards = sidebar.dataset.style === 'cards';
    if (listEl && listEl.classList.contains('sb-list')) {
      listEl.style.paddingBottom = isCards ? '10px' : '10px';
      const FADE = 22;
      // gradient scrims, NOT mask-image: a masked scroller is composited and
      // resampled (blurry text on fractional-DPR displays)
      if (getComputedStyle(sidebar).position === 'static') sidebar.style.position = 'relative';
      listEl.style.webkitMaskImage = 'none';
      listEl.style.maskImage = 'none';
      const mkScrim = dir => {
        const s = document.createElement('div');
        s.className = 'sb-scrim';
        s.style.cssText = 'position:absolute; left:0; right:0; height:' + FADE + 'px; pointer-events:none; z-index:3; opacity:0; transition:opacity .18s ease;' +
          'background:linear-gradient(to ' + dir + ', var(--panel), color-mix(in oklab, var(--panel) 55%, transparent) 55%, transparent);';
        sidebar.appendChild(s);
        return s;
      };
      const scrimTop = mkScrim('bottom'), scrimBot = mkScrim('top');
      const updateFade = () => {
        const top = listEl.offsetTop, h = listEl.clientHeight;
        scrimTop.style.top = top + 'px';
        scrimBot.style.top = (top + h - FADE) + 'px';
        scrimTop.style.opacity = listEl.scrollTop > 4 ? '1' : '0';
        scrimBot.style.opacity = (listEl.scrollTop + h < listEl.scrollHeight - 4) ? '1' : '0';
      };
      listEl.addEventListener('scroll', updateFade);
      if (window.ResizeObserver) new ResizeObserver(updateFade).observe(listEl);
      listEl._updateFade = updateFade;
      updateFade();
    }
    const YMIN = 2018, YMAX = 2024, CMIN = 0, CMAX = 20000;
    const F = { venue: '', title: '', abstract: '', author: '', yearLo: YMIN, yearHi: YMAX, citesLo: CMIN, citesHi: CMAX };
    const fmtK = n => this.kfmt(n);
    const field = (r, key) => {
      if (key === 'venue') return (r.dataset.venue || '').toLowerCase();
      if (key === 'title') return (r.querySelector('.pr-title')?.textContent || '').toLowerCase();
      if (key === 'abstract') return (r.dataset.abstract || '').toLowerCase();
      if (key === 'author') return (r.querySelector('.pr-authors')?.textContent || '').toLowerCase();
      return '';
    };
    const parseTerms = g => g.match(/"[^"]*"|\S+/g) || [];
    const termMatch = (text, term) => {
      if (/^".*"$/.test(term)) { const p = term.slice(1, -1).toLowerCase(); return !p || text.includes(p); }
      if (term.includes('*')) { const re = new RegExp(term.toLowerCase().replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*')); return re.test(text); }
      return text.includes(term.toLowerCase());
    };
    const exprMatch = (text, query) => {
      query = (query || '').trim();
      if (!query) return true;
      text = (text || '').toLowerCase();
      return query.split(/\s+OR\s+/).some(g => parseTerms(g).filter(t => t !== 'AND').every(t => termMatch(text, t)));
    };
    const matches = r => {
      if (r.dataset.seeded === '1') return false;
      if (!exprMatch(field(r, 'title') + ' ' + field(r, 'venue') + ' ' + field(r, 'author'), searchEl.value)) return false;
      if (!exprMatch(field(r, 'venue'), F.venue)) return false;
      if (!exprMatch(field(r, 'title'), F.title)) return false;
      if (!exprMatch(field(r, 'abstract'), F.abstract)) return false;
      if (!exprMatch(field(r, 'author'), F.author)) return false;
      const y = +r.dataset.year, c = +r.dataset.cites;
      if (y < F.yearLo || y > F.yearHi) return false;
      if (c < F.citesLo) return false;
      if (F.citesHi < CMAX && c > F.citesHi) return false;
      return true;
    };
    const activeCount = () => {
      let n = 0;
      if (F.venue) n++; if (F.title) n++; if (F.abstract) n++; if (F.author) n++;
      if (F.yearLo > YMIN || F.yearHi < YMAX) n++;
      if (F.citesLo > CMIN || F.citesHi < CMAX) n++;
      return n;
    };
    const fbtn = sidebar.querySelector('.sb-filter-btn');
    const flabel = sidebar.querySelector('.sb-filter-label');
    const updateBadge = () => {
      const n = activeCount();
      if (flabel) flabel.textContent = n ? ('Filter  ' + n) : 'Filter';
      if (fbtn) { fbtn.style.borderColor = n ? 'var(--fg)' : 'var(--border)'; fbtn.style.color = n ? 'var(--fg)' : 'var(--fg2)'; }
    };
    const PAGE = 10;
    const pg = { page: 1, loaded: PAGE, sig: null, style: 'Compact stepper' };
    const pager = document.createElement('div');
    pager.className = 'sb-pager';
    pager.style.display = 'none';
    sidebar.appendChild(pager);
    const isFlow = () => pg.style === 'Load more' || pg.style === 'Infinite scroll';
    const paginate = shown => {
      if (shown.length <= PAGE) return shown;
      let rows;
      if (isFlow()) {
        pg.loaded = Math.max(PAGE, Math.min(pg.loaded, shown.length));
        rows = shown.slice(0, pg.loaded);
      } else {
        const pages = Math.max(1, Math.ceil(shown.length / PAGE));
        pg.page = Math.min(Math.max(1, pg.page), pages);
        rows = shown.slice((pg.page - 1) * PAGE, pg.page * PAGE);
      }
      const keep = new Set(rows);
      shown.forEach(r => { if (!keep.has(r)) r.style.display = 'none'; });
      return rows;
    };
    const pagerApi = {
      listEl,
      goTo: p => { pg.page = p; apply(); if (listEl) listEl.scrollTop = 0; },
      loadMore: () => { pg.loaded += PAGE; apply(); }
    };
    const apply = () => {
      const state = sidebar.dataset.state || 'normal';
      if (state === 'error') {
        rowEls.forEach(r => { r.style.display = 'none'; });
        if (emptyEl) emptyEl.style.display = 'none';
        if (errorEl) errorEl.style.display = 'flex';
        countEl.textContent = '–';
        pager.style.display = 'none';
        return;
      }
      if (errorEl) errorEl.style.display = 'none';
      const forceEmpty = state === 'empty';
      rowEls.forEach(r => { r.style.display = (!forceEmpty && matches(r)) ? '' : 'none'; });
      const shown = rowEls.filter(r => r.style.display !== 'none');
      const sig = searchEl.value + '|' + JSON.stringify(F) + '|' + state;
      if (pg.sig !== sig) { pg.sig = sig; pg.page = 1; pg.loaded = PAGE; }
      const pageRows = paginate(shown);
      if (!isCards) {
        rowEls.forEach(r => { r.style.borderBottom = '1px solid var(--divider)'; });
        if (pageRows.length) pageRows[pageRows.length - 1].style.borderBottom = 'none';
      }
      if (emptyEl) emptyEl.style.display = shown.length ? 'none' : 'flex';
      countEl.textContent = shown.length + (shown.length === 1 ? ' paper' : ' papers');
      updateBadge();
      const animKey = sig + '|' + pg.page + '|' + pg.loaded;
      if (pager._animKey && pager._animKey !== animKey) {
        const start = isFlow() ? Math.min(pager._animN || 0, pageRows.length) : 0;
        pageRows.forEach((r, i) => { if (i >= start && r.animate) r.animate([{ opacity: 0, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }], { duration: 200, delay: Math.min((i - start) * 20, 180), easing: 'ease-out', fill: 'backwards' }); });
      }
      pager._animKey = animKey; pager._animN = pageRows.length;
      this.paintPager(pager, pg, PAGE, shown.length, pageRows.length, pagerApi);
      if (listEl._updateFade) listEl._updateFade();
    };
    sidebar._applyList = apply;
    sidebar._setPager = s => { pg.style = s; pg.page = 1; pg.loaded = PAGE; pg._loading = false; apply(); if (listEl) listEl.scrollTop = 0; };
    if (listEl) listEl.addEventListener('scroll', () => {
      if (pg.style !== 'Infinite scroll' || pg._loading) return;
      if (listEl.scrollTop + listEl.clientHeight < listEl.scrollHeight - 70) return;
      if (pg.loaded >= rowEls.filter(matches).length) return;
      pg._loading = true; apply();
      setTimeout(() => { pg._loading = false; pg.loaded += PAGE; apply(); }, 420);
    });

    // custom sort dropdown
    let sortMode = 'cites';
    const sort = () => {
      const ordered = rowEls.slice().sort((a, b) => {
        if (sortMode === 'cites') return (+b.dataset.cites) - (+a.dataset.cites);
        if (sortMode === 'year') return (+b.dataset.year) - (+a.dataset.year);
        if (sortMode === 'title') return a.querySelector('.pr-title').textContent.localeCompare(b.querySelector('.pr-title').textContent);
        return 0;
      });
      ordered.forEach(r => listEl.appendChild(r));
      rowEls.length = 0;
      ordered.forEach(r => rowEls.push(r));
      pg.page = 1; pg.loaded = PAGE;
    };
    const sortBtn = sidebar.querySelector('.sb-sort-btn');
    const sortMenu = sidebar.querySelector('.sb-sort-menu');
    const sortLabel = sidebar.querySelector('.sb-sort-label');
    if (sortBtn && sortMenu) {
      sortBtn.addEventListener('click', e => { e.stopPropagation(); const open = sortMenu.style.display === 'block'; sortMenu.style.display = open ? 'none' : 'block'; if (!open) this.popIn(sortMenu); });
      document.addEventListener('click', () => { sortMenu.style.display = 'none'; });
      sortMenu.addEventListener('click', e => e.stopPropagation());
      sortMenu.querySelectorAll('.sb-opt').forEach(opt => {
        opt.addEventListener('click', () => {
          sortMode = opt.dataset.value;
          sortLabel.textContent = opt.textContent.replace('✓', '').trim();
          sortMenu.querySelectorAll('.sb-check').forEach(ch => { ch.style.visibility = 'hidden'; });
          opt.querySelector('.sb-check').style.visibility = 'visible';
          sortMenu.style.display = 'none';
          sort(); pager._animKey = 'resort'; apply();
        });
      });
    }

    // Remote-search semantics: typing runs ONE Semantic Scholar search (Enter or a
    // 600ms pause), shown as a brief skeleton state; filters/sort stay local+instant.
    const searchNow = () => {
      clearTimeout(sidebar._sq);
      const q = searchEl.value.trim();
      if (sidebar.dataset.state === 'empty') sidebar.dataset.state = 'normal';
      if (q === (sidebar._lastQ || '')) return;
      sidebar._lastQ = q;
      const tok = sidebar._sqTok = (sidebar._sqTok || 0) + 1;
      if (!q) { sidebar.removeAttribute('data-searching'); apply(); return; }
      sidebar.setAttribute('data-searching', '1');
      if (countEl) countEl.textContent = 'Searching…';
      setTimeout(() => {
        if (sidebar._sqTok !== tok) return;
        sidebar.removeAttribute('data-searching');
        if (sidebar.getAttribute('data-first') === '1') sidebar.setAttribute('data-searched', '1');
        apply();
      }, 850);
    };
    searchEl.addEventListener('input', () => { clearTimeout(sidebar._sq); sidebar._sq = setTimeout(searchNow, 600); });
    searchEl.addEventListener('keydown', e => { if (e.key === 'Enter') searchNow(); });
    sidebar.querySelectorAll('.sb-first .fc-chips button').forEach(b => b.addEventListener('click', () => { searchEl.value = b.textContent; searchEl.focus(); searchNow(); }));

    // filter panel
    const panel = sidebar.querySelector('.sb-filter-panel');
    const closeBtn = sidebar.querySelector('.sb-filter-close');
    const applyBtn = sidebar.querySelector('.sb-filter-apply');
    const resetBtn = sidebar.querySelector('.sb-filter-reset');
    const applyCount = sidebar.querySelector('.sb-apply-count');
    const yearVal = sidebar.querySelector('.fl-year-val');
    const citesVal = sidebar.querySelector('.fl-cites-val');
    const updateApplyCount = () => { if (applyCount) applyCount.textContent = rowEls.filter(matches).length; };
    const headerDiv = searchEl.parentElement.parentElement;
    let panelOpen = false;
    let inlineContentH = 0;
    if (panel) {
      panel.style.transition = 'opacity .22s ease, transform .28s cubic-bezier(.22,.61,.36,1), max-height .28s cubic-bezier(.22,.61,.36,1)';
      panel.style.display = 'flex';
      panel.style.opacity = '0';
      panel.style.pointerEvents = 'none';
      if (mode === 'popover') {
        (fbtn ? fbtn.parentElement : sidebar).appendChild(panel);
        panel.style.position = 'absolute';
        panel.style.inset = 'auto';
        panel.style.top = 'calc(100% + 8px)';
        panel.style.left = '0';
        panel.style.width = '320px';
        panel.style.maxHeight = '440px';
        panel.style.border = '1px solid var(--border)';
        panel.style.borderRadius = '12px';
        panel.style.boxShadow = '0 14px 34px rgba(var(--sh),.16)';
        panel.style.overflow = 'hidden';
        panel.style.transformOrigin = 'top left';
        panel.style.transform = 'translateY(-6px) scale(.98)';
      } else if (mode === 'inline') {
        headerDiv.insertAdjacentElement('afterend', panel);
        panel.style.position = 'static';
        panel.style.inset = 'auto';
        panel.style.borderBottom = '1px solid var(--divider)';
        panel.style.overflow = 'hidden';
        inlineContentH = panel.scrollHeight;
        panel.style.maxHeight = '0px';
      } else {
        panel.style.inset = '0';
        panel.style.transform = 'translateY(100%)';
      }
    }
    const openPanel = () => {
      if (!panel) return;
      panelOpen = true;
      panel.style.pointerEvents = 'auto';
      updateApplyCount();
      requestAnimationFrame(() => {
        panel.style.opacity = '1';
        if (mode === 'slideover') panel.style.transform = 'translateY(0)';
        else if (mode === 'popover') panel.style.transform = 'translateY(0) scale(1)';
        else if (mode === 'inline') {
          const prev = panel.style.maxHeight;
          panel.style.maxHeight = 'none';
          const full = panel.scrollHeight;
          const avail = Math.max(220, sidebar.clientHeight - headerDiv.offsetHeight - 8);
          panel.style.maxHeight = prev;
          requestAnimationFrame(() => {
            panel.style.maxHeight = Math.min(full, avail) + 'px';
            panel.style.overflowY = full > avail ? 'auto' : 'hidden';
          });
        }
      });
    };
    const closePanel = () => {
      if (!panel) return;
      panelOpen = false;
      panel.style.pointerEvents = 'none';
      panel.style.opacity = '0';
      if (mode === 'slideover') panel.style.transform = 'translateY(100%)';
      else if (mode === 'popover') panel.style.transform = 'translateY(-6px) scale(.98)';
      else if (mode === 'inline') panel.style.maxHeight = '0px';
    };
    if (fbtn && panel) fbtn.addEventListener('click', e => { e.stopPropagation(); panelOpen ? closePanel() : openPanel(); });
    if (closeBtn) closeBtn.addEventListener('click', closePanel);
    if (applyBtn) applyBtn.addEventListener('click', () => { closePanel(); apply(); });
    document.addEventListener('click', e => { if (panelOpen && mode !== 'slideover' && panel && !panel.contains(e.target) && fbtn && !fbtn.contains(e.target)) closePanel(); });

    sidebar.querySelectorAll('.dr').forEach(dr => {
      const fill = dr.querySelector('.dr-fill');
      const hLo = dr.querySelector('[data-side=lo]'), hHi = dr.querySelector('[data-side=hi]');
      const key = dr.dataset.key, dmin = +dr.dataset.min, dmax = +dr.dataset.max, step = +dr.dataset.step || 1;
      const PAD = 8;
      let vLo = dmin, vHi = dmax;
      const span = () => Math.max(1, dr.clientWidth - PAD * 2);
      const clampStep = v => { v = Math.round((v - dmin) / step) * step + dmin; return Math.max(dmin, Math.min(dmax, v)); };
      const layout = () => {
        const loX = PAD + ((vLo - dmin) / (dmax - dmin)) * span();
        const hiX = PAD + ((vHi - dmin) / (dmax - dmin)) * span();
        hLo.style.left = loX + 'px'; hHi.style.left = hiX + 'px';
        fill.style.left = loX + 'px'; fill.style.width = Math.max(0, hiX - loX) + 'px';
        if (key === 'year') { F.yearLo = vLo; F.yearHi = vHi; if (yearVal) yearVal.textContent = vLo + '–' + vHi; }
        else { F.citesLo = vLo; F.citesHi = vHi; if (citesVal) citesVal.textContent = fmtK(vLo) + '–' + (vHi >= dmax ? fmtK(dmax) + '+' : fmtK(vHi)); }
      };
      dr._sync = layout;
      const valAt = clientX => clampStep(dmin + Math.max(0, Math.min(1, (clientX - dr.getBoundingClientRect().left - PAD) / span())) * (dmax - dmin));
      const drag = (side, e) => {
        e.preventDefault();
        const move = ev => {
          const cx = ev.touches ? ev.touches[0].clientX : ev.clientX;
          const v = valAt(cx);
          if (side === 'lo') vLo = Math.min(v, vHi); else vHi = Math.max(v, vLo);
          layout(); updateApplyCount(); apply();
        };
        const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); window.removeEventListener('pointercancel', up); };
        window.addEventListener('pointermove', move); window.addEventListener('pointerup', up); window.addEventListener('pointercancel', up);
      };
      hLo.addEventListener('pointerdown', e => { hLo.style.zIndex = 3; hHi.style.zIndex = 2; drag('lo', e); });
      hHi.addEventListener('pointerdown', e => { hHi.style.zIndex = 3; hLo.style.zIndex = 2; drag('hi', e); });
      dr.addEventListener('pointerdown', e => {
        if (e.target.classList.contains('dr-h')) return;
        const v = valAt(e.clientX);
        if (Math.abs(v - vLo) <= Math.abs(v - vHi)) vLo = Math.min(v, vHi); else vHi = Math.max(v, vLo);
        layout(); updateApplyCount(); apply();
      });
      dr._reset = () => { vLo = dmin; vHi = dmax; layout(); };
      dr._relayout = layout;
      layout();
    });

    const SYN = 'Combine terms with AND / OR / NOT, group with ( ), truncate with *, quote an exact phrase with " ".';
    sidebar.querySelectorAll('.ff-input').forEach(inp => {
      const lb = inp.parentElement && inp.parentElement.querySelector('label');
      if (!lb || lb.querySelector('.q-info')) return;
      lb.style.display = 'flex'; lb.style.alignItems = 'center'; lb.style.gap = '6px';
      const b = document.createElement('button');
      b.className = 'q-info'; b.type = 'button'; b.setAttribute('data-tip', SYN); b.setAttribute('aria-label', 'Query syntax');
      b.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"></circle><path d="M12 16v-5M12 8h.01"></path></svg>';
      lb.appendChild(b);
    });
    sidebar.querySelectorAll('.ff-input').forEach(inp => {
      const host = inp.parentElement;
      let err = host.querySelector('.ff-err');
      if (!err) {
        err = document.createElement('div');
        err.className = 'ff-err';
        err.setAttribute('data-show', '0');
        err.style.cssText = 'display:flex; align-items:center; gap:5px; font-size:11px; font-weight:500; color:var(--error); line-height:1.4;';
        err.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><circle cx="12" cy="12" r="9"></circle><path d="M12 8v4M12 16h.01"></path></svg><span></span>';
        host.appendChild(err);
      }
      const check = () => {
        const r = this.validateQuery(inp.value);
        inp.style.borderColor = r.ok ? '' : 'var(--error)';
        err.setAttribute('data-show', r.ok ? '0' : '1');
        err.style.display = r.ok ? 'none' : 'flex';
        if (!r.ok) err.querySelector('span').textContent = r.msg.charAt(0).toUpperCase() + r.msg.slice(1) + '.';
      };
      err.style.display = 'none';
      inp.addEventListener('input', () => { F[inp.dataset.key] = (inp.value || '').trim(); check(); updateApplyCount(); apply(); });
    });


    const doReset = () => {
      F.venue = F.title = F.abstract = F.author = '';
      F.yearLo = YMIN; F.yearHi = YMAX; F.citesLo = CMIN; F.citesHi = CMAX;
      sidebar.querySelectorAll('.ff-input').forEach(i => { i.value = ''; });
      sidebar.querySelectorAll('.dr').forEach(dr => { if (dr._reset) dr._reset(); });
      updateApplyCount(); apply();
    };
    if (resetBtn) resetBtn.addEventListener('click', doReset);

    const clearBtn = sidebar.querySelector('.sb-clear');
    if (clearBtn) clearBtn.addEventListener('click', () => { searchEl.value = ''; clearTimeout(sidebar._sq); sidebar._sqTok = (sidebar._sqTok || 0) + 1; sidebar.removeAttribute('data-searching'); sidebar._lastQ = ''; doReset(); sidebar.dataset.state = 'normal'; apply(); if (demo) paintDemo(); });
    const retryBtn = sidebar.querySelector('.sb-retry');
    if (retryBtn) retryBtn.addEventListener('click', () => { sidebar.dataset.state = 'normal'; apply(); if (demo) paintDemo(); });
    const demo = sidebar.closest('section') ? sidebar.closest('section').querySelector('.sb-demo') : null;
    const paintDemo = () => {
      if (!demo) return;
      demo.querySelectorAll('button').forEach(bt => {
        const on = (sidebar.dataset.state || 'normal') === bt.dataset.state;
        bt.style.background = on ? 'var(--chip-bg)' : 'var(--panel)';
        bt.style.color = on ? '#fff' : 'var(--fg2)';
        bt.style.borderColor = on ? 'var(--chip-bg)' : 'var(--border)';
      });
    };
    if (demo) {
      demo.querySelectorAll('button').forEach(bt => bt.addEventListener('click', () => { sidebar.dataset.state = bt.dataset.state; apply(); paintDemo(); }));
      paintDemo();
    }
    sort();
    apply();
  }
  setupAbstracts(root) {
    const mode = 'drill';
    const avMode = 'smart';
    // paper-row.js may still be loading: rows painted later must be wired too, and the
    // pass has to be idempotent so the re-run can't append a second drill panel.
    root.querySelectorAll('.sidebar').forEach(sb => { if (sb._absW) return; sb._absW = 1; this.wireAbstract(sb, mode, avMode); });
    if (!window.KLPaperRow && !root._absDefer) {
      root._absDefer = 1;
      document.addEventListener('kl-paper-row-ready', () => {
        root.querySelectorAll('.sidebar .pr').forEach(r => { r.style.paddingRight = '46px'; });
      }, { once: true });
    }
  }
  kfmt(n) {
    if (n < 1000) return String(n);
    if (n < 999500) { const v = n / 1000; return (v < 10 ? v.toFixed(1).replace(/\.0$/, '') : String(Math.round(v))) + 'k'; }
    const m = n / 1e6; return (m < 10 ? m.toFixed(1).replace(/\.0$/, '') : String(Math.round(m))) + 'M';
  }
  wireAbstract(sidebar, mode, avMode) {
    const rows = Array.from(sidebar.querySelectorAll('.pr'));
    const fmt = n => this.kfmt(n);
    const meta = r => r.dataset.year + '  ' + fmt(+r.dataset.cites) + ' cites';
    const accent = () => 'var(--muted2)';
    const dotColor = () => 'var(--muted2)';
    const venueFull = r => { const l = r.querySelector('.pr-venue'); return l ? l.textContent.trim() : (r.dataset.venue || ''); };
    const titleOf = r => r.querySelector('.pr-title').textContent;
    const authorsOf = r => { const a = r.querySelector('.pr-authors'); return a ? a.textContent : ''; };
    const fullAuthorMap = {"Haotian Cui":"Haotian Cui  Chloe Wang  Hassaan Maan  Kuan Pang  Fengning Luo  Nan Duan  Bo Wang","Daniil A. Boiko":"Daniil A. Boiko  R. MacKnight  Benjamin C Kline  Gabe Gomes","B. Romera-Paredes":"B. Romera-Paredes  M. Barekatain  Alexander Novikov  Matej Balog  M. P. Kumar  Emilien Dupont  Francisco J. R. Ruiz  J. Ellenberg  Pengming Wang  Omar Fawzi  Pushmeet Kohli  Alhussein Fawzi  Joshua A. Grochow  Andrea Lodi  Jean-Baptiste Mouret  Talia Ringer  Tao Yu","Chen Gao":"Chen Gao  Xiaochong Lan  Nian Li  Yuan Yuan  Jingtao Ding  Zhilun Zhou  Fengli Xu  Yong Li","Juraj Gottweis":"Juraj Gottweis  Wei-Hung Weng  A. Daryin  Tao Tu  Anil Palepu  Petar Sirkovic  Artiom Myaskovsky  Felix Weissenberger  Keran Rong  Ryutaro Tanno  Khaled Saab  D. Popovici  Jacob Blum  Fan Zhang  Katherine Chou  Avinatan Hassidim  Burak Gokturk  Amin Vahdat  Pushmeet Kohli  Yossi Matias  A. Carroll  Kavita Kulkarni  Nenad Tomašev  Yuan Guan  Vikram Dhillon  E. D. Vaishnav  Byron Lee  Tiago R. D. Costa  José R. Penadés  Gary Peltz  Yunhan Xu  Annalisa Pawlosky  A. Karthikesalingam  Vivek Natarajan","Yuhuai Wu":"Yuhuai Wu  Albert Qiaochu Jiang  Wenda Li  M. Rabe  Charles Staats  M. Jamnik  Christian Szegedy","Fei Liu":"Fei Liu  Xialiang Tong  Mingxuan Yuan  Xi Lin  Fu Luo  Zhenkun Wang  Zhichao Lu  Qingfu Zhang","Jun Shern Chan":"Jun Shern Chan  Neil Chowdhury  Oliver Jaffe  James Aung  Dane Sherburn  E. Mays  Giulio Starace  Kevin Liu  Leon Maksin  Tejal Patwardhan  Lilian Weng  Aleksander Mkadry","Qian Huang":"Qian Huang  Jian Vora  Percy Liang  J. Leskovec","Shuai Wang":"Shuai Wang  Harrisen Scells  B. Koopman  G. Zuccon","Yue Liu":"Yue Liu  Zhengwei Yang  Zhenyao Yu  Zitu Liu  Dahui Liu  Hailong Lin  Mingqing Li  Shuchang Ma  M. Avdeev  Siqi Shi","Paula Maddigan":"Paula Maddigan  Teo Sušnjak","Qusai Khraisha":"Qusai Khraisha  Sophie Put  Johanna Kappenberg  Azza Warraitch  K. Hadfield","Qingyan Guo":"Qingyan Guo  Rui Wang  Junliang Guo  Bei Li  Kaitao Song  Xu Tan  Guoqing Liu  Jiang Bian  Yujiu Yang  T. University  Microsoft Research","Jakub L'ala":"Jakub L'ala  Odhran O'Donoghue  Aleksandar Shtedritski  Sam Cox  S. Rodriques  Andrew D. White","Stanislas Polu":"Stanislas Polu  Jesse Michael Han  Kunhao Zheng  Mantas Baksys  Igor Babuschkin  I. Sutskever","Tao Song":"Tao Song  Man Luo  Xiaolong Zhang  Linjiang Chen  Yan Huang  Jiaqi Cao  Qing Zhu  Daobin Liu  Baicheng Zhang  Gang Zou  Guoqing Zhang  Fei Zhang  Weiwei Shang  Yao Fu  Jun Jiang  Yi Luo","Ziru Chen":"Ziru Chen  Shijie Chen  Yuting Ning  Qianheng Zhang  Boshi Wang  Botao Yu  Yifei Li  Zeyi Liao  Chen Wei  Zitong Lu  Vishal Dey  Mingyi Xue  Frazier N. Baker  Benjamin Burns  Daniel Adu-Ampratwum  Xuhui Huang  Xia Ning  Song Gao  Yu Su  Huan Sun","Zhiyu Yang":"Zhiyu Yang  Zihan Zhou  Shuo Wang  X. Cong  Xu Han  Yukun Yan  Zhenghao Liu  Zhixing Tan  Pengyuan Liu  Dong Yu  Zhiyuan Liu  Xiaodong Shi  Maosong Sun","Hjalmar Wijk":"Hjalmar Wijk  T. Lin  Joel Becker  Sami Jawhar  Neev Parikh  Thomas Broadley  Lawrence Chan  Michael Chen  Joshua Clymer  Jai Dhyani  Elena Ericheva  Katharyn Garcia  Brian Goodrich  Nikola Jurkovic  Megan Kinniment  Aron Lajko  Seraphina Nix  L. Sato  William Saunders  M. Taran  Ben West  Elizabeth Barnes","Michael Li":"Michael Li  Jianping Sun  Xianming Tan","Pingchuan Ma":"Pingchuan Ma  Rui Ding  Shuai Wang  Shi Han  Dongmei Zhang","P. Shojaee":"P. Shojaee  Kazem Meidani  Shashank Gupta  A. Farimani  Chandan K. Reddy","Alireza Ghafarollahi":"Alireza Ghafarollahi  Markus J. Buehler","Yiren Liu":"Yiren Liu  Si Chen  Haocong Cheng  Mengxia Yu  Xiao Ran  Andrew Mo  Yiliu Tang  Yun Huang","Minjun Zhu":"Minjun Zhu  Yixuan Weng  Linyi Yang  Yue Zhang","H. Lai":"H. Lai  Long Ge  Mingyao Sun  Bei Pan  Jiajie Huang  Liangying Hou  Qiuyu Yang  Jiayi Liu  Jianing Liu  Ziying Ye  Danni Xia  Weilong Zhao  Xiaoman Wang  Ming Liu  J. R. Talukdar  Jinhui Tian  Kehu Yang  J. Estill","Bogdan Georgiev":"Bogdan Georgiev  Javier G'omez-Serrano  Terence Tao  Adam Zsolt Wagner","Zhaoyu Li":"Zhaoyu Li  Jialiang Sun  Logan Murphy  Qidong Su  Zenan Li  Xian Zhang  Kaiyu Yang  Xujie Si","Chih-Hsuan Wei":"Chih-Hsuan Wei  Alexis Allot  Po-Ting Lai  Robert Leaman  Shubo Tian  Ling Luo  Qiao Jin  Zhizheng Wang  Qingyu Chen  Zhiyong Lu","Kehan Wu":"Kehan Wu  Yingce Xia  Pan Deng  Renhe Liu  Yuan Zhang  Han Guo  Yumeng Cui  Qizhi Pei  Lijun Wu  Shufang Xie  Si Chen  Xi Lu  Song Hu  Jinzhi Wu  C. Chan  Shawn Chen  Liangliang Zhou  Nenghai Yu  Enhong Chen  Haiguang Liu  Jinjiang Guo  Tao Qin  Tie-Yan Liu","Hengxing Cai":"Hengxing Cai  Xiaochen Cai  Junhan Chang  Sihang Li  Lin Yao  Changxin Wang  Zhifeng Gao  Hongshuai Wang  Yongge Li  Mujie Lin  Shuwen Yang  Jiankun Wang  Yuqi Yin  Yaqi Li  Linfeng Zhang  Guolin Ke","Ruochen Li":"Ruochen Li  Teerth Patel  Qingyun Wang  Qingyun Wang  Xinya Du","Hanchen Wang":"Hanchen Wang  Yichun He  Paula Coelho  M. Bucci  A. Nazir  Bo Chen  L. Trinh  Serena Zhang  Kexin Huang  Vineethkrishna Chandrasekar  D.C. Chung  Minsheng Hao  Ana Carolina Leote  Yongju Lee  Bo Li  Tianyu Liu  Jin Liu  Romain Lopez  Tawaun A. Lucas  Mingyu Derek Ma  Nikita Makarov  Lisa M. McGinnis  L. Peng  Stephen Ra  Gabriele Scalia  Avtar Singh  Liming Tao  Masatoshi Uehara  Chenyu Wang  Runmin Wei  Ryan Copping  O. Rozenblatt-Rosen  J. Leskovec  Aviv Regev","Akari Asai":"Akari Asai  Jacqueline He  Rulin Shao  Weijia Shi  Amanpreet Singh  J. Chang  Kyle Lo  Luca Soldaini  Sergey Feldman  Mike D'Arcy  David Wadden  Matt Latzke  Jenna Sparks  Jena D. Hwang  V. Kishore  Minyang Tian  Pan Ji  Shengyan Liu  Hao Tong  Bohao Wu  Yanyu Xiong  Luke S. Zettlemoyer  Graham Neubig  Dan Weld  Doug Downey  Wen-tau Yih  Pang Wei Koh  Hanna Hajishirzi","A. Ghareeb":"A. Ghareeb  Benjamin Chang  L. Mitchener  Angela Yiu  Caralyn J. Szostkiewicz  D. Shved  Gavin Gyimesi  Jon M. Laurent  S. Wright  Muhammed Razzak  Andrew D. White  S. Finnemann  Michaela M. Hinks  S. Rodriques","Aydin Ozcan":"Aydin Ozcan  François-Xavier Coudert  Sven M. J. Rogge  Greta Heydenrych  Dong Fan  Antonios P. Sarikas  Seda Keskin  G. Maurin  G. Froudakis  Stefan Wuttke  Ilknur Eruçar","Moritz Schaefer":"Moritz Schaefer  Peter Peneder  Daniel Malzl  S. Lombardo  Mihaela Peycheva  Jake Burton  Anna H. Hakobyan  Varun Sharma  Thomas Krausgruber  Celine Sin  J. Menche  E. Tomazou  Christoph Bock","Yingce Xia":"Yingce Xia  Peiran Jin  Shufang Xie  Liang He  Chuan Cao  Renqian Luo  Guoqing Liu  Yue Wang  Zequn Liu  Yuan Chen  Zekun Guo  Yeqi Bai  Pan Deng  Yaosen Min  Zi‐Ang Lu  Hongxia Hao  Han Yang  Jielan Li  Chang Liu  Jia Zhang  Jian-Bo Zhu  Ke-Ming Wu  Wei Zhang  Kaiyuan Gao  Qizhi Pei  Qian Wang  Xixian Liu  Yanting Li  Houtian Zhu  Yeqing Lu  Mingqian Ma  Zun Wang  Tian Xie  Krzysztof Maziarz  Marwin H. S. Segler  Zhao Yang  Zi-wei Chen  Yu Shi  Shuxin Zheng  Lijun Wu  Chen Hu  Peggy Dai  Tiemin Liu  Haiguang Liu  Tao Qin","Qiushi Sun":"Qiushi Sun  Zhoumianze Liu  Chang Ma  Zichen Ding  Fangzhi Xu  Zhangyue Yin  Haiteng Zhao  Zhenyu Wu  Kanzhi Cheng  Zhaoyang Liu  Jianing Wang  Qintong Li  Xiangru Tang  Tianbao Xie  Xiachong Feng  Xiang Li  Ben Kao  Wenhai Wang  Biqing Qi  Lingpeng Kong  Zhiyong Wu","Kexin Huang":"Kexin Huang  Serena Zhang  Hanchen Wang  Yuanhao Qu  Yingzhou Lu  Ryan Li  Yusuf H. Roohani  Lin Qiu  Shiyi Cao  Gavin Li  Junze Zhang  Di Yin  R. Wierenga  Deniz Kavi  Sherry Liu  Tianwei She  S. Marwaha  Jennefer N Carter  Xin Zhou  Matthew T. Wheeler  Jonathan A. Bernstein  Mengdi Wang  Peng He  Jingtian Zhou  Michael P. Snyder  Le Cong  Aviv Regev  J. Leskovec","Eser Aygün":"Eser Aygün  Anastasiya Belyaeva  Gheorghe Comanici  Marc Coram  Hao Cui  Jake Garrison  Renee Johnston  A. Kast  Cory Y. McLean  Peter C. Norgaard  Zahra Shamsi  David Smalling  James Thompson  S. Venugopalan  Brian P. Williams  Chujun He  Sarah Martinson  Martyna Plomecka  Lai Wei  Yuchen Zhou  Qian-Ze Zhu  Matthew Abraham  Erica Brand  Anna Bulanova  J. Cardille  Chris Co  Scott Ellsworth  Grace Joseph  M. Kane  R. Krueger  Johan Kartiwa  D. Liebling  Jan-Matthis Lueckmann  Paul Raccuglia  Xuefei Wang  Katherine Chou  J. Manyika  Y. Matias  J.C. Platt  Lizzie Dorfman  Shibl Mourad  Michael P. Brenner"};
    const fullAuthors = r => { const raw = authorsOf(r); for (const k in fullAuthorMap) { if (raw.indexOf(k) === 0) return fullAuthorMap[k]; } return raw.replace(/…$/, ''); };

    // drill mode: the row itself opens the abstract, so no per-row trigger button
    rows.forEach(row => {
      let trig = null;
      if (mode !== 'drill') {
        trig = document.createElement('button');
        trig.setAttribute('aria-label', 'Show abstract');
        trig.title = 'Show abstract';
        trig.innerHTML = '<svg class="ab-chev" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="transition:transform .2s ease;"><path d="M6 9l6 6 6-6"></path></svg>';
        trig.addEventListener('mouseenter', () => { trig.style.background = 'var(--ghost-hover)'; });
        trig.addEventListener('mouseleave', () => { trig.style.background = 'none'; });
        // in-flow control on the meta line: stable position, gap-spaced, no crowding
        trig.style.cssText = 'flex:none; background:none; border:none; padding:3px; margin:0; cursor:pointer; display:flex; align-items:center; border-radius:6px; transition:background .16s ease;';
        row.__trig = trig;
      }
      // Placement is DETERMINISTIC (§ Star / save control): the star stays exactly where
      // prCardInner put it, absolute against the row's right edge. It used to be relocated
      // here, which made its position depend on whether paper-row.js had painted the row
      // yet — two different layouts for the same page. Only the inline-mode chevron is
      // placed on the meta line, and only when that anchor actually exists.
      if (trig) { const meta = row.querySelector('.pr-meta'); if (meta) meta.appendChild(trig); }
      // the absolute star's lane — reserved only on rows that carry a star;
      // starless clones (.cs-row seed rows) keep their own box, title full width
      if (row.querySelector('.pr-star')) row.style.paddingRight = '46px';
    });

    if (mode === 'inline') {
      rows.forEach(row => {
        const ab = document.createElement('div');
        ab.style.cssText = 'overflow:hidden; max-height:0; opacity:0; transition:max-height .28s cubic-bezier(.22,.61,.36,1), opacity .22s ease;';
        const inner = document.createElement('div');
        inner.style.cssText = 'margin-top:10px; padding-top:10px; border-top:1px solid var(--divider); font-size:12.5px; line-height:1.55; color:var(--fg2);';
        const venueEl = venueLabel(row);
        row.__venueEl = venueEl;
        row.__avExpand = avMode === 'expand';
        const abbreviated = venueFull(row).trim().toLowerCase() !== (row.dataset.venue || '').trim().toLowerCase();
        if (avMode === 'always' || (avMode === 'smart' && abbreviated)) {
          const vrow = document.createElement('div');
          vrow.style.cssText = 'display:flex; align-items:center; gap:7px; margin-bottom:8px;';
          const vdot = document.createElement('span');
          vdot.style.cssText = 'width:6px; height:6px; border-radius:50%; flex:none; background:' + dotColor(row) + ';';
          const vname = document.createElement('span');
          vname.style.cssText = 'font-size:11px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:' + accent(row) + ';';
          vname.textContent = venueFull(row);
          vrow.appendChild(vdot); vrow.appendChild(vname);
          inner.appendChild(vrow);
        }
        if (row.dataset.url) {
          const link = document.createElement('a');
          link.href = row.dataset.url; link.target = '_blank'; link.rel = 'noopener';
          link.textContent = 'View on publisher →';
          link.style.cssText = 'display:inline-block; margin:0 0 10px; font-size:12px; font-weight:600; color:var(--accent); text-decoration:none;';
          link.addEventListener('click', e => e.stopPropagation());
          inner.appendChild(link);
        }
        const txt = document.createElement('div');
        txt.textContent = row.dataset.abstract || '';
        inner.appendChild(txt);
        ab.className = 'pr-abs';
        ab.appendChild(inner);
        row.appendChild(ab);
        row.__ab = ab;
      });
      const chev = r => r.__trig && r.__trig.querySelector('.ab-chev');
      const setVen = (r, full) => {
        if (!r.__avExpand || !r.__venueEl) return;
        const v = r.__venueEl;
        if (full) { v.style.whiteSpace = 'normal'; v.style.overflow = 'visible'; v.style.textOverflow = 'clip'; }
        else { v.style.whiteSpace = 'nowrap'; v.style.overflow = 'hidden'; v.style.textOverflow = 'ellipsis'; }
      };
      rows.forEach(row => {
        row.__onSelect = () => {
          const opening = !row.__open;
          rows.forEach(r => {
            if (r.__ab && r !== row) {
              r.__ab.style.maxHeight = '0'; r.__ab.style.opacity = '0'; r.__open = false;
              if (chev(r)) chev(r).style.transform = 'none';
              setVen(r, false);
            }
          });
          if (opening) { row.__ab.style.maxHeight = row.__ab.scrollHeight + 'px'; row.__ab.style.opacity = '1'; row.__open = true; }
          else { row.__ab.style.maxHeight = '0'; row.__ab.style.opacity = '0'; row.__open = false; }
          if (chev(row)) chev(row).style.transform = row.__open ? 'rotate(180deg)' : 'none';
          setVen(row, row.__open);
        };
      });
    } else if (mode === 'drill' || mode === 'panel') {
      const dp = document.createElement('div');
      dp.style.cssText = 'position:absolute; inset:0; z-index:40; background:var(--panel); display:flex; flex-direction:column; opacity:0; transform:translateX(14px); pointer-events:none; transition:opacity .2s ease, transform .26s cubic-bezier(.22,.61,.36,1);';
      dp.innerHTML =
        '<div style="padding:14px var(--pin); border-bottom:1px solid var(--divider); display:flex; align-items:center;">' +
          '<button class="ab-back" style="display:flex; align-items:center; gap:6px; background:none; border:none; cursor:pointer; font-family:inherit; font-size:13px; color:var(--fg2); padding:4px;">' +
            '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"></path></svg>Back</button>' +
        '</div>' +
        '<div style="flex:1; overflow-y:auto; padding:20px var(--pin-sc) 20px var(--pin); scrollbar-width:thin; scrollbar-gutter:stable; scrollbar-color:var(--scroll) transparent;">' +
          '<div style="display:flex; align-items:baseline; gap:7px; margin-bottom:11px;"><span class="ab-ic" data-ic="ven" style="display:inline-block; line-height:0; flex:none; position:relative; top:1px;"></span><span class="ab-venue" style="flex:1; min-width:0; font-size:11px; font-weight:600; letter-spacing:.05em; line-height:1.45; text-transform:uppercase;"></span></div>' +
          '<div class="ab-title" style="font-family:\'Newsreader\',serif; font-size:22px; line-height:1.28; font-weight:500; letter-spacing:-0.01em; color:var(--fg); text-wrap:pretty; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden;"></div>' +
          '<div style="display:flex; align-items:center; gap:13px; margin-top:10px; font-size:11.5px; color:var(--muted);"><span style="display:inline-flex; align-items:center; gap:5px;"><span class="ab-ic" data-ic="cite" style="display:inline-flex; flex:none;"></span><span class="ab-cites" style="font-variant-numeric:tabular-nums;"></span></span><span style="display:inline-flex; align-items:center; gap:5px;"><span class="ab-ic" data-ic="cal" style="display:inline-flex; flex:none;"></span><span class="ab-year" style="font-variant-numeric:tabular-nums;"></span></span></div>' +
          '<div style="display:flex; align-items:flex-start; gap:7px; margin-top:13px;"><span class="ab-ic" data-ic="user" style="display:inline-flex; flex:none; margin-top:4px; color:var(--muted);"></span><div class="ab-authors" style="flex:1; min-width:0; font-size:13px; color:var(--fg2); line-height:1.5; display:-webkit-box; -webkit-line-clamp:5; -webkit-box-orient:vertical; overflow:hidden;"></div></div>' +
          '<div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-top:16px;">' +
            '<a class="ab-link2" href="#" target="_blank" rel="noopener" style="display:inline-flex; align-items:center; gap:6px; padding:7px 12px; border:1px solid var(--border); border-radius:9px; background:var(--card); font-size:12.5px; font-weight:600; color:var(--fg); text-decoration:none;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4h6v6M20 4l-8 8M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"></path></svg>View on publisher</a>' +
            '<button class="ab-copy" style="display:inline-flex; align-items:center; gap:6px; padding:7px 12px; border:1px solid var(--border); border-radius:9px; background:var(--card); font-family:inherit; font-size:12.5px; font-weight:600; color:var(--fg2); cursor:pointer;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"></rect><path d="M5 15V5a2 2 0 0 1 2-2h10"></path></svg>Copy link</button>' +
            '<button class="ab-save" data-on="0" style="display:inline-flex; align-items:center; gap:6px; padding:7px 12px; border:1px solid var(--border); border-radius:9px; background:var(--card); font-family:inherit; font-size:12.5px; font-weight:600; color:var(--fg2); cursor:pointer;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l2.9 5.9 6.1.9-4.5 4.3 1.1 6-5.6-3-5.6 3 1.1-6L3 9.8l6.1-.9z"></path></svg><span>Save</span></button>' +
          '</div>' +
          '<div style="font-size:10.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--muted2); margin:22px 0 8px;">Abstract</div>' +
          '<div class="ab-abstract" style="font-size:13.5px; line-height:1.65; color:var(--abstract-fg); text-wrap:pretty;"></div>' +
        '</div>';
      sidebar.appendChild(dp);
      const q = c => dp.querySelector(c);
      const scroller = dp.querySelector('div[style*="overflow-y:auto"]');
      const open = () => { dp.style.pointerEvents = 'auto'; requestAnimationFrame(() => { dp.style.opacity = '1'; dp.style.transform = 'translateX(0)'; }); };
      const close = () => { dp.style.opacity = '0'; dp.style.transform = 'translateX(14px)'; dp.style.pointerEvents = 'none'; };
      q('.ab-back').addEventListener('click', close);
      rows.forEach(row => {
        row.__onSelect = () => {
          if (!dp._ics && window.KLPaperRow) { dp._ics = 1; dp.querySelectorAll('[data-ic]').forEach(el => { el.innerHTML = window.KLPaperRow.PR_ICONS[el.dataset.ic] || ''; }); }
          const v = q('.ab-venue'); v.textContent = venueFull(row); v.title = venueFull(row); v.style.color = accent(row);
          const vi = dp.querySelector('[data-ic="ven"]'); if (vi) vi.style.color = accent(row);
          q('.ab-cites').textContent = fmt(+row.dataset.cites) + ' cites';
          q('.ab-year').textContent = row.dataset.year;
          q('.ab-title').textContent = titleOf(row);
          q('.ab-authors').textContent = String(fullAuthors(row)).replace(/ \u00b7 /g, ', ');
          q('.ab-authors').title = fullAuthors(row);
          const lk2 = q('.ab-link2');
          if (lk2) { lk2.href = row.dataset.url || '#'; lk2.style.display = row.dataset.url ? 'inline-flex' : 'none'; }
          const cp = q('.ab-copy');
          if (cp && !cp._w) { cp._w = 1; cp.addEventListener('click', () => this.abCopyFlash(cp)); }
          const sv = q('.ab-save');
          if (sv) { const on = row.dataset.saved === '1'; sv.dataset.on = on ? '1' : '0'; sv.querySelector('span').textContent = on ? 'Saved' : 'Save'; sv.style.color = on ? 'var(--accent)' : 'var(--fg2)';
            if (!sv._w) { sv._w = 1; sv.addEventListener('click', () => { const cur = sv.dataset.on === '1'; sv.dataset.on = cur ? '0' : '1'; sv.querySelector('span').textContent = cur ? 'Save' : 'Saved'; sv.style.color = cur ? 'var(--fg2)' : 'var(--accent)'; if (this._drillRow) { this._drillRow.dataset.saved = cur ? '0' : '1'; const st = this._drillRow.querySelector('.pr-star'); if (st && st.__set) st.__set(!cur); } }); } }
          this._drillRow = row;
          q('.ab-abstract').textContent = row.dataset.abstract || '';
          if (scroller) scroller.scrollTop = 0;
          open();
        };
      });
    } else {
      const pop = document.createElement('div');
      pop.style.cssText = 'position:absolute; z-index:40; width:300px; background:var(--card); border:1px solid var(--border); border-radius:12px; box-shadow:0 14px 34px rgba(var(--sh),.16); padding:14px 16px; opacity:0; transform:translateY(-6px); pointer-events:none; transition:opacity .18s ease, transform .2s ease;';
      pop.innerHTML =
        '<div style="display:flex; align-items:center; gap:7px; margin-bottom:8px;"><span class="ab-dot" style="width:6px; height:6px; border-radius:50%; flex:none;"></span><span class="ab-venue" style="font-size:10.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase;"></span></div>' +
        '<div class="ab-title" style="font-family:\'Newsreader\',serif; font-size:15px; line-height:1.3; font-weight:500; color:var(--fg); margin-bottom:8px;"></div>' +
        '<a class="ab-link" href="#" target="_blank" rel="noopener" style="display:inline-block; margin:0 0 10px; font-size:12px; font-weight:600; color:var(--accent); text-decoration:none;">View on publisher →</a>' +
        '<div class="ab-abstract" style="font-size:12.5px; line-height:1.55; color:var(--fg2); max-height:180px; overflow-y:auto;"></div>';
      sidebar.appendChild(pop);
      const q = c => pop.querySelector(c);
      let openRow = null;
      const show = row => {
        q('.ab-dot').style.background = dotColor(row);
        const v = q('.ab-venue'); v.textContent = venueFull(row); v.style.color = accent(row);
        q('.ab-title').textContent = titleOf(row);
        q('.ab-abstract').textContent = row.dataset.abstract || '';
        const lk = q('.ab-link'); if (lk) { lk.href = row.dataset.url || '#'; lk.style.display = row.dataset.url ? 'inline-block' : 'none'; }
        pop.style.pointerEvents = 'auto';
        const rr = row.getBoundingClientRect(), sr = sidebar.getBoundingClientRect();
        const ph = pop.offsetHeight || 170;
        let top = rr.bottom - sr.top + 6;
        if (top + ph > sidebar.clientHeight - 8) top = Math.max(8, rr.top - sr.top - ph - 6);
        pop.style.top = top + 'px'; pop.style.left = '20px';
        requestAnimationFrame(() => { pop.style.opacity = '1'; pop.style.transform = 'translateY(0)'; });
        rows.forEach(r => { const c = r.__trig && r.__trig.querySelector('.ab-chev'); if (c) c.style.transform = 'none'; if (r !== row && r.__trig) r.__trig.style.opacity = '0'; });
        const cv = row.__trig && row.__trig.querySelector('.ab-chev'); if (cv) cv.style.transform = 'rotate(180deg)';
        if (row.__trig) row.__trig.style.opacity = '1';
        openRow = row;
      };
      const hide = () => {
        pop.style.opacity = '0'; pop.style.transform = 'translateY(-6px)'; pop.style.pointerEvents = 'none';
        if (openRow && openRow.__trig) { const c = openRow.__trig.querySelector('.ab-chev'); if (c) c.style.transform = 'none'; openRow.__trig.style.opacity = '0'; }
        openRow = null;
      };
      rows.forEach(row => { row.__onSelect = () => { openRow === row ? hide() : show(row); }; });
      document.addEventListener('click', e => { if (openRow && !pop.contains(e.target) && !e.target.closest('.pr')) hide(); });
    }
    rows.forEach(row => { if (row.__trig) row.__trig.addEventListener('click', e => { e.stopPropagation(); if (row.__onSelect) row.__onSelect(); }); });
  }
  componentDidUpdate(prev) {
    if (prev.theme !== this.props.theme) this.applyTheme(this.props.theme || 'light');
    if (prev.projAnchor !== this.props.projAnchor) this.paintProjAnchor();
    if (prev.layout !== this.props.layout) this.applyLayout();
    if (prev.pageState !== this.props.pageState || prev.layout !== this.props.layout) this.applyPage(this.props.pageState || 'Has results');
    if (prev.colorScheme !== this.props.colorScheme) this.applyColor(this.props.colorScheme || 'Warm paper');
    if (prev.logoStyle !== this.props.logoStyle) this.applyLogo(this.props.logoStyle || 'Soft ink tile');

  }
  applyTabs() {
    const root = this.rootRef.current;
    if (root) root.setAttribute('data-tabs', 'Quiet type');
    this.applySeg();
  }
  applySeg() {
    const root = this.rootRef.current;
    if (root) root.setAttribute('data-seg', 'Recessed shade');
  }
  applyAddBtn() {
    const root = this.rootRef.current;
    if (root) root.setAttribute('data-addbtn', 'Circle');
  }
  applyChip() {
    const root = this.rootRef.current;
    if (root) root.setAttribute('data-chip', 'Quiet mono');
  }
  setupRubber(root) {
    const sel = ['.sb-list', '.cfg-scroll', '.cfg-detail-body', '.pl-scroll', '.sb-filter-panel'];
    const opts = { damping: 0.16, thumbMinSize: 14, alwaysShowTracks: false, continuousScrolling: true,
      plugins: { overscroll: { effect: 'bounce', damping: 0.22, maxOverscroll: 45 } } };
    let used = false;
    const init = () => {
      if (!window.Scrollbar || !window.OverscrollPlugin) return;
      if (!used) { window.Scrollbar.use(window.OverscrollPlugin); used = true; }
      sel.forEach(s => root.querySelectorAll(s).forEach(el => {
        if (el._ssb) return;
        if (el.offsetWidth === 0 && el.offsetHeight === 0) return;
        el._ssb = true;
        window.Scrollbar.init(el, opts);
      }));
    };
    const poll = setInterval(() => { if (window.Scrollbar && window.OverscrollPlugin) { clearInterval(poll); [0, 200, 600, 1500].forEach(ms => setTimeout(init, ms)); } }, 80);
    new MutationObserver(() => init()).observe(root, { childList: true, subtree: true });
    this._ssbInit = init;
  }
  applyColor(v) {
    const root = this.rootRef.current;
    if (!root) return;
    root.setAttribute('data-color', v || 'Warm paper');
  }
  applyLogo(v) {
    const root = this.rootRef.current;
    if (!root) return;
    root.setAttribute('data-logo', v || 'Soft ink tile');
  }
  applyPage(v) {
    // 'Before first run' = first open: search panel shows an invite until the user
    // searches (search stays fully functional), the canvas invites a preset choice,
    // and the seed set is empty. Everything is dismissed by normal use.
    const root = this.rootRef.current;
    if (!root) return;
    const pre = v === 'Before first run';
    const wasPre = this._wasPre;
    this._wasPre = pre;
    if (pre && !wasPre) this._plFirstDone = false;
    root.setAttribute('data-state', pre ? 'pre' : 'ready');
    root.setAttribute('data-first', pre && !this._plFirstDone ? '1' : '0');
    root.querySelectorAll('.sidebar').forEach(sb => {
      sb.setAttribute('data-first', pre ? '1' : '0');
      if (pre) {
        const inp = sb.querySelector('.sb-search');
        sb.setAttribute('data-searched', inp && inp.value.trim() ? '1' : '0');
      } else sb.removeAttribute('data-searched');
      if (!sb.__fWrap && sb._applyList) {
        sb.__fWrap = 1;
        const orig = sb._applyList;
        sb._applyList = () => { orig(); this.sbFirstCount(sb); };
      }
      if (!sb.__fSearch) {
        sb.__fSearch = 1;
        const inp = sb.querySelector('.sb-search');
        if (inp) inp.addEventListener('input', () => {
          if (sb.getAttribute('data-first') !== '1') return;
          // the invite stays until a search actually completes (searchNow sets data-searched)
          if (!inp.value.trim()) { sb.setAttribute('data-searched', '0'); this.sbFirstCount(sb); }
        });
      }
      this.sbFirstCount(sb);
    });
    const seeds = this._seeds = this._seeds || [];
    if (pre) {
      if (!this._seedStash) this._seedStash = seeds.slice();
      seeds.slice().forEach(s => this.seedMark(s.title, false));
      this._seeds = [];
    } else if (this._seedStash) {
      const back = this._seedStash;
      this._seedStash = null;
      back.forEach(s => { if (!this._seeds.some(x => x.title === s.title)) this._seeds.push(s); });
      this._seeds.forEach(s => this.seedMark(s.title, true));
    }
    this.seedRender();
  }
  sbFirstCount(sb) {
    const c = sb.querySelector('.sb-count');
    if (!c) return;
    if (sb.getAttribute('data-first') === '1' && sb.getAttribute('data-searched') !== '1') c.textContent = 'No search yet';
  }
  applyScheme(v) {
    const root = this.rootRef.current;
    if (!root) return;
    root.setAttribute('data-scheme', v || 'Recessed canvas');
  }
  applyLayout() {
    const L = this.props.layout || 'List';
    this.applyRowStyle(L === 'Cards' ? 'cards' : 'list');
    this.applyPipe();
    if (this._seeds) this.seedRender();
  }
  applyRowStyle(rs) {
    const root = this.rootRef.current;
    if (!root) return;
    root.querySelectorAll('.sidebar').forEach(sb => { sb.style.display = (sb.dataset.style === rs) ? 'flex' : 'none'; });
  }
  applyTheme(t) {
    this._theme = t;
    const root = this.rootRef.current;
    if (!root) return;
    root.setAttribute('data-theme', t);
    const tg = root.querySelector('.pc-theme-toggle');
    if (tg) {
      tg.querySelector('.tt-label').textContent = t === 'dark' ? 'Light' : 'Dark';
      tg.querySelector('.tt-sun').style.display = t === 'dark' ? 'block' : 'none';
      tg.querySelector('.tt-moon').style.display = t === 'dark' ? 'none' : 'block';
    }
  }
  applyPipe() {
    const root = this.rootRef.current; if (!root) return;
    const pipe = root.querySelector('.cfg-pipe'); if (!pipe) return;
    pipe.setAttribute('data-layout', (this.props.layout || 'List') === 'Cards' ? 'Cards' : 'Seamless');
    pipe.setAttribute('data-marker', 'Icon');
    const aw = pipe.querySelector('.cfg-add-wrap'); if (aw) aw.setAttribute('data-add', 'Solid');
    pipe.setAttribute('data-number', 'Column');
  }
  cfIcons() {
    return {
      year: '<rect x="3" y="4" width="18" height="18" rx="2"></rect><path d="M3 10h18M8 2v4M16 2v4"></path>',
      citation: '<path d="M3 17l6-6 4 4 7-7"></path><path d="M17 8h4v4"></path>',
      keyword: '<path d="M4 9h16M4 15h16M10 3L8 21M16 3l-2 18"></path>',
      author: '<circle cx="12" cy="8" r="4"></circle><path d="M4 21a8 8 0 0 1 16 0"></path>',
      venue: '<path d="M4 5a2 2 0 0 1 2-2h12v16H6a2 2 0 0 0-2 2z"></path><path d="M18 3v16"></path>',
      llm: '<path d="M12 3l1.8 4.2L18 9l-4.2 1.8L12 15l-1.8-4.2L6 9l4.2-1.8z"></path>',
      similarity: '<circle cx="9" cy="12" r="6"></circle><circle cx="15" cy="12" r="6"></circle>'
    };
  }
  cfTools() {
    return '<span class="cf-tools"><button class="cf-tool" data-act="up" aria-label="Move up"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V6M6 12l6-6 6 6"></path></svg></button><button class="cf-tool" data-act="down" aria-label="Move down"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v13M6 12l6 6 6-6"></path></svg></button><button class="cf-tool" data-act="copy" aria-label="Copy filter"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"></rect><path d="M5 15V5a2 2 0 0 1 2-2h10"></path></svg></button><button class="cf-tool" data-act="delete" aria-label="Delete filter"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13"></path></svg></button></span>';
  }
  makeFilterRow(type, label, val) {
    const ic = this.cfIcons()[type] || this.cfIcons().keyword;
    const d = document.createElement('div');
    d.className = 'cf-filter'; d.dataset.type = type; d.dataset.label = label; d.dataset.value = val; d.dataset.applied = '0';
    d.innerHTML = '<span class="cf-rail"></span><span class="cf-node"><svg class="cf-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' + ic + '</svg><span class="cf-num">1</span></span><span class="cf-body"><span class="cf-type">' + label + '</span><span class="cf-val">' + val + '</span></span>' + this.cfTools();
    return d;
  }
  wireFilter(row) {
    const tools = row.querySelector('.cf-tools');
    if (tools && !tools.querySelector('[data-act="delete"]')) {
      const d = document.createElement('button');
      d.className = 'cf-tool'; d.dataset.act = 'delete'; d.setAttribute('aria-label', 'Delete filter');
      d.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13"></path></svg>';
      tools.appendChild(d);
    }
    row.addEventListener('click', () => {
      if (this._cfDragged) return;
      const pipe = row.closest('.cfg-pipe');
      const was = row.dataset.sel === '1';
      pipe.querySelectorAll('.cf-filter').forEach(f => { f.dataset.sel = '0'; });
      if (!was) { row.dataset.sel = '1'; this.openConfig(row); }
      else { if (row._draft) row._silentChecked = true; this.closeConfig(); }
    });
    row.querySelectorAll('.cf-tool').forEach(t => {
      t.addEventListener('click', e => {
        e.stopPropagation();
        const act = t.dataset.act;
        if (act === 'up') this.moveFilter(row, -1);
        else if (act === 'down') this.moveFilter(row, 1);
        else if (act === 'copy') this.copyFilter(row, t);
        else if (act === 'delete') this.deleteFilter(row);
      });
    });
  }
  // iOS-grade reorder: the row rides the pointer 1:1 from the first millimetre (mouse)
  // or after a short hold (touch, so the list still scrolls); neighbours slide aside with
  // transforms only, and the DOM commits ONCE on release via a FLIP settle.
  wireCfDrag(pipe) {
    if (pipe._cfDrag) return; pipe._cfDrag = 1;
    pipe.addEventListener('pointerdown', e => {
      if (e.button !== 0) return;
      const row = e.target.closest('.cf-filter');
      if (!row || e.target.closest('.cf-tools') || e.target.closest('button, input, textarea')) return;
      const touch = e.pointerType !== 'mouse';
      const sx = e.clientX, sy = e.clientY;
      let live = false, holdT = null, rows = null, rects = null, dIdx = -1, tIdx = -1, minDy = 0, maxDy = 0, H = 0;
      const rowsOf = () => [...pipe.querySelectorAll('.cf-filter')];
      const blk = ev => { if (live && ev.cancelable) ev.preventDefault(); };
      const lift = () => {
        rows = rowsOf(); dIdx = rows.indexOf(row); tIdx = dIdx;
        if (dIdx < 0) return;
        live = true;
        rects = rows.map(x => x.getBoundingClientRect());
        const gap = rows.length > 1 ? Math.max(0, rects[1].top - rects[0].bottom) : 8;
        H = rects[dIdx].height + gap;
        minDy = rects[0].top - rects[dIdx].top - 6;
        maxDy = rects[rects.length - 1].bottom - rects[dIdx].bottom + 6;
        row.style.position = 'relative'; row.style.zIndex = '30';
        row.style.boxShadow = '0 3px 9px rgba(var(--sh),.12), 0 16px 34px rgba(var(--sh),.16)';
        row.style.background = 'var(--card)';
        rows.forEach(x => { if (x !== row) x.style.transition = 'transform .22s cubic-bezier(.22,.61,.36,1)'; });
        document.body.style.cursor = 'grabbing'; document.body.style.userSelect = 'none';
        if (touch && navigator.vibrate) { try { navigator.vibrate(8); } catch (err) {} }
      };
      if (touch) holdT = setTimeout(lift, 240);
      const place = dy => {
        dy = Math.max(minDy, Math.min(maxDy, dy));
        row.style.transform = 'translateY(' + dy + 'px) scale(1.012)';
        const dc = rects[dIdx].top + rects[dIdx].height / 2 + dy;
        let t = 0;
        rows.forEach((x, i) => { if (i !== dIdx && rects[i].top + rects[i].height / 2 < dc) t++; });
        tIdx = t;
        rows.forEach((x, i) => {
          if (i === dIdx) return;
          let sh = 0;
          if (i >= t && i < dIdx) sh = H;
          else if (i <= t && i > dIdx) sh = -H;
          x.style.transform = sh ? 'translateY(' + sh + 'px)' : '';
        });
      };
      const move = mv => {
        if (!live) {
          if (touch) { if (Math.abs(mv.clientY - sy) > 9 || Math.abs(mv.clientX - sx) > 11) off(); return; }
          if (Math.abs(mv.clientY - sy) < 3 && Math.abs(mv.clientX - sx) < 5) return;
          lift();
          if (!live) return;
        }
        if (mv.cancelable) mv.preventDefault();
        place(mv.clientY - sy);
      };
      const clean = () => {
        rows.forEach(x => { x.style.transition = ''; x.style.transform = ''; });
        row.style.position = ''; row.style.zIndex = ''; row.style.boxShadow = ''; row.style.background = '';
        document.body.style.cursor = ''; document.body.style.userSelect = '';
      };
      const off = () => {
        clearTimeout(holdT);
        document.removeEventListener('pointermove', move);
        document.removeEventListener('pointerup', up);
        document.removeEventListener('pointercancel', cancel);
        document.removeEventListener('touchmove', blk);
      };
      const up = () => {
        off();
        if (!live) return;
        this._cfDragged = true; setTimeout(() => { this._cfDragged = false; }, 0);
        const before = new Map(rows.map(x => [x, x.getBoundingClientRect()]));
        clean();
        if (tIdx !== dIdx) {
          this.pushHistory(pipe);
          const others = rows.filter(x => x !== row);
          const nxt = others[tIdx];
          if (nxt) pipe.insertBefore(row, nxt);
          else pipe.querySelector('.cfg-add-wrap').insertAdjacentElement('beforebegin', row);
          this.rebuildPipe(pipe);
        }
        rows.forEach(x => {
          const b = before.get(x), a = x.getBoundingClientRect();
          const d = b.top - a.top;
          if (d && x.animate) x.animate([{ transform: 'translateY(' + d + 'px)' + (x === row ? ' scale(1.012)' : '') }, { transform: 'none' }], { duration: x === row ? 260 : 220, easing: 'cubic-bezier(.22,.61,.36,1)' });
        });
      };
      // a system gesture mid-drag aborts: transforms spring back, original order kept
      const cancel = () => {
        off();
        if (!live) return;
        live = false;
        row.style.transition = 'transform .22s cubic-bezier(.22,.61,.36,1)';
        row.style.transform = '';
        rows.forEach(x => { if (x !== row) x.style.transform = ''; });
        setTimeout(() => { row.style.transition = ''; clean(); }, 240);
      };
      document.addEventListener('pointermove', move);
      document.addEventListener('pointerup', up);
      document.addEventListener('pointercancel', cancel);
      if (touch) document.addEventListener('touchmove', blk, { passive: false });
    });
  }
  plDragWire(canvas) {
    if (canvas._plDrag) return; canvas._plDrag = 1;
    canvas.addEventListener('pointerdown', e => {
      if (e.button !== 0) return;
      if (e.target.closest('button, .pl-addmenu, .pb-add, .pl-tailadd, .pl-join-add')) return;
      const card = e.target.closest('.pb-card, .v6d-node');
      if (!card) return;
      const st = this.plFind(card.dataset.id);
      if (!st) return;
      if (st.type === 'source') {
        // the seed node never drags, but a touch tap must still land on lift — the
        // click event alone loses the first tap to iOS hover-emulation
        if (e.pointerType !== 'mouse') {
          const upSeed = () => {
            document.removeEventListener('pointerup', upSeed);
            this._plDragged = true; setTimeout(() => { this._plDragged = false; }, 150);
            this.plHideMenu();
            if (this._cfgRow) this.cancelConfig();
            this._plSel = card.dataset.id;
            this.plRender();
          };
          document.addEventListener('pointerup', upSeed);
        }
        return;
      }
      const root = this.rootRef.current;
      const sx = e.clientX, sy = e.clientY, touch = e.pointerType !== 'mouse';
      let live = false, ghost = null, ind = null, slots = null, active = null;
      const move = mv => {
        if (!live) {
          const th = touch ? 13 : 5;
          if (Math.abs(mv.clientX - sx) < th && Math.abs(mv.clientY - sy) < th) return;
          live = true;
          this.plHideMenu();
          const r = card.getBoundingClientRect();
          /* drag proxy: a compact carry chip, not a card clone (clones lose
             the flow-scoped tokens and collapse). Icon + name + code. */
          const nameEl = card.querySelector('.v6d-name, .pb-title'), codeEl = card.querySelector('.v6d-code, .pb-code');
          const esc = s => String(s || '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
          ghost = this.plH('<div style="position:fixed; z-index:90; pointer-events:none; display:flex; align-items:center; gap:9px; height:42px; width:max-content; max-width:300px; padding:0 15px 0 13px; background:var(--card); border:1px solid var(--border); border-radius:11px; box-shadow:0 2px 6px rgba(var(--sh),.10), 0 16px 36px rgba(var(--sh),.20); opacity:0; transform:translateY(3px) scale(.96); transition:opacity .14s ease, transform .14s cubic-bezier(.22,.61,.36,1);">'
            + '<span style="flex:none; display:flex; color:var(--fg2);">' + this.plIconSvg(st.type, 13) + '</span>'
            + '<span style="min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-family:var(--font-serif,\'Newsreader\',serif); font-size:15px; font-weight:500; color:var(--fg);">' + esc(nameEl ? nameEl.textContent : 'Step') + '</span>'
            + (codeEl ? '<span style="flex:none; font-size:11px; font-weight:600; letter-spacing:.04em; color:var(--muted);">' + esc(codeEl.textContent) + '</span>' : '')
            + '</div>');
          ghost._dx = 21; ghost._dy = touch ? 64 : 21;
          ghost.style.left = (mv.clientX - ghost._dx) + 'px'; ghost.style.top = (mv.clientY - ghost._dy) + 'px';
          requestAnimationFrame(() => { if (ghost) { ghost.style.opacity = '1'; ghost.style.transform = 'none'; } });
          root.appendChild(ghost);
          card.style.opacity = '.3';
          document.body.style.cursor = 'grabbing'; document.body.style.userSelect = 'none';
          slots = this.plSlots(card.dataset.id);
          ind = this.plH('<div style="position:absolute; height:3px; border-radius:2px; background:var(--fg2); box-shadow:0 0 0 3px color-mix(in oklab, var(--fg2) 22%, transparent); z-index:25; pointer-events:none; display:none;"></div>');
          canvas.appendChild(ind);
        }
        mv.preventDefault();
        ghost.style.left = (mv.clientX - ghost._dx) + 'px';
        ghost.style.top = (mv.clientY - ghost._dy) + 'px';
        let best = null, bd = Infinity;
        for (const s of slots) {
          let d;
          if (s.kind === 'fan') { const dx = mv.clientX - s.x, dy = Math.max(0, Math.abs(mv.clientY - s.y) - s.hh); d = dx * dx + dy * dy; }
          else { const dx = Math.max(0, Math.abs(mv.clientX - s.x) - s.hw), dy = mv.clientY - s.y; d = dx * dx + dy * dy; }
          if (d < bd) { bd = d; best = s; }
        }
        const rr = touch ? 160 : 120;
        active = (best && bd < rr * rr) ? best : null;
        if (active) {
          const cr = canvas.getBoundingClientRect();
          ind.style.display = 'block';
          if (active.kind === 'fan') {
            ind.style.width = '3px'; ind.style.height = (active.hh * 2) + 'px';
            ind.style.left = (active.x - 1.5 - cr.left + canvas.scrollLeft) + 'px';
            ind.style.top = (active.y - active.hh - cr.top + canvas.scrollTop) + 'px';
          } else {
            ind.style.height = '3px'; ind.style.width = (active.hw * 2) + 'px';
            ind.style.left = (active.x - active.hw - cr.left + canvas.scrollLeft) + 'px';
            ind.style.top = (active.y - 1.5 - cr.top + canvas.scrollTop) + 'px';
          }
        } else ind.style.display = 'none';
      };
      const off = () => {
        document.removeEventListener('pointermove', move);
        document.removeEventListener('pointerup', up);
        document.removeEventListener('pointercancel', cancel);
      };
      const up = () => {
        off();
        if (!live) {
          // touch: a finger tap can wobble past the browser's click slop, which would
          // swallow the click entirely - select on lift instead, first tap always lands
          if (touch) {
            this._plDragged = true; setTimeout(() => { this._plDragged = false; }, 150);
            this.plHideMenu();
            if (this._cfgRow) this.cancelConfig();
            this._plSel = card.dataset.id;
            this.plRender();
          }
          return;
        }
        this._plDragged = true; setTimeout(() => { this._plDragged = false; }, 0);
        if (ghost) ghost.remove();
        if (ind) ind.remove();
        card.style.opacity = '';
        document.body.style.cursor = ''; document.body.style.userSelect = '';
        if (active) this.plMoveTo(card.dataset.id, active);
      };
      // a system gesture mid-drag aborts: ghost and indicator gone, nothing moved
      const cancel = () => {
        off();
        if (!live) return;
        live = false;
        if (ghost) ghost.remove();
        if (ind) ind.remove();
        card.style.opacity = '';
        document.body.style.cursor = ''; document.body.style.userSelect = '';
      };
      document.addEventListener('pointermove', move);
      document.addEventListener('pointerup', up);
      document.addEventListener('pointercancel', cancel);
    });
  }
  plSlots(dragId) {
    const root = this.rootRef.current, out = [];
    const el = id => root.querySelector('.pl-flow :is(.pb-card,.v6d-node,.pl-par,.v6d-group-unit)[data-id="' + id + '"]');
    const seqSlots = (seq, startIdx) => {
      for (let i = startIdx; i <= seq.length; i++) {
        const prev = seq[i - 1], next = seq[i];
        if (prev && prev.id === dragId) continue;
        if (next && next.id === dragId) continue;
        const pr = prev ? el(prev.id) : null, nr = next ? el(next.id) : null;
        if (!pr && !nr) continue;
        const a = pr && pr.getBoundingClientRect(), b = nr && nr.getBoundingClientRect();
        if ((a && !a.width) || (b && !b.width)) continue;
        const ref = b || a;
        out.push({
          y: a && b ? (a.bottom + b.top) / 2 : a ? a.bottom + 14 : b.top - 14,
          x: ref.left + ref.width / 2,
          hw: Math.min(ref.width, 176) / 2,
          before: next ? next.id : null,
          after: !next && prev ? prev.id : null,
          // fallback anchors: pulling the drag out can dissolve a 1-branch-left parallel
          // group, killing the id the slot points at — the OTHER neighbour still resolves
          prevId: prev ? prev.id : null,
          trunkTail: !next && seq === this.pipe
        });
      }
    };
    seqSlots(this.pipe, 1);
    const walk = seq => seq.forEach(x => { if (x.type === 'parallel') x.branches.forEach(b => { seqSlots(b, 0); walk(b); }); });
    walk(this.pipe);
    this.pipe.forEach(x => {
      if (x.type === 'parallel' || x.type === 'source' || x.id === dragId) return;
      const r = el(x.id) && el(x.id).getBoundingClientRect();
      if (!r || !r.width) return;
      out.push({ kind: 'fan', anchor: x.id, x: r.right + 14, y: r.top + r.height / 2, hh: r.height / 2 });
      out.push({ kind: 'fan', side: 'l', anchor: x.id, x: r.left - 14, y: r.top + r.height / 2, hh: r.height / 2 });
    });
    return out.filter(s => this.plTry(this.plMoveFn(dragId, s), true));
  }
  plMoveFn(id, slot) {
    return p => {
      const loc = this.plLocate(id, p); if (!loc) throw 0;
      const st = loc.seq.splice(loc.i, 1)[0];
      const pruned = this.plPrune(p); p.length = 0; pruned.forEach(x => p.push(x));
      if (slot.kind === 'fan') {
        const t = this.plLocate(slot.anchor, p); if (!t) throw 0;
        if (t.par) { if (t.par.branches.length >= 2) throw 0; if (slot.side === 'l') t.par.branches.unshift([st]); else t.par.branches.push([st]); }
        else t.seq[t.i] = { id: 'p' + (++this._plSid), type: 'parallel', branches: slot.side === 'l' ? [[st], [t.el]] : [[t.el], [st]] };
      } else if (slot.before) {
        const t = this.plLocate(slot.before, p);
        if (t) t.seq.splice(t.i, 0, st);
        else if (slot.prevId != null) { const t2 = this.plLocate(slot.prevId, p); if (!t2) throw 0; t2.seq.splice(t2.i + 1, 0, st); }
        else throw 0;
      } else {
        const t = this.plLocate(slot.after, p);
        if (t) t.seq.splice(t.i + 1, 0, st);
        else if (slot.trunkTail) p.push(st);
        else throw 0;
      }
    };
  }
  plMoveTo(id, slot) {
    const ok = this.plTry(this.plMoveFn(id, slot));
    if (ok) { this._plSel = id; this.plRender(); }
    else this.pcToast('canvas', 'Can\u2019t move it there.');
  }
  moveFilter(row, dir) {
    const pipe = row.closest('.cfg-pipe');
    const rows = [...pipe.querySelectorAll('.cf-filter')];
    const i = rows.indexOf(row), j = i + dir;
    if (j < 0 || j >= rows.length) return;
    this.pushHistory(pipe);
    const before = new Map(rows.map(r => [r, r.getBoundingClientRect().top]));
    if (dir < 0) pipe.insertBefore(row, rows[j]);
    else rows[j].insertAdjacentElement('afterend', row);
    this.rebuildPipe(pipe);
    rows.forEach(r => {
      const d = before.get(r) - r.getBoundingClientRect().top;
      if (d && r.animate) r.animate([{ transform: 'translateY(' + d + 'px)' }, { transform: 'none' }], { duration: 300, easing: 'cubic-bezier(.22,.61,.36,1)' });
    });
  }
  copyFilter(row, btn) {
    const pipe = row.closest('.cfg-pipe');
    const menu = pipe.querySelector('.cfg-add-menu'); if (!menu) return;
    const type = row.dataset.type, label = row.dataset.label, val = row.dataset.value;
    const icon = this.cfIcons()[type] || this.cfIcons().keyword;
    let clip = menu.querySelector('.cfg-opt[data-clip="1"]');
    if (!clip) {
      const sep = document.createElement('div');
      sep.style.cssText = 'margin:6px 8px 3px; padding-top:7px; border-top:1px solid var(--divider); font-size:10.5px; font-weight:600; letter-spacing:.07em; text-transform:uppercase; color:var(--muted2);';
      sep.textContent = 'Clipboard';
      menu.appendChild(sep);
      clip = document.createElement('button');
      clip.className = 'cfg-opt'; clip.dataset.clip = '1';
      clip.style.cssText = 'width:100%; text-align:left; display:flex; align-items:center; gap:10px; padding:8px 10px; border:none; background:none; border-radius:8px; cursor:pointer; font-family:inherit; font-size:13px; color:var(--fg);';
      clip.addEventListener('mouseenter', () => { clip.style.background = 'var(--row-hover)'; });
      clip.addEventListener('mouseleave', () => { clip.style.background = 'none'; });
      this.wireOpt(clip, pipe);
      menu.appendChild(clip);
    }
    clip.dataset.type = type; clip.dataset.label = label; clip.dataset.val = val;
    clip.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="flex:none;">' + icon + '</svg><span style="min-width:0; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">' + label + '</span><span style="flex:none; font-size:10.5px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; color:var(--muted2);">copy</span>';
    this.flashCopied(btn);
  }
  // copy actions always answer on the button itself: label -> "Copied" + success tint
  abCopyFlash(cp) {
    const lk = cp.parentElement && cp.parentElement.querySelector('.ab-link2');
    try { const w = navigator.clipboard.writeText((lk && lk.href) || location.href); if (w && w.catch) w.catch(() => {}); } catch (err) {}
    clearTimeout(cp._t);
    const txt = cp.childNodes[cp.childNodes.length - 1];
    cp._lbl = cp._lbl || txt.textContent;
    txt.textContent = 'Copied';
    cp.style.color = 'var(--success)';
    cp.style.borderColor = 'color-mix(in oklab, var(--success) 45%, var(--border))';
    if (cp.animate) cp.animate([{ transform: 'scale(1)' }, { transform: 'scale(1.045)' }, { transform: 'scale(1)' }], { duration: 220, easing: 'ease-out' });
    cp._t = setTimeout(() => { txt.textContent = cp._lbl; cp.style.color = 'var(--fg2)'; cp.style.borderColor = ''; }, 1400);
  }
  flashCopied(btn) {
    const tools = btn && btn.closest('.cf-tools');
    if (!tools || tools._flash) return;
    tools.style.opacity = '1';
    const kids = [...tools.children];
    kids.forEach(k => { k.style.display = 'none'; });
    const tag = document.createElement('span');
    tag.style.cssText = 'display:flex; align-items:center; gap:4px; padding-right:2px; color:var(--accent); font-size:11px; font-weight:600; letter-spacing:.02em; white-space:nowrap;';
    tag.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12l5 5L20 6"></path></svg>Copied';
    tools.appendChild(tag);
    tools._flash = setTimeout(() => { tag.remove(); kids.forEach(k => { k.style.display = ''; }); tools.style.opacity = ''; tools._flash = null; }, 1300);
  }
  // one job per lens: a step never carries two filters doing the same screen —
  // one Year, one Citation, and per keyword/LLM type one Title + one Abstract pass
  cfgMenuSync(pipe, menu) {
    const rows = [...pipe.querySelectorAll('.cf-filter')];
    const used = { year: 0, citation: 0, keyword: {}, llm: {} };
    rows.forEach(r => {
      const t = r.dataset.type, v = r.dataset.value || '';
      if (t === 'year') used.year++;
      else if (t === 'citation') used.citation++;
      else if (t === 'keyword' || t === 'llm') { if (/^title/i.test(v)) used[t].title = 1; else if (/^abstract/i.test(v)) used[t].abstract = 1; }
    });
    menu.querySelectorAll('.cfg-opt').forEach(o => {
      const t = o.dataset.type; let why = '';
      if (t === 'year' && used.year) why = 'This step already has a Year filter. Edit that one instead.';
      else if (t === 'citation' && used.citation) why = 'This step already has a Citation filter. Edit that one instead.';
      else if ((t === 'keyword' || t === 'llm') && used[t].title && used[t].abstract) why = 'This step already screens both the title and the abstract with ' + (t === 'llm' ? 'LLM' : 'keyword') + ' filters.';
      o.dataset.dis = why ? '1' : '0';
      o.dataset.why = why;
      o.style.opacity = why ? '.4' : '';
      o.style.cursor = why ? 'default' : 'pointer';
      o.title = why;
    });
  }
  wireOpt(o, pipe) {
    const menu = pipe.querySelector('.cfg-add-menu');
    const addWrap = pipe.querySelector('.cfg-add-wrap');
    o.addEventListener('click', () => {
      if (o.dataset.dis === '1') { this.pcToast('canvas', o.dataset.why || 'Already in this pipeline.'); return; }
      this.pushHistory(pipe);
      const r = this.makeFilterRow(o.dataset.type, o.dataset.label, o.dataset.val);
      addWrap.insertAdjacentElement('beforebegin', r);
      this.wireFilter(r);
      this.rebuildPipe(pipe);
      this.expandIn(r);
      if (menu) menu.style.display = 'none';
      r.click();
    });
  }
  deleteFilter(row) {
    const pipe = row.closest('.cfg-pipe');
    this.pushHistory(pipe);
    const next = row.nextElementSibling;
    if (next && next.classList.contains('cf-conn')) next.remove();
    this.collapseOut(row, () => { row.remove(); this.rebuildPipe(pipe); });
  }
  pipeSnapshot(pipe) { return [...pipe.querySelectorAll('.cf-filter')].map(r => ({ t: r.dataset.type, l: r.dataset.label, v: r.dataset.value })); }
  pushHistory(pipe) { (this._hist = this._hist || []).push({ rows: this.pipeSnapshot(pipe), counts: this.plSteps().map(s => [s.id, s.filters || 0]) }); if (this._hist.length > 50) this._hist.shift(); this.updateUndo(); }
  updateUndo() { const b = this.rootRef.current && this.rootRef.current.querySelector('.cfg-undo'); if (b) b.disabled = !(this._hist && this._hist.length); }
  restorePipe(pipe, snap) {
    pipe.querySelectorAll('.cf-conn').forEach(c => c.remove());
    pipe.querySelectorAll('.cf-filter').forEach(r => r.remove());
    const addWrap = pipe.querySelector('.cfg-add-wrap');
    snap.forEach(o => { const r = this.makeFilterRow(o.t, o.l, o.v); addWrap.insertAdjacentElement('beforebegin', r); this.wireFilter(r); });
    this.rebuildPipe(pipe);
    [...pipe.querySelectorAll('.cf-filter')].forEach((r, i) => { if (r.animate) r.animate([{ opacity: 0, transform: 'translateY(5px)' }, { opacity: 1, transform: 'none' }], { duration: 220, delay: i * 26, easing: 'ease-out', fill: 'backwards' }); });
  }
  undo() {
    const pipe = this.rootRef.current && this.rootRef.current.querySelector('.cfg-pipe');
    if (!pipe || !this._hist || !this._hist.length) return;
    const h = this._hist.pop();
    this.restorePipe(pipe, h.rows || h);
    if (h.counts) {
      let ch = false;
      h.counts.forEach(en => { const s = this.plFind(en[0]); if (s && (s.filters || 0) !== en[1]) { s.filters = en[1]; ch = true; } });
      if (ch) this.plRender();
    }
    this.updateUndo();
  }
  rebuildPipe(pipe) {
    pipe.querySelectorAll('.cf-conn').forEach(c => c.remove());
    const rows = [...pipe.querySelectorAll('.cf-filter')];
    rows.forEach((r, idx) => {
      const c = document.createElement('div'); c.className = 'cf-conn';
      r.insertAdjacentElement('afterend', c);
      const num = idx + 1;
      const cv = r.querySelector('.cf-val'); if (cv && r.dataset.value != null && cv.textContent !== r.dataset.value) cv.textContent = r.dataset.value;
      const nn = r.querySelector('.cf-num'); if (nn) nn.textContent = num;
      let col = r.querySelector('.cf-idx-col');
      if (!col) { col = document.createElement('span'); col.className = 'cf-idx-col'; r.insertBefore(col, r.querySelector('.cf-node')); }
      col.textContent = num;
      const ty = r.querySelector('.cf-type');
      if (ty) ty.innerHTML = '<span class="cf-idx-lbl">' + num + '\u00a0\u00a0</span>' + this.filterName(r.dataset.type);
      const up = r.querySelector('[data-act=up]'), dn = r.querySelector('[data-act=down]');
      if (up) up.disabled = idx === 0;
      if (dn) dn.disabled = idx === rows.length - 1;

    });
    const count = pipe.parentElement.querySelector('.cfg-count');
    if (count) count.textContent = String(rows.length);
    if (this._fpPaint) this._fpPaint();
  }
  _vpPlace() {
    const root = this.rootRef.current; if (!root) return;
    const nar = root.getAttribute('data-vp') === 'narrow';
    const aside = root.querySelector('.pc-drawer'), host = root.querySelector('.cfg-shost'), main = root.querySelector('main');
    if (!aside || !host || !main) return;
    if (nar && aside.parentElement !== host) host.appendChild(aside);
    else if (!nar && aside.parentElement === host) main.insertBefore(aside, main.firstElementChild);
  }
  cfgConfirm(anchor, msg, actionLabel, onOk) {
    const root = this.rootRef.current; if (!root) return;
    const host = (anchor.closest && anchor.closest('.panel-config, .panel-canvas')) || root.querySelector('.panel-config');
    if (!host) return;
    root.querySelectorAll('.cfg-confirm').forEach(o => o.remove());
    const pop = document.createElement('div');
    pop.className = 'cfg-pop cfg-confirm';
    pop.style.width = '246px';
    pop.innerHTML = '<div style="font-size:12.5px; line-height:1.55; color:var(--fg);">' + msg + '</div>'
      + '<div style="display:flex; justify-content:flex-end; gap:8px; margin-top:12px;">'
      + '<button class="cc-cancel" style="border:none; background:none; padding:7px 10px; border-radius:8px; font-family:inherit; font-size:12.5px; font-weight:600; color:var(--fg2); cursor:pointer;">Cancel</button>'
      + '<button class="cc-ok" style="border:none; background:var(--primary); color:var(--primary-fg); padding:7px 14px; border-radius:9px; font-family:inherit; font-size:12.5px; font-weight:600; cursor:pointer;">' + actionLabel + '</button>'
      + '</div>';
    host.appendChild(pop);
    const ar = anchor.getBoundingClientRect(), hr = host.getBoundingClientRect();
    pop.style.left = Math.round(Math.max(10, Math.min(ar.left - hr.left, hr.width - 246 - 10))) + 'px';
    pop.style.top = Math.round(ar.bottom - hr.top + 8) + 'px';
    requestAnimationFrame(() => pop.setAttribute('data-open', '1'));
    const close = () => { pop.setAttribute('data-open', '0'); setTimeout(() => pop.remove(), 200); document.removeEventListener('click', outside); };
    const outside = e => { if (!pop.contains(e.target)) close(); };
    setTimeout(() => document.addEventListener('click', outside), 0);
    const cb = pop.querySelector('.cc-cancel'), ob = pop.querySelector('.cc-ok');
    cb.addEventListener('mouseenter', () => { cb.style.background = 'var(--row-hover)'; cb.style.color = 'var(--fg)'; });
    cb.addEventListener('mouseleave', () => { cb.style.background = 'none'; cb.style.color = 'var(--fg2)'; });
    ob.addEventListener('mouseenter', () => { ob.style.background = 'var(--primary-hover)'; });
    ob.addEventListener('mouseleave', () => { ob.style.background = 'var(--primary)'; });
    cb.addEventListener('click', e => { e.stopPropagation(); close(); });
    ob.addEventListener('click', e => { e.stopPropagation(); close(); onOk(); });
  }
  pcToast(where, html, withUndo) {
    const root = this.rootRef.current; if (!root) return;
    const host = root.querySelector(where === 'canvas' ? '.panel-canvas' : '.panel-config');
    if (!host) return;
    let t = host.querySelector('.pc-toast');
    if (!t) { t = document.createElement('div'); t.className = 'pc-toast'; host.appendChild(t); }
    t.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;"><circle cx="12" cy="12" r="9"></circle><path d="M8.5 12.3l2.3 2.3 4.7-4.7"></path></svg>'
      + '<span style="min-width:0;">' + html + '</span>'
      + (withUndo ? '<button class="pc-toast-undo">Undo</button>' : '');
    const ub = t.querySelector('.pc-toast-undo');
    if (ub) ub.addEventListener('click', () => { if (where === 'canvas') this.plUndo(); else this.undo(); t.setAttribute('data-show', '0'); });
    t.setAttribute('data-show', '0');
    requestAnimationFrame(() => t.setAttribute('data-show', '1'));
    clearTimeout(t._hideT);
    t._hideT = setTimeout(() => t.setAttribute('data-show', '0'), 3800);
  }
  plSetFilters(st, n) {
    if ((st.filters || 0) === n) return;
    st.filters = n;
    this.plRender();
  }
  plPresetFx() {
    const root = this.rootRef.current; if (!root) return;
    const flow = root.querySelector('.pl-flow'); if (!flow || !flow.animate) return;
    flow.animate([{ opacity: 0, transform: 'translateY(10px)' }, { opacity: 1, transform: 'none' }], { duration: 230, easing: 'cubic-bezier(.22,.61,.36,1)' });
    let i = 0;
    flow.querySelectorAll('.pb-card, .v6d-node').forEach(c => { if (c.animate) c.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 200, delay: 60 + (i++) * 28, easing: 'ease-out', fill: 'backwards' }); });
  }
  plPreset(kind) {
    const names = { scout: 'Scout', survey: 'Survey', dragnet: 'Dragnet' };
    const ok = this.plTry(p => {
      const src = p.find(x => x.type === 'source') || this.plNew('source');
      const par = bs => ({ id: 'p' + (++this._plSid), type: 'parallel', branches: bs });
      const mk = t => this.plNew(t, (t === 'fwd' || t === 'bwd' || t === 'db') ? { filters: 5 } : {});
      const f = mk('fwd'), b = mk('bwd');
      const seq = [src, par([[f], [b]]), mk('sem')];
      if (kind !== 'scout') seq.push(this.plNew('bwd', { filters: 5 }), mk('sem'));
      if (kind === 'dragnet') seq.push(par([[mk('rrk'), this.plNew('fwd', { filters: 5 })], [this.plNew('bwd', { filters: 5 })]]), mk('sem'));
      p.length = 0;
      seq.forEach(x => p.push(x));
    });
    if (!ok) return;
    this._plSel = null;
    this._plCentred = false;
    this.plRender();
    this.plPresetFx();
    this.pcToast('canvas', '<b>' + names[kind] + '</b> preset applied&nbsp;&nbsp;' + this.plCount() + ' steps', true);
  }
  setupConfigPanel(root) {
    const pipe = root.querySelector('.cfg-pipe'); if (!pipe) return;
    this.applyPipe();
    pipe.querySelectorAll('.cf-filter').forEach(r => this.wireFilter(r));
    pipe.querySelectorAll('.cf-filter').forEach(r => { r.dataset.applied = '1'; r._applied = this.cfgParse(r.dataset.type, r.dataset.value, r); });
    this._hist = [];
    this.rebuildPipe(pipe);
    this.wireCfDrag(pipe);
    const undoBtn = root.querySelector('.cfg-undo');
    if (undoBtn) undoBtn.addEventListener('click', () => this.undo());
    this.updateUndo();
    const tools = root.querySelector('.cfg-fp-tools');
    if (tools) {
      const cBtn = tools.querySelector('[data-fp="copy"]'), pBtn = tools.querySelector('[data-fp="paste"]'), aBtn = tools.querySelector('[data-fp="all"]');
      const FT = ['fwd', 'bwd', 'db', 'sem'];
      const codeOf = () => this._plSel ? this.plCode(this._plSel, this.plIndexMap(), this.plMeta()) : '';
      const paint = () => {
        const n = pipe.querySelectorAll('.cf-filter').length;
        if (cBtn) cBtn.disabled = !n;
        if (aBtn) aBtn.disabled = !n;
        if (pBtn) pBtn.disabled = !this._fclip;
      };
      this._fpPaint = paint;
      paint();
      cBtn.addEventListener('click', () => {
        const snap = this.pipeSnapshot(pipe);
        if (!snap.length) return;
        this._fclip = { snap: snap, code: codeOf() };
        paint();
        this.pcToast('config', 'Copied <b>' + snap.length + (snap.length === 1 ? ' filter' : ' filters') + '</b> from ' + this._fclip.code);
      });
      pBtn.addEventListener('click', e => {
        e.stopPropagation();
        if (!this._fclip) return;
        const n = this._fclip.snap.length;
        const cur = pipe.querySelectorAll('.cf-filter').length;
        const run = () => {
          this.pushHistory(pipe);
          this.restorePipe(pipe, this._fclip.snap);
          const st = this.plFind(this._plSel);
          if (st && st.type !== 'source') this.plSetFilters(st, this._fclip.snap.length);
          this.pcToast('config', 'Pasted <b>' + n + (n === 1 ? ' filter' : ' filters') + '</b> from ' + this._fclip.code + ' into ' + codeOf(), true);
        };
        if (!cur) { run(); return; }
        this.cfgConfirm(pBtn, 'Replace the <b>' + cur + (cur === 1 ? ' filter' : ' filters') + '</b> on ' + codeOf() + ' with the <b>' + n + '</b> copied from ' + this._fclip.code + '? Undo can bring them back.', 'Paste', run);
      });
      aBtn.addEventListener('click', e => {
        e.stopPropagation();
        const snap = this.pipeSnapshot(pipe);
        if (!snap.length) return;
        let total = 0;
        this.plSteps().forEach(s => { if (FT.indexOf(s.type) >= 0) total++; });
        this.cfgConfirm(aBtn, 'Give all <b>' + total + ' searchers</b> these <b>' + snap.length + (snap.length === 1 ? ' filter' : ' filters') + '</b>? Their current filters will be replaced. Undo can bring them back.', 'Apply to all', () => {
          this.pushHistory(pipe);
          let changed = false;
          this.plSteps().forEach(s => { if (FT.indexOf(s.type) >= 0 && (s.filters || 0) !== snap.length) { s.filters = snap.length; changed = true; } });
          if (changed) this.plRender();
          this.pcToast('config', 'These <b>' + snap.length + (snap.length === 1 ? ' filter' : ' filters') + '</b> now apply to all ' + total + ' searchers', true);
        });
      });
    }
    const addBtn = pipe.querySelector('.cfg-add');
    const menu = pipe.querySelector('.cfg-add-menu');
    if (addBtn && menu) {
      addBtn.addEventListener('click', e => { e.stopPropagation(); const open = menu.style.display === 'block'; menu.style.display = open ? 'none' : 'block'; if (!open) { this.cfgMenuSync(pipe, menu); this.popIn(menu); this.revealPop(menu); } });
      menu.addEventListener('click', e => e.stopPropagation());
      document.addEventListener('click', () => { menu.style.display = 'none'; });
      menu.querySelectorAll('.cfg-opt').forEach(o => this.wireOpt(o, pipe));
    }
  }
    yearFromValue(val) {
    val = val || '';
    const m = val.match(/\d{4}/g) || [];
    if (/up to/i.test(val)) return { from: '', to: m[0] || '' };
    if (/from/i.test(val)) return { from: m[0] || '', to: '' };
    return { from: m[0] || '', to: m[1] || '' };
  }
  yearSummary(from, to) {
    from = (from || '').trim(); to = (to || '').trim();
    if (from && to) return from + ' – ' + to;
    if (from) return 'From ' + from;
    if (to) return 'Up to ' + to;
    return 'Any year';
  }
  citationFromValue(val) {
    val = val || '';
    const beta = (val.match(/β\s*=\s*(\d+)/) || [])[1] || (val.match(/\d+/) || [])[0] || '';
    const cur = (val.match(/≥\s*(\d+)/) || [])[1] || '';
    return { beta, cur };
  }
  citationSummary(beta, cur) {
    beta = (beta || '').trim();
    if (!beta) return 'No threshold';
    return beta + ' cites / year';
  }
  kwCount(q) {
    const s = (q || '').trim();
    if (!s) return 0;
    return s.replace(/[()]/g, ' ').split(/\s+(?:AND|OR|NOT)\s+|,/i).map(x => x.trim()).filter(Boolean).length || 1;
  }
  llmModels() {
    return [
      { g: 'Google Gemini', items: [['Gemini 3.1 Flash-Lite', '$0.25/$1.5'], ['Gemini 3.5 Flash', '$1.5/$9'], ['Gemini 3.1 Pro (preview)', '$2/$12'], ['Gemini 2.5 Flash', '$0.3/$2.5']] },
      { g: 'OpenAI', items: [['GPT-5.6 Sol', '$5/$30'], ['GPT-5.6 Terra', '$2.5/$15'], ['GPT-5.6 Luna', '$1/$6'], ['GPT-5.6 Chat (latest)', '$5/$30'], ['GPT-5.5', '$5/$30'], ['GPT-5.5 Pro', '$30/$180'], ['GPT-5.4', '$2.5/$15'], ['GPT-5.4 mini', '$0.75/$4.5'], ['GPT-5.4 nano', '$0.2/$1.25'], ['GPT-5.4 Pro', '$30/$180'], ['GPT-5.3 Codex', '$1.75/$14']] }
    ];
  }
  llmTree(list) {
    if (!list) return [];
    return [...list.children].filter(c => c.classList.contains('fc-item')).map(it => {
      const g = it.querySelector(':scope > .fc-group > .fc-glist');
      if (g) return { kind: 'g', op: it.dataset.op || 'AND', children: this.llmTree(g) };
      const t = it.querySelector(':scope > .fc-qbody > .fc-ta');
      return { kind: 'q', op: it.dataset.op || 'AND', text: t ? t.value : '' };
    });
  }
  llmCount(tree) { return (tree || []).reduce((n, x) => n + (x.kind === 'g' ? this.llmCount(x.children) : 1), 0); }
  llmRule(tree) {
    const ops = []; let grouped = false;
    const walk = list => (list || []).forEach((x, i) => { if (i > 0) ops.push(x.op || 'AND'); if (x.kind === 'g') { grouped = true; walk(x.children); } });
    walk(tree);
    if (grouped) return 'grouped logic';
    if (!ops.length) return '';
    if (ops.every(o => o === 'AND')) return 'match all';
    if (ops.every(o => o === 'OR')) return 'match any';
    return 'mixed AND / OR';
  }
  cfgDefault(type) {
    if (type === 'year') return { from: '2022', to: '' };
    if (type === 'citation') return { beta: '', cur: '' };
    if (type === 'keyword') return { target: 'Abstract', query: '' };
    if (type === 'llm') return { target: 'Title', model: 'Default from Settings (gemini-3.1-flash-lite)', effort: 'Low', tree: [{ kind: 'q', op: 'AND', text: '' }] };
    if (type === 'venue') return { query: '' };
    if (type === 'similarity') return { min: '', missing: 'Pass' };
    return {};
  }
  cfgParse(type, val, row) {
    val = val || '';
    if (type === 'year') { const y = this.yearFromValue(val); return { from: y.from, to: y.to }; }
    if (type === 'citation') { const c = this.citationFromValue(val); return { beta: c.beta, cur: c.cur || (row && row.dataset.cur) || '' }; }
    if (type === 'keyword') {
      const m = val.match(/^(Abstract|Title)[\s\u00a0]+(.*)$/i);
      const tail = (m ? m[2] : val).trim();
      const stored = row && row.dataset.query;
      const q = stored != null ? stored : (/^(any|\d+\s+keywords?)$/i.test(tail) ? '' : tail);
      return { target: m && /title/i.test(m[1]) ? 'Title' : 'Abstract', query: q };
    }
    if (type === 'llm') { const d = this.cfgDefault('llm'); const m = val.match(/^(Title|Abstract|Venue)[\s\u00a0]+(\d+)/i); if (m) { d.target = m[1][0].toUpperCase() + m[1].slice(1).toLowerCase(); const op = /match any/i.test(val) ? 'OR' : 'AND'; d.tree = Array.from({ length: Math.max(1, parseInt(m[2], 10)) }, () => ({ kind: 'q', op, text: '' })); } return d; }
    if (type === 'venue') {
      const stored = row && row.dataset.query;
      const t = (val || '').trim();
      return { query: stored != null ? stored : (/^(any venue|\d+\s+venues?)$/i.test(t) ? '' : t) };
    }
    if (type === 'similarity') {
      const m = (val || '').match(/≥\s*([\d.]+)/);
      return { min: (row && row.dataset.min) || (m ? m[1] : ''), missing: /reject/i.test((row && row.dataset.missing) || val || '') ? 'Reject' : 'Pass' };
    }
    return this.cfgDefault(type);
  }
  cfgSummary(type, m) {
    if (type === 'year') return this.yearSummary(m.from, m.to);
    if (type === 'citation') return this.citationSummary(m.beta, m.cur);
    if (type === 'keyword') { const n = this.kwCount(m.query); return m.target + '  ' + (n ? n + (n === 1 ? ' keyword' : ' keywords') : 'any'); }
    if (type === 'llm') { const n = this.llmCount(m.tree) || 1; return m.target + '  ' + n + (n === 1 ? ' query' : ' queries'); }
    if (type === 'venue') { const n = this.kwCount(m.query); return n ? n + (n === 1 ? ' venue' : ' venues') : 'Any venue'; }
    if (type === 'similarity') { const t = (m.min || '').trim(); return (t ? '≥ ' + t : 'No threshold') + '  unscored ' + (m.missing === 'Reject' ? 'reject' : 'pass'); }
    return '';
  }
  cfgRead(type, w) {
    const val = q => { const el = w.querySelector(q); return el ? el.value.trim() : ''; };
    if (type === 'year') return { from: val('.fc-from'), to: val('.fc-to') };
    if (type === 'citation') return { beta: val('.fc-beta'), cur: val('.fc-cur') };
    if (type === 'keyword') { const on = w.querySelector('.fc-seg button[data-on="1"]'); return { target: on ? on.dataset.t : 'Abstract', query: (w.querySelector('.fc-ta') || {}).value || '' }; }
    if (type === 'llm') { const on = w.querySelector('.fc-seg button[data-on="1"]'); const md = w.querySelector('.fc-dd-model .fc-dd-btn span'), ef = w.querySelector('.fc-dd-effort .fc-dd-btn span'); return { target: on ? on.dataset.t : 'Title', model: md ? md.textContent : 'Default from Settings (gemini-3.1-flash-lite)', effort: ef ? ef.textContent : 'Low', tree: this.llmTree(w.querySelector('.fc-qs')) }; }
    if (type === 'venue') return { query: (w.querySelector('.fc-ta') || {}).value || '' };
    if (type === 'similarity') { const on = w.querySelector('.fc-seg button[data-on="1"]'); return { min: ((w.querySelector('.fc-min') || {}).value || '').trim(), missing: on ? on.dataset.t : 'Pass' }; }
    return {};
  }
  afterOpen(row) {
    this.showFooter(row);
    if (row._draft && row._silentChecked) {
      const n = this.validateEditor(row);
      if (n) { const st = this.getFooter().querySelector('.cfg-foot-status'); st.dataset.state = 'err'; st.textContent = n + (n === 1 ? ' issue to fix' : ' issues to fix'); }
    }
  }
  cancelConfig() { const row = this._cfgRow; if (!row) return; row._draft = null; row._silentChecked = false; row.dataset.sel = '0'; this.closeConfig(); }
  validateQuery(v) {
    const t = (v || '').trim();
    if (!t) return { ok: true, msg: '' };
    if ((t.match(/"/g) || []).length % 2) return { ok: false, msg: 'unclosed double quote' };
    let d = 0;
    for (const c of t) { if (c === '(') d++; else if (c === ')') { d--; if (d < 0) return { ok: false, msg: 'unbalanced ")"' }; } }
    if (d !== 0) return { ok: false, msg: 'unbalanced parentheses' };
    if (/(^|\s)(AND|OR|NOT)(\s*$)/i.test(t)) return { ok: false, msg: 'expression ends with an operator' };
    if (/(\bAND\b|\bOR\b)\s*(\bAND\b|\bOR\b)/i.test(t)) return { ok: false, msg: 'two operators in a row' };
    return { ok: true, msg: '' };
  }
  buildLLM(wrap, row) {
    const CE = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
    const m = row._draft ? row._draft : (row._applied ? row._applied : this.cfgDefault('llm'));
    const stage = () => { row._draft = this.cfgRead('llm', wrap); };
    const f1 = CE('div', 'fc-field', '<span class="fc-t">Screen on</span>');
    const seg = CE('div', 'fc-seg'); seg.dataset.variant = 'Segmented';
    ['Title', 'Abstract', 'Venue'].forEach(t => { const b = CE('button', null, t); b.dataset.t = t; b.dataset.on = t === m.target ? '1' : '0'; seg.appendChild(b); });
    f1.appendChild(seg); wrap.appendChild(f1);
    seg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => { [...seg.children].forEach(x => x.dataset.on = '0'); b.dataset.on = '1'; stage(); }));
    const f2 = CE('div', 'fc-field', '<span class="fc-t">Model</span>');
    const dd = CE('div', 'fc-dd fc-dd-model'), ddBtn = CE('button', 'fc-dd-btn'), ddMenu = CE('div', 'fc-dd-menu');
    ddBtn.type = 'button';
    let model = m.model;
    const setBtn = () => { ddBtn.innerHTML = '<span>' + model + '</span><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"></path></svg>'; };
    setBtn();
    const paintOpts = () => ddMenu.querySelectorAll('.fc-dd-opt').forEach(o => o.dataset.on = o.dataset.val === model ? '1' : '0');
    const addOpt = (nm, sub, val) => { const o = CE('button', 'fc-dd-opt fc-dd-opt2', '<span class="fc-dd-txt"><span class="fc-dd-nm">' + nm + '</span>' + (sub ? '<span class="fc-dd-sub">' + sub + '</span>' : '') + '</span>'); o.type = 'button'; o.dataset.val = val; o.addEventListener('click', e => { e.stopPropagation(); model = val; setBtn(); ddMenu.style.display = 'none'; paintOpts(); stage(); }); ddMenu.appendChild(o); };
    addOpt('Default', 'gemini-3.1-flash-lite', 'Default from Settings (gemini-3.1-flash-lite)');
    this.llmModels().forEach(grp => { ddMenu.appendChild(CE('div', 'fc-dd-grp', grp.g)); grp.items.forEach(it => addOpt(it[0], it[1], it[0])); });
    ddBtn.addEventListener('click', e => { e.stopPropagation(); const open = ddMenu.style.display === 'block'; ddMenu.style.display = open ? 'none' : 'block'; if (!open) { paintOpts(); const closer = ev => { if (dd.contains(ev.target)) return; ddMenu.style.display = 'none'; document.removeEventListener('click', closer); }; setTimeout(() => document.addEventListener('click', closer), 0); } });
    dd.appendChild(ddBtn); dd.appendChild(ddMenu); f2.appendChild(dd); wrap.appendChild(f2);
    const f3 = CE('div', 'fc-field', '<span class="fc-t">Reasoning effort</span>');
    const edd = CE('div', 'fc-dd fc-dd-effort'), eBtn = CE('button', 'fc-dd-btn'), eMenu = CE('div', 'fc-dd-menu');
    eBtn.type = 'button';
    let effort = m.effort;
    const eSet = () => { eBtn.innerHTML = '<span>' + effort + '</span><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"></path></svg>'; };
    eSet();
    const ePaint = () => eMenu.querySelectorAll('.fc-dd-opt').forEach(o => o.dataset.on = o.dataset.val === effort ? '1' : '0');
    ['Low', 'Medium', 'High'].forEach(v => { const o = CE('button', 'fc-dd-opt', v); o.type = 'button'; o.dataset.val = v; o.addEventListener('click', e => { e.stopPropagation(); effort = v; eSet(); eMenu.style.display = 'none'; ePaint(); stage(); }); eMenu.appendChild(o); });
    eBtn.addEventListener('click', e => { e.stopPropagation(); const open = eMenu.style.display === 'block'; eMenu.style.display = open ? 'none' : 'block'; if (!open) { ePaint(); const closer = ev => { if (edd.contains(ev.target)) return; eMenu.style.display = 'none'; document.removeEventListener('click', closer); }; setTimeout(() => document.addEventListener('click', closer), 0); } });
    edd.appendChild(eBtn); edd.appendChild(eMenu); f3.appendChild(edd); wrap.appendChild(f3);
    const f4 = CE('div', 'fc-field', '<span class="fc-t">Queries</span><span class="fc-d">Each query is a natural-language instruction. Pick how it joins the ones above; group queries to bracket them.</span>');
    const qs = CE('div', 'fc-qs'); f4.appendChild(qs);
    const foot = CE('div', 'fc-addrow');
    const addQBtn = CE('button', 'fc-addq', '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"></path></svg>Add query'), addGBtn = CE('button', 'fc-addq', '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"></path></svg>Add group');
    addQBtn.type = 'button'; addGBtn.type = 'button';
    foot.appendChild(addQBtn); foot.appendChild(addGBtn);
    f4.appendChild(foot);
    const QEX = ['e.g. The paper proves a new bound on prime gaps', 'e.g. The work formalizes its results in Lean or Coq', 'e.g. The paper introduces a new sieve method', 'e.g. The paper is a survey or expository article'];
    const cap = o => o === 'OR' ? 'Or' : 'And';
    const items = list => [...list.children].filter(c => c.classList.contains('fc-item'));
    const lastOp = list => { const it = items(list).pop(); return it && items(list).length > 1 ? (it.dataset.op || 'AND') : 'AND'; };
    const paintGutter = (it, mode) => {
      const old = it.querySelector(':scope > .fc-gut'); if (old) old.remove();
      if (mode === 'none') return;
      const g = CE('div', 'fc-gut');
      it.insertBefore(g, it.firstChild);
      if (mode === 'spacer') return;
      const w = CE('div', 'fc-opw'), b = CE('button', 'fc-op', '<span>' + cap(it.dataset.op) + '</span><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"></path></svg>'), menu = CE('div', 'fc-opm');
      b.type = 'button'; b.setAttribute('aria-label', 'Join with the query above');
      ['AND', 'OR'].forEach(op => {
        const o = CE('button', null, cap(op)); o.type = 'button'; o.dataset.op = op;
        o.addEventListener('click', e => { e.stopPropagation(); it.dataset.op = op; menu.dataset.open = '0'; sync(); stage(); });
        menu.appendChild(o);
      });
      b.addEventListener('click', e => {
        e.stopPropagation();
        const open = menu.dataset.open === '1';
        qs.querySelectorAll('.fc-opm').forEach(mm => mm.dataset.open = '0');
        menu.dataset.open = open ? '0' : '1';
        if (!open) {
          menu.querySelectorAll('button').forEach(o => o.dataset.on = o.dataset.op === (it.dataset.op || 'AND') ? '1' : '0');
          const closer = ev => { if (w.contains(ev.target)) return; menu.dataset.open = '0'; document.removeEventListener('click', closer); };
          setTimeout(() => document.addEventListener('click', closer), 0);
        }
      });
      w.appendChild(b); w.appendChild(menu); g.appendChild(w);
    };
    const sync = () => {
      [qs, ...qs.querySelectorAll('.fc-glist')].forEach(list => {
        const rows = items(list);
        rows.forEach((it, i) => paintGutter(it, i === 0 ? (rows.length < 2 ? 'none' : 'spacer') : 'op'));
      });
      const tas = [...qs.querySelectorAll('.fc-ta')];
      tas.forEach((t, i) => { t.placeholder = QEX[Math.min(i, QEX.length - 1)]; if (t._fit) t._fit(); });
      const root = items(qs);
      const solo = root.length === 1 && !root[0].querySelector('.fc-group');
      root.forEach(it => { const rm = it.querySelector(':scope > .fc-qbody > .fc-q-rm'); if (rm) rm.style.display = solo ? 'none' : 'flex'; });
    };
    const mkQuery = (text, op) => {
      const it = CE('div', 'fc-item'); it.dataset.op = op === 'OR' ? 'OR' : 'AND';
      const q = CE('div', 'fc-qbody'), t = CE('textarea', 'fc-ta');
      t.rows = 2; if (text) t.value = text;
      t._fit = () => { t.style.height = 'auto'; t.style.height = Math.max(44, t.scrollHeight) + 'px'; };
      t.addEventListener('input', () => { t._fit(); this.fieldOk(t); stage(); });
      const rm = CE('button', 'fc-q-rm', '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"></path></svg>'); rm.type = 'button'; rm.setAttribute('aria-label', 'Remove query');
      rm.addEventListener('click', () => { const list = it.parentElement; it.remove(); if (list !== qs && !items(list).length) { const host = list.closest('.fc-item'); if (host) host.remove(); } sync(); stage(); });
      q.appendChild(t); q.appendChild(rm); it.appendChild(q);
      return it;
    };
    const mkGroup = (nodes, op) => {
      const it = CE('div', 'fc-item'); it.dataset.op = op === 'OR' ? 'OR' : 'AND';
      const g = CE('div', 'fc-group'), head = CE('div', 'fc-ghead', '<span class="fc-gtag">Group</span>');
      const rm = CE('button', 'fc-q-rm', '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"></path></svg>'); rm.type = 'button'; rm.setAttribute('aria-label', 'Remove group');
      rm.addEventListener('click', () => { it.remove(); sync(); stage(); });
      head.appendChild(rm); g.appendChild(head);
      const list = CE('div', 'fc-glist'); g.appendChild(list);
      const gf = CE('div', 'fc-gfoot'), ab = CE('button', 'fc-mini', '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"></path></svg>Query'); ab.type = 'button';
      ab.addEventListener('click', () => { list.appendChild(mkQuery('', lastOp(list))); sync(); stage(); const t = [...list.querySelectorAll('.fc-ta')].pop(); if (t) t.focus(); });
      gf.appendChild(ab); g.appendChild(gf); it.appendChild(g);
      (nodes && nodes.length ? nodes : [{ text: '' }, { text: '', op: 'OR' }]).forEach(n => list.appendChild(mkQuery(n.text, n.op)));
      return it;
    };
    const seed = (m.tree && m.tree.length) ? m.tree
      : (m.queries && m.queries.length ? m.queries.map((q, i) => ({ kind: 'q', text: q, op: (m.joiners || [])[i - 1] })) : [{ kind: 'q', text: '' }]);
    seed.forEach(n => qs.appendChild(n.kind === 'g' ? mkGroup(n.children, n.op) : mkQuery(n.text, n.op)));
    addQBtn.addEventListener('click', () => { qs.appendChild(mkQuery('', lastOp(qs))); sync(); stage(); const t = [...qs.querySelectorAll('.fc-ta')].pop(); if (t) t.focus(); });
    addGBtn.addEventListener('click', () => { qs.appendChild(mkGroup(null, lastOp(qs))); sync(); stage(); const t = [...qs.querySelectorAll('.fc-ta')].pop(); if (t) t.focus(); });
    wrap.appendChild(f4); sync();
    requestAnimationFrame(() => qs.querySelectorAll('.fc-ta').forEach(t => t._fit && t._fit()));
  }
  buildEditor(row) {
    const type = row.dataset.type;
    const wrap = document.createElement('div');
    wrap.className = 'fc-body';
    if (type === 'year') {
      const yr = row._draft ? row._draft : (row._applied ? row._applied : this.cfgDefault('year'));
      const stage = () => { row._draft = this.cfgRead('year', wrap); };
      wrap.innerHTML = '<div class="fc-fields"><label class="fc-field"><span>From</span><input class="fc-from" type="number" placeholder="Any" value="' + yr.from + '"></label><span class="fc-dash">–</span><label class="fc-field"><span>To</span><input class="fc-to" type="number" placeholder="Any" value="' + yr.to + '"></label></div>'
        + '<div class="fc-chips"><button data-preset="last5">Last 5 years</button><button data-preset="2020">Since 2020</button><button data-preset="any">Any year</button></div>';
      const fromI = wrap.querySelector('.fc-from'), toI = wrap.querySelector('.fc-to');
      fromI.addEventListener('input', stage); toI.addEventListener('input', stage);
      wrap.querySelectorAll('.fc-chips button').forEach(b => b.addEventListener('click', () => {
        const p = b.dataset.preset;
        if (p === 'any') { fromI.value = ''; toI.value = ''; }
        else if (p === '2020') { fromI.value = '2020'; toI.value = ''; }
        else if (p === 'last5') { fromI.value = '2022'; toI.value = '2026'; }
        stage();
      }));
    } else if (type === 'citation') {
      const cv = row._draft ? row._draft : (row._applied ? row._applied : this.cfgDefault('citation'));
      const stage = () => { row._draft = this.cfgRead('citation', wrap); };
      const CUR = 2026;
      wrap.innerHTML = '<div class="fc-field"><span class="fc-t">Citations per year of age (β)</span><span class="fc-d fc-beta-note"></span><input class="fc-beta" type="number" min="0" inputmode="numeric" placeholder="e.g. 30" value="' + cv.beta + '"></div>'
        + '<div class="fc-field"><span class="fc-t">Minimum citations for ' + CUR + ' papers</span><span class="fc-d">Optional; leave blank to let all ' + CUR + ' papers pass.</span><input class="fc-cur" type="number" min="0" inputmode="numeric" placeholder="e.g. 5" value="' + cv.cur + '"></div>';
      const betaI = wrap.querySelector('.fc-beta'), curI = wrap.querySelector('.fc-cur'), note = wrap.querySelector('.fc-beta-note');
      const updNote = () => {
        const raw = betaI.value.trim(), b = Number(raw);
        if (raw && /^\d+$/.test(raw)) note.textContent = 'e.g., if β=' + b + ', a ' + (CUR - 1) + ' paper needs ' + b + ' citations to pass; a ' + (CUR - 2) + ' paper needs ' + (b * 2) + '; a ' + (CUR - 3) + ' paper needs ' + (b * 3) + ', and so on. ' + CUR + ' papers are governed by the field below.';
        else note.textContent = 'Each paper must have at least β citations for every year since it was published.';
      };
      const onInput = () => { updNote(); stage(); };
      betaI.addEventListener('input', onInput); curI.addEventListener('input', onInput);
      updNote();
      // Saved filter groups — one library with the Runs Refine pane (localStorage bridge)
      let groups = [];
      try { groups = JSON.parse(localStorage.getItem('kl-filter-groups') || '[]') || []; } catch (e) { groups = []; }
      if (groups.length) {
        const gw = document.createElement('div');
        gw.className = 'fc-field';
        gw.innerHTML = '<span class="fc-t">Saved groups</span><span class="fc-d">Saved eligibility thresholds: click to load a \u03b2 value.</span><div class="fc-chips"></div>';
        wrap.appendChild(gw);
        const rowEl = gw.querySelector('.fc-chips');
        rowEl.innerHTML = groups.map((x, i) => '<button type="button" data-gi="' + i + '">' + x.name + '</button>').join('');
        rowEl.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
          const x = groups[+b.dataset.gi]; if (!x) return;
          betaI.value = String(x.beta);
          onInput();
        }));
      }
    } else if (type === 'keyword') {
      const mm = row._draft ? row._draft : (row._applied ? row._applied : this.cfgDefault('keyword'));
      const stage = () => { row._draft = this.cfgRead('keyword', wrap); };
      wrap.innerHTML = '<div class="fc-field"><span class="fc-t">Search in</span><div class="fc-seg" data-variant="Segmented"><button data-t="Abstract">Abstract</button><button data-t="Title">Title</button></div></div>'
        + '<div class="fc-field"><span class="fc-t">Query<button class="q-info" type="button" data-tip="Combine terms with AND / OR, group with ( ), truncate with *, quote an exact phrase with &quot; &quot;." aria-label="Query syntax"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"></circle><path d="M12 16v-5M12 8h.01"></path></svg></button></span><textarea class="fc-ta" placeholder="(automorph* OR modular) AND L-function"></textarea></div>';
      const seg = [...wrap.querySelectorAll('.fc-seg button')], ta = wrap.querySelector('.fc-ta');
      ta.value = mm.query || '';
      let cur = mm.target;
      const paint = () => seg.forEach(b => b.dataset.on = (b.dataset.t === cur) ? '1' : '0');
      paint();
      seg.forEach(b => b.addEventListener('click', () => { cur = b.dataset.t; paint(); stage(); }));
      ta.addEventListener('input', stage);
    } else if (type === 'venue') {
      const vm = row._draft ? row._draft : (row._applied ? row._applied : this.cfgDefault('venue'));
      const stage = () => { row._draft = this.cfgRead('venue', wrap); };
      wrap.innerHTML = '<div class="fc-field"><span class="fc-t">Query<button class="q-info" type="button" data-tip="Combine venue names with AND / OR / NOT, group with ( ), truncate with *, quote an exact name with &quot; &quot;." aria-label="Query syntax"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"></circle><path d="M12 16v-5M12 8h.01"></path></svg></button></span>'
        + '<span class="fc-d">Matches the publication venue: journal, conference or repository.</span>'
        + '<textarea class="fc-ta" placeholder="Nature OR Science OR &quot;Cell Reports&quot;"></textarea></div>';
      const ta = wrap.querySelector('.fc-ta');
      ta.value = vm.query || '';
      ta.addEventListener('input', () => { this.fieldOk(ta); stage(); });
    } else if (type === 'similarity') {
      const sm = row._draft ? row._draft : (row._applied ? row._applied : this.cfgDefault('similarity'));
      const stage = () => { row._draft = this.cfgRead('similarity', wrap); };
      wrap.innerHTML = '<div class="fc-field"><span class="fc-t">Minimum similarity</span><span class="fc-d">Cosine similarity to the seed set, from 0 to 1. Scores run low in practice, useful thresholds sit around 0.02 – 0.1.</span><input class="fc-min" type="number" min="0" max="1" step="0.01" inputmode="decimal" placeholder="e.g. 0.05" style="font-variant-numeric:tabular-nums;"></div>'
        + '<div class="fc-chips"><button type="button" data-v="0.02">0.02  broad</button><button type="button" data-v="0.05">0.05  balanced</button><button type="button" data-v="0.1">0.10  strict</button></div>'
        + '<div class="fc-field"><span class="fc-t">Papers with no similarity score</span><span class="fc-d">Some papers have no abstract or embedding to score against.</span><div class="fc-seg" data-variant="Segmented"><button data-t="Pass">Pass</button><button data-t="Reject">Reject</button></div></div>';
      const mi = wrap.querySelector('.fc-min');
      mi.value = sm.min || '';
      mi.addEventListener('input', () => { this.fieldOk(mi); stage(); });
      wrap.querySelectorAll('.fc-chips button').forEach(b => b.addEventListener('click', () => { mi.value = b.dataset.v; this.fieldOk(mi); stage(); }));
      const seg = [...wrap.querySelectorAll('.fc-seg button')];
      let cur = sm.missing === 'Reject' ? 'Reject' : 'Pass';
      const paint = () => seg.forEach(b => b.dataset.on = (b.dataset.t === cur) ? '1' : '0');
      paint();
      seg.forEach(b => b.addEventListener('click', () => { cur = b.dataset.t; paint(); stage(); }));
    } else if (type === 'llm') {
      this.buildLLM(wrap, row);
    } else {
      const lbl = row.querySelector('.cf-type'); 
      wrap.innerHTML = '<div class="fc-label">' + (lbl ? lbl.textContent : 'Filter') + '</div><div class="fc-hint">Configuration for this filter type is coming soon.</div>';
    }
    return wrap;
  }
  filterName(type) { const m = { year: 'Year', citation: 'Citation', keyword: 'Keyword', author: 'Author', venue: 'Venue', llm: 'LLM', similarity: 'Similarity' }; return (m[type] || 'Filter') + ' Filter'; }
  typeColor(type) {
    const m = { year: '--f1', citation: '--f2', keyword: '--f3', author: '--f4', venue: '--f5', llm: '--f6', similarity: '--f7' };
    return 'var(' + (m[type] || '--accent') + ')';
  }
  openDrill(row) {
    const root = this.rootRef.current, panel = root.querySelector('.panel-config');
    const type = row.dataset.type, col = this.typeColor(type), label = this.filterName(type);
    const ic = this.cfIcons()[type] || this.cfIcons().keyword;
    const selCard = root.querySelector('.pb-card[data-sel="1"]');
    const stepName = (selCard && (selCard.dataset.title || (selCard.querySelector('.pb-title') || {}).textContent)) || 'Step';
    const d = document.createElement('div'); d.className = 'cfg-detail'; d.setAttribute('data-open', '0');
    d.innerHTML = '<div class="cfg-detail-head">'
      + '<button class="cfg-back" type="button" aria-label="Back to pipeline"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"></path></svg></button>'
      + '<span class="cfg-detail-node" style="color:' + col + '; border-color:' + col + ';"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' + ic + '</svg></span>'
      + '<span class="cfg-detail-title"><span class="s">' + stepName + '</span><span class="t">' + label + '</span></span>'
      + '</div>'
      + '<div class="cfg-detail-body"></div>'
      + '<div class="cfg-foot" data-show="1"><span class="cfg-foot-status"></span><span class="cfg-foot-btns"><button class="cfg-cancel" type="button">Cancel</button><button class="cfg-apply" type="button">Apply</button></span></div>';
    d.querySelector('.cfg-detail-body').appendChild(this.buildEditor(row));
    panel.appendChild(d);
    const st = d.querySelector('.cfg-foot-status'); st.textContent = 'Editing ' + label;
    const cancel = d.querySelector('.cfg-cancel'); cancel.style.display = row.dataset.applied === '1' ? '' : 'none';
    d.querySelector('.cfg-apply').addEventListener('click', () => this.applyConfig());
    d.querySelector('.cfg-cancel').addEventListener('click', () => this.cancelConfig());
    d.querySelector('.cfg-back').addEventListener('click', () => { if (row._draft) row._silentChecked = true; this.closeConfig(); });
    if (row._draft && row._silentChecked) { const n = this.validateEditor(row); if (n) { st.dataset.state = 'err'; st.textContent = n + (n === 1 ? ' issue to fix' : ' issues to fix'); } }
    requestAnimationFrame(() => d.setAttribute('data-open', '1'));
  }
  openConfig(row) {
    this.closeConfig();
    this._cfgRow = row;
    this.openDrill(row);
  }
  _openConfigOld(row) {
    const cfg = this.props.configStyle || 'Inline';
    const sel = this.props.selectStyle || 'Dim rest';
    const col = this.typeColor(row.dataset.type);
    const label = row.dataset.label || 'Filter';
    const pipe = row.closest('.cfg-pipe');

    // --- Selection style: how the clicked row + its siblings react ---
    if (sel === 'Dim rest') {
      pipe.querySelectorAll('.cf-filter').forEach(f => { if (f !== row) f.style.opacity = '.34'; });
      this._focused = true;
    } else if (sel === 'Lift') {
      row.style.boxShadow = '0 8px 22px rgba(var(--sh),.16)';
      row.style.transform = 'translateY(-1px)';
      row.style.zIndex = '3';
    }
    // 'Highlight only' relies on the existing [data-sel] highlight — no sibling change.

    // --- Config display: how the editor is presented ---
    if (cfg === 'Popover') {
      const pop = document.createElement('div'); pop.className = 'cfg-pop cf-editor';
      const holder = document.createElement('div');
      holder.innerHTML = '<div style="display:flex; align-items:center; gap:8px; margin-bottom:11px; font-size:10.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:' + col + ';"><span style="width:6px; height:6px; border-radius:50%; background:' + col + '; flex:none;"></span>' + label + '</div>';
      holder.appendChild(this.buildEditor(row));
      pop.appendChild(holder);
      pop.addEventListener('click', e => e.stopPropagation());
      const host = row.closest('.panel-config') || pipe;
      host.appendChild(pop);
      const hr = host.getBoundingClientRect(), rr = row.getBoundingClientRect();
      pop.style.width = row.offsetWidth + 'px';
      pop.style.left = (rr.left - hr.left) + 'px';
      pop.style.top = (rr.bottom - hr.top + 7) + 'px';
      this.afterOpen(row);
      requestAnimationFrame(() => pop.setAttribute('data-open', '1'));
      return;
    }

    const ed = document.createElement('div'); ed.className = 'cf-editor';
    const trans = 'transition:max-height .28s cubic-bezier(.22,.61,.36,1), opacity .22s ease;';
    ed.style.cssText = 'overflow:hidden; max-height:0; opacity:0; ' + trans;
    const inner = document.createElement('div');
    if (cfg === 'Framed card') {
      const head = '<div style="display:flex; align-items:center; gap:8px; padding:9px 12px; background:var(--card); border-bottom:1px solid var(--border); font-size:10.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:' + col + ';"><span style="width:6px; height:6px; border-radius:50%; background:' + col + '; flex:none;"></span>' + label + '</div>';
      inner.innerHTML = head + '<div class="fc-pad" style="padding:13px;"></div>';
      inner.querySelector('.fc-pad').appendChild(this.buildEditor(row));
      inner.style.cssText = 'margin:8px 0; border:1px solid var(--border); border-radius:12px; background:var(--card2); overflow:hidden; box-shadow:0 3px 10px rgba(var(--sh),.06);';
    } else {
      inner.style.cssText = 'padding:12px 12px 4px;';
      inner.appendChild(this.buildEditor(row));
    }
    ed.appendChild(inner);
    row.insertAdjacentElement('afterend', ed);
    ed.addEventListener('transitionend', e => { if (e.propertyName === 'max-height' && ed.style.opacity === '1') ed.style.maxHeight = 'none'; });
    this.afterOpen(row);
    requestAnimationFrame(() => { ed.style.maxHeight = (inner.scrollHeight + 30) + 'px'; ed.style.opacity = '1'; });
  }
  closeConfig() {
    const root = this.rootRef.current; if (!root) return;
    root.querySelectorAll('.cf-editor').forEach(e => {
      if (e.classList.contains('cfg-pop')) { e.setAttribute('data-open', '0'); setTimeout(() => e.remove(), 180); }
      else { e.style.maxHeight = '0'; e.style.opacity = '0'; setTimeout(() => e.remove(), 280); }
    });
    root.querySelectorAll('.cf-filter').forEach(f => { f.style.flexWrap = ''; f.style.opacity = ''; f.style.boxShadow = ''; f.style.transform = ''; f.style.zIndex = ''; });
    const dt = root.querySelector('.cfg-detail'); if (dt) {
      dt.setAttribute('data-open', '0'); setTimeout(() => dt.remove(), 260);
      const scb = root.querySelector('.panel-config .cfg-scroll');
      if (scb && scb.animate) scb.animate([{ opacity: .4, transform: 'translateX(-10px)' }, { opacity: 1, transform: 'none' }], { duration: 260, easing: 'cubic-bezier(.22,.61,.36,1)' });
    }
    this._focused = false;
    const foot = root.querySelector('.cfg-foot'); if (foot) foot.setAttribute('data-show', '0');
    this._cfgRow = null;
  }
  getFooter() {
    const panel = this.rootRef.current.querySelector('.panel-config');
    let f = panel.querySelector('.cfg-foot');
    if (!f) {
      f = document.createElement('div'); f.className = 'cfg-foot';
      f.innerHTML = '<span class="cfg-foot-status"></span><span class="cfg-foot-btns"><button class="cfg-cancel" type="button">Cancel</button><button class="cfg-apply" type="button">Apply</button></span>';
      panel.appendChild(f);
      f.querySelector('.cfg-apply').addEventListener('click', () => this.applyConfig());
      f.querySelector('.cfg-cancel').addEventListener('click', () => this.cancelConfig());
    }
    return f;
  }
  showFooter(row) {
    const f = this.getFooter(), st = f.querySelector('.cfg-foot-status');
    st.removeAttribute('data-state'); st.textContent = 'Editing ' + (row.dataset.label || 'filter');
    const cancel = f.querySelector('.cfg-cancel'); if (cancel) cancel.style.display = row.dataset.applied === '1' ? '' : 'none';
    f.setAttribute('data-show', '1');
  }
  fieldErr(el, msg) {
    if (!el) return;
    const host = el.closest('.fc-field') || el.parentElement;
    if (!host) return;
    let e = [...host.children].find(c => c.classList && c.classList.contains('fc-err'));
    if (!e) {
      e = document.createElement('div'); e.className = 'fc-err';
      e.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"></circle><path d="M12 8v4M12 16h.01"></path></svg><span></span>';
      host.appendChild(e);
    }
    e.querySelector('span').textContent = msg;
    e.setAttribute('data-show', '1');
  }
  fieldOk(el) {
    if (!el) return;
    const host = el.closest('.fc-field') || el.parentElement;
    if (!host) return;
    const e = [...host.children].find(c => c.classList && c.classList.contains('fc-err'));
    if (e) e.setAttribute('data-show', '0');
  }
  validateEditor(row) {
    const ed = this.rootRef.current.querySelector('.cfg-detail'); if (!ed) return 0;
    const type = row.dataset.type; let issues = 0;
    const bad = (el, msg) => { if (el) { el.dataset.invalid = '1'; this.fieldErr(el, msg); issues++; } };
    const ok = el => { if (el) { el.dataset.invalid = '0'; this.fieldOk(el); } };
    if (type === 'citation') {
      const b = ed.querySelector('.fc-beta'), c = ed.querySelector('.fc-cur');
      [b, c].forEach(el => {
        if (!el) return;
        const raw = el.value.trim();
        if (raw === '') { ok(el); return; }
        if (!/^\d+$/.test(raw)) bad(el, /^-/.test(raw) ? 'Must be zero or greater (no negative values).' : 'Whole numbers only (no decimals, letters or symbols).');
        else ok(el);
      });
      if (b && b.value.trim() === '') bad(b, 'Required: enter the citations expected per year of age.');
    } else if (type === 'llm') {
      const tas = [...ed.querySelectorAll('.fc-qs .fc-ta')];
      tas.forEach(ok);
      if (!tas.some(t => t.value.trim()) && tas[0]) bad(tas[0], 'Enter at least one screening criterion.');
    } else if (type === 'keyword' || type === 'venue') {
      const ta = ed.querySelector('.fc-ta');
      if (ta) { const r = this.validateQuery(ta.value); if (r.ok) ok(ta); else bad(ta, 'Invalid query: ' + r.msg + '.'); }
    } else if (type === 'similarity') {
      const mi = ed.querySelector('.fc-min');
      if (mi) {
        const raw = mi.value.trim();
        if (raw === '') ok(mi);
        else if (!/^(0(\.\d+)?|\.\d+|1(\.0+)?)$/.test(raw)) bad(mi, 'Enter a number between 0 and 1 (e.g. 0.05).');
        else ok(mi);
      }
    }
    return issues;
  }
  applyConfig() {
    const root = this.rootRef.current, row = this._cfgRow; if (!row) return;
    const d = root.querySelector('.cfg-detail'); if (!d) return;
    const st = d.querySelector('.cfg-foot-status'), type = row.dataset.type;
    const issues = this.validateEditor(row);
    if (issues) { st.dataset.state = 'err'; st.textContent = issues + (issues === 1 ? ' issue to fix' : ' issues to fix'); return; }
    if (['year', 'citation', 'keyword', 'llm', 'venue', 'similarity'].indexOf(type) >= 0) {
      const dm = this.cfgRead(type, d.querySelector('.fc-body') || d);
      row._applied = dm; row.dataset.value = this.cfgSummary(type, dm);
      const v = row.querySelector('.cf-val'); if (v) v.textContent = row.dataset.value;
    }
    row.dataset.applied = '1'; row._draft = null; row._silentChecked = false;
    st.dataset.state = 'ok'; st.textContent = 'Applied ✓'; row.dataset.sel = '0'; setTimeout(() => this.closeConfig(), 950);
  }
  setupTips(root) {
    let tip = null, timer = null;
    const place = el => {
      const t = el.getAttribute('data-tip') || el.getAttribute('aria-label');
      if (!t) return;
      if (!tip) { tip = document.createElement('div'); tip.className = 'om-tip'; document.body.appendChild(tip); }
      tip.setAttribute('data-style', 'Card');
      { const rt = this.rootRef.current; if (rt) { const cs = getComputedStyle(rt); ['--card','--fg','--border','--panel','--fg2','--accent','--sh'].forEach(k => tip.style.setProperty(k, cs.getPropertyValue(k))); } }
      tip.textContent = t;
      tip.setAttribute('data-show', '1');
      const r = el.getBoundingClientRect();
      const tr = tip.getBoundingClientRect();
      let pos = 'top', top = r.top - tr.height - 8;
      if (top < 6) { pos = 'bottom'; top = r.bottom + 8; }
      let left = r.left + r.width / 2 - tr.width / 2;
      left = Math.max(6, Math.min(left, window.innerWidth - tr.width - 6));
      tip.setAttribute('data-pos', pos);
      tip.style.left = left + 'px';
      tip.style.top = top + 'px';
      tip.style.setProperty('--arrow', (r.left + r.width / 2 - left) + 'px');
    };
    const hide = () => { if (tip) tip.setAttribute('data-show', '0'); clearTimeout(timer); };
    root.addEventListener('click', e => { const b = e.target.closest('.q-info'); if (!b) return; e.preventDefault(); e.stopPropagation(); clearTimeout(timer); place(b); });
    const SEL = '[data-tip], button[aria-label], [role="button"][aria-label]';
    root.addEventListener('mouseover', e => { const el = e.target.closest(SEL); if (!el || !root.contains(el)) return; clearTimeout(timer); timer = setTimeout(() => place(el), 180); });
    root.addEventListener('mouseout', e => { const el = e.target.closest(SEL); if (el) hide(); });
    root.addEventListener('mousedown', hide);
    window.addEventListener('scroll', hide, true);

    this.setupPipeline(root);
    this.seedInit(root);
  }

  // ————— Import into the seed set (resolver flow; mock parse in import-resolver.js) —————
  sbImpWire(sb) {
    if (sb._impW) return; sb._impW = 1;
    const q = s => sb.querySelector(s);
    sb.querySelectorAll('.sb-imode button').forEach(b => b.addEventListener('click', () => {
      sb.setAttribute('data-sm', b.dataset.sm);
      sb.querySelectorAll('.sb-imode button').forEach(x => x.setAttribute('data-on', x === b ? '1' : '0'));
      if (b.dataset.sm === 'import') this.sbImpState(sb, sb._impRes ? 'review' : 'drop');
    }));
    const fi = q('.sbi-file'), br = q('.sbi-browse'), sm = q('.sbi-sample'), dz = q('.sbi-drop');
    if (br) { br.addEventListener('click', () => fi && fi.click()); br.addEventListener('mouseenter', () => { br.style.background = 'var(--row-hover)'; }); br.addEventListener('mouseleave', () => { br.style.background = 'var(--card)'; }); }
    if (fi) fi.addEventListener('change', () => { if (fi.files && fi.files.length) this.sbImpGo(sb, Array.prototype.slice.call(fi.files)); fi.value = ''; });
    if (sm) { sm.addEventListener('click', () => this.sbImpGo(sb, null)); sm.addEventListener('mouseenter', () => { sm.style.color = 'var(--accent)'; }); sm.addEventListener('mouseleave', () => { sm.style.color = 'var(--fg2)'; }); }
    if (dz) {
      ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.style.borderColor = 'var(--accent)'; dz.style.background = 'var(--row-hover)'; }));
      ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.style.borderColor = ''; dz.style.background = ''; }));
      dz.addEventListener('drop', e => { const fs = Array.prototype.slice.call((e.dataTransfer && e.dataTransfer.files) || []); if (fs.length) this.sbImpGo(sb, fs); });
    }
    const rev = q('.sbi-rev');
    if (rev) rev.addEventListener('click', e => {
      const ck = e.target.closest('.ki-ck');
      if (ck && sb._impRes) { const en = sb._impRes.entries[+ck.dataset.i]; if (en) { en.checked = !en.checked; this.sbImpRev(sb); } return; }
      const mu = e.target.closest('.ki-multi');
      if (mu) { e.stopPropagation(); this.sbImpCand(sb, mu, +mu.dataset.i); }
    });
    const ab = q('.sbi-addbtn');
    if (ab) ab.addEventListener('click', () => this.sbImpAdd(sb));
  }
  sbImpState(sb, s) {
    const q = c => sb.querySelector(c);
    if (q('.sbi-drop')) q('.sbi-drop').style.display = s === 'drop' ? 'flex' : 'none';
    if (q('.sbi-parse')) q('.sbi-parse').style.display = s === 'parse' ? 'block' : 'none';
    if (q('.sbi-rev')) q('.sbi-rev').style.display = s === 'review' ? 'block' : 'none';
    if (q('.sbi-foot')) q('.sbi-foot').style.display = s === 'review' ? 'flex' : 'none';
  }
  sbImpGo(sb, files) {
    const K = window.KLImport; if (!K) return;
    sb._impRes = files && files.length ? K.parse(files) : K.sample();
    const seeds = this._seeds || [];
    sb._impRes.entries.forEach(e => { if (e.state === 'ok' && seeds.some(s => s.title === e.ti)) { e.state = 'dupe'; e.checked = false; } });
    this.sbImpState(sb, 'parse');
    const host = sb.querySelector('.sbi-parse'); if (!host) return;
    host.innerHTML = '<div style="padding:4px var(--pin) 6px; font-size:10.5px; font-weight:600; letter-spacing:.07em; text-transform:uppercase; color:var(--muted2);">Parsing ' + sb._impRes.files.length + (sb._impRes.files.length === 1 ? ' file' : ' files') + '</div>' + sb._impRes.files.map((f, i) => K.fileRowHtml(f, i)).join('');
    const gen = sb._impGen = (sb._impGen || 0) + 1;
    const settle = i => {
      if (sb._impGen !== gen || !this.rootRef.current) return;
      if (i >= sb._impRes.files.length) { setTimeout(() => { if (sb._impGen === gen) { this.sbImpRev(sb); this.sbImpState(sb, 'review'); } }, 420); return; }
      const f = sb._impRes.files[i];
      const st = host.querySelector('.ki-file[data-f="' + i + '"] .ki-fst');
      if (st) {
        if (f.err) st.innerHTML = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="var(--error)" stroke-width="2.2" stroke-linecap="round"><path d="M12 8v5"></path><path d="M12 16.5h.01"></path><circle cx="12" cy="12" r="9"></circle></svg><span style="color:color-mix(in oklab, var(--error) 62%, var(--fg)); white-space:normal;">' + f.err + '</span>';
        else st.innerHTML = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="var(--muted2)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"></path></svg>' + f.n + (f.n === 1 ? ' reference' : ' references');
      }
      setTimeout(() => settle(i + 1), 340 + Math.random() * 280);
    };
    setTimeout(() => settle(0), 420);
  }
  sbImpRev(sb) {
    const K = window.KLImport; if (!K || !sb._impRes) return;
    const rev = sb.querySelector('.sbi-rev'); if (!rev) return;
    const gs = K.groups(sb._impRes.entries, 'In the seed set');
    sb._impOpen = sb._impOpen || {};
    // triage order + progressive disclosure: attention sections stay open, bulk sections
    // collapse to a count so "Couldn't match" is never buried under 40 matched rows
    rev.style.padding = '12px 12px 2px';
    rev.innerHTML = gs.map(g => {
      const open = sb._impOpen[g.key] != null ? sb._impOpen[g.key] : (g.key === 'multi' || g.key === 'none');
      return K.groupHtml(g, g.rows.map(i => K.rowHtml(sb._impRes.entries[i], i, {})), { open: open });
    }).join('');
    if (!rev._secW) {
      rev._secW = 1;
      rev.addEventListener('click', e => {
        const s = e.target.closest('.ki-sec'); if (!s) return;
        const body = rev.querySelector('.ki-body[data-k="' + s.dataset.k + '"]'); if (!body) return;
        const open = body.style.display === 'none';
        sb._impOpen[s.dataset.k] = open;
        this.sbImpRev(sb);
      });
    }
    this.sbImpFoot(sb);
  }
  sbImpFoot(sb) {
    const R = sb._impRes; if (!R) return;
    const n = R.entries.filter(e => e.checked && e.state === 'ok').length;
    const bad = R.entries.filter(e => e.state === 'none').length;
    const err = sb.querySelector('.sbi-foot-err');
    if (err) {
      err.style.display = bad ? 'block' : 'none';
      err.textContent = bad ? bad + (bad === 1 ? ' entry' : ' entries') + ' couldn\u2019t be matched: reported under \u201cCouldn\u2019t match\u201d, never added silently.' : '';
    }
    const ab = sb.querySelector('.sbi-addbtn');
    if (ab) {
      ab.textContent = 'Add ' + n + ' to seed set';
      ab.disabled = !n;
      ab.style.opacity = n ? '1' : '.45';
      ab.style.cursor = n ? 'pointer' : 'default';
    }
  }
  sbImpCand(sb, anchor, i) {
    const root = this.rootRef.current; if (!root) return;
    const K = window.KLImport, R = sb._impRes; if (!K || !R) return;
    const e = R.entries[i]; if (!e || !e.cand) return;
    root.querySelectorAll('.sbi-candpop').forEach(x => x.remove());
    if (getComputedStyle(root).position === 'static') root.style.position = 'relative';
    const el = document.createElement('div');
    el.className = 'sbi-candpop';
    el.style.cssText = 'position:absolute; z-index:80; width:228px; background:var(--card); border:1px solid var(--border); border-radius:12px; box-shadow:0 14px 34px rgba(var(--sh),.18); padding:7px; opacity:0; transform:translateY(4px); transition:opacity .16s ease, transform .16s ease;';
    el.innerHTML = K.candPopHtml(e);
    root.appendChild(el);
    const rr = root.getBoundingClientRect(), ar = anchor.getBoundingClientRect();
    el.style.left = Math.round(Math.max(10, Math.min(rr.width - 238, ar.right - rr.left - 228))) + 'px';
    const belowTop = ar.bottom - rr.top + 6, h = el.offsetHeight || 120;
    el.style.top = Math.round(belowTop + h > rr.height - 10 ? ar.top - rr.top - h - 6 : belowTop) + 'px';
    requestAnimationFrame(() => { el.style.opacity = '1'; el.style.transform = 'none'; });
    const close = () => { el.style.opacity = '0'; setTimeout(() => el.remove(), 170); document.removeEventListener('click', out); };
    const out = ev => { if (!el.contains(ev.target)) close(); };
    setTimeout(() => document.addEventListener('click', out), 0);
    el.querySelectorAll('.ki-cand').forEach(b => {
      b.addEventListener('mouseenter', () => { b.style.background = 'var(--row-hover)'; });
      b.addEventListener('mouseleave', () => { b.style.background = 'none'; });
      b.addEventListener('click', ev => {
        ev.stopPropagation();
        const c = e.cand[+b.dataset.c];
        if (c) { e.ve = c.ve; e.yr = c.yr; e.ci = c.ci; }
        e.state = 'ok'; e.checked = true;
        close();
        this.sbImpRev(sb);
      });
    });
  }
  sbImpAdd(sb) {
    const R = sb._impRes; if (!R) return;
    const picks = R.entries.filter(e => e.checked && e.state === 'ok');
    if (!picks.length) return;
    const ab = sb.querySelector('.sbi-addbtn');
    if (ab) { ab.disabled = true; ab.textContent = 'Adding ' + picks.length + '\u2026'; ab.style.opacity = '.7'; }
    setTimeout(() => {
      if (!this.rootRef.current) return;
      this.seedImport(picks);
      sb._impRes = null;
      sb.setAttribute('data-sm', 'search');
      sb.querySelectorAll('.sb-imode button').forEach(x => x.setAttribute('data-on', x.dataset.sm === 'search' ? '1' : '0'));
      this.sbImpState(sb, 'drop');
      if (ab) { ab.disabled = false; ab.style.opacity = '1'; }
    }, 700);
  }
  seedImport(entries) {
    const seeds = this._seeds = this._seeds || [];
    let first = null;
    entries.forEach(e => {
      const info = { title: e.ti, authors: e.au, venue: e.ve, year: String(e.yr) };
      if (!info.title || seeds.some(s => s.title === info.title)) return;
      seeds.unshift(info);
      if (!first) first = info.title;
    });
    if (first) this._seedNew = first;
    const src = this.plSteps().find(s => s.type === 'source');
    if (src && this._plSel !== src.id) { this._plSel = src.id; this.plRender(); }
    else this.seedRender();
  }
  // ================= seed set =================
  seedInfo(row) {
    const t = row.querySelector('.pr-title'), a = row.querySelector('.pr-authors');
    return {
      title: (t ? t.textContent : '').trim(),
      authors: (a ? a.textContent : '').trim(),
      venue: row.dataset.venue || '', year: row.dataset.year || ''
    };
  }
  seedMark(title, on) {
    const root = this.rootRef.current; if (!root) return;
    root.querySelectorAll('.sidebar .pr').forEach(r => {
      const t = r.querySelector('.pr-title');
      if (!t || t.textContent.trim() !== title) return;
      r.dataset.seeded = on ? '1' : '0';
      if (r.__star) { r.__star.dataset.saved = on ? '1' : '0'; if (r.__setStar) r.__setStar(); }
    });
    root.querySelectorAll('.sidebar').forEach(sb => { if (sb._applyList) sb._applyList(); });
  }
  seedAdd(row) {
    const seeds = this._seeds = this._seeds || [];
    const info = this.seedInfo(row);
    if (!info.title || seeds.some(s => s.title === info.title)) return;
    seeds.unshift(info);
    this._seedNew = info.title;
    this.seedMark(info.title, true);
    const src = this.plSteps().find(s => s.type === 'source');
    if (src && this._plSel !== src.id) { this._plSel = src.id; this.plRender(); }
    else this.seedRender();
  }
  seedRemove(title) {
    this._seeds = (this._seeds || []).filter(s => s.title !== title);
    this.seedMark(title, false);
    this.seedRender();
  }
  seedRender() {
    const root = this.rootRef.current; if (!root) return;
    const seeds = this._seeds = this._seeds || [];
    const n = seeds.length, lbl = n + (n === 1 ? ' paper' : ' papers');
    const cnt = root.querySelector('.cfg-seed-count');
    if (cnt) cnt.textContent = lbl;
    const segn = root.querySelector('.cfg-src-n'); if (segn) segn.textContent = n;
    root.querySelectorAll('.pb-card').forEach(c => {
      const code = c.querySelector('.pb-code');
      if (code && code.textContent.indexOf('SED') === 0) { const chip = c.querySelector('.pb-chip'); if (chip) chip.textContent = lbl; }
    });
    const list = root.querySelector('.cfg-seed-list'); if (!list) return;
    list.innerHTML = '';
    if (!n) {
      const e = document.createElement('div'); e.className = 'cfg-seed-empty';
      e.innerHTML = 'No seed papers yet.<br>Star <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-1.5px;"><path d="M11.48 3.5a.5.5 0 0 1 1 0l2.4 5.05 5.5.7a.5.5 0 0 1 .28.86l-4.05 3.8 1.05 5.5a.5.5 0 0 1-.74.53L12 17.7l-4.9 2.54a.5.5 0 0 1-.73-.53l1.05-5.5-4.05-3.8a.5.5 0 0 1 .28-.86l5.5-.7z"></path></svg> a paper in the search panel to add it here.';
      list.appendChild(e);
      return;
    }
    const bars = [...root.querySelectorAll('.sidebar')];
    const bar = bars.find(b => getComputedStyle(b).display !== 'none') || bars[0];
    if (!bar) return;
    const cards = bar.dataset.style === 'cards';
    // cards variant sets its own box; list variant clears inline styles so the
    // .cfg-seed-list class rule (well 6px gutter, margin:0 -8px) owns the layout
    list.style.margin = cards ? '0' : '';
    list.style.gap = cards ? '10px' : '';
    list.style.padding = cards ? '2px 0' : '';
    const src = {};
    bar.querySelectorAll('.pr').forEach(r => {
      const t = r.querySelector('.pr-title'); if (t) src[t.textContent.trim()] = r;
    });
    seeds.forEach((s, i) => {
      const origin = src[s.title]; if (!origin) return;
      const r = origin.cloneNode(true);
      r.className = 'pr cs-row';
      delete r.dataset.seeded; delete r.dataset.selected; delete r.dataset.hover;
      r.style.display = '';
      r.dataset.rowStyle = cards ? 'cards' : 'list';
      // full-bleed list rows keep the sidebar card's own box (incl. the 46px lane the trash reuses)
      if (!cards) r.style.borderBottom = i === seeds.length - 1 ? 'none' : '1px solid var(--divider)';
      r.querySelectorAll('button, .abs-panel, .pr-abs').forEach(b => {
        const holder = b.parentElement;
        b.remove();
        if (holder && holder !== r && !holder.children.length && !holder.textContent.trim()) holder.remove();
      });
      const del = document.createElement('button');
      del.className = 'cs-del'; del.setAttribute('aria-label', 'Remove from seed set');
      del.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13"></path></svg>';
      del.addEventListener('click', e => { e.stopPropagation(); this.collapseOut(r, () => this.seedRemove(s.title)); });
      r.appendChild(del);
      this.wireRow(r, root);
      r.querySelectorAll('button[aria-label="Save paper"]').forEach(b => b.remove());
      if (r.__paint) r.__paint();
      this.paintVenue(r);
      list.appendChild(r);
    });
    const host = root.querySelector('.panel-config') || list;
    if (host._seedAbs) { host._seedAbs.remove(); host._seedAbs = null; }
    const before = host.lastElementChild;
    this.wireAbstract(host, 'drill', 'smart');
    if (host.lastElementChild && host.lastElementChild !== before) host._seedAbs = host.lastElementChild;
    // no 46px reservation: the hover-revealed trash overlays the row's right edge (title keeps full width)
    if (this._seedNew) {
      const t = this._seedNew; this._seedNew = null;
      const nr = [...list.querySelectorAll('.cs-row')].find(el => { const x = el.querySelector('.pr-title'); return x && x.textContent.trim() === t; });
      if (nr) {
        this._seedScrollTo = nr;
      }
      if (nr && nr.animate) {
        const h = nr.offsetHeight;
        nr.style.overflow = 'hidden';
        const ac = getComputedStyle(this.rootRef.current).getPropertyValue('--accent').trim() || '#cc785c';
        nr.animate([{ height: '0px', opacity: 0 }, { height: h + 'px', opacity: 1 }], { duration: 260, easing: 'cubic-bezier(.22,.61,.36,1)' }).onfinish = () => { nr.style.overflow = ''; };
        this.scrollToTop(nr);
        nr.animate([{ boxShadow: 'inset 3px 0 0 ' + ac, backgroundColor: 'color-mix(in oklab, ' + ac + ' 14%, transparent)' }, { boxShadow: 'inset 3px 0 0 rgba(0,0,0,0)', backgroundColor: 'rgba(0,0,0,0)' }], { duration: 1200, easing: 'ease-out' });
      }
    }
  }
  seedInit(root) {
    const seeds = this._seeds = this._seeds || [];
    if (seeds.length) { this.seedRender(); return; }
    const seen = new Set();
    const rows = [...root.querySelectorAll('.sidebar .pr')].reverse();
    for (const r of rows) {
      const info = this.seedInfo(r);
      if (!info.title || seen.has(info.title)) continue;
      seen.add(info.title);
      seeds.push(info);
      if (seeds.length >= 7) break;
    }
    seeds.forEach(s => this.seedMark(s.title, true));
    this.seedRender();
  }
  // ================= pipeline: data model =================
  plIconSvg(type, size) {
    const P = {
      source: '<path d="M12 3.6l2.5 5.2 5.7.8-4.1 4 1 5.7-5.1-2.7-5.1 2.7 1-5.7-4.1-4 5.7-.8z"></path>',
      fwd: '<path d="M7 17L17 7"></path><path d="M9 7h8v8"></path>',
      bwd: '<path d="M17 7L7 17"></path><path d="M15 17H7V9"></path>',
      db: '<ellipse cx="12" cy="5.5" rx="7.5" ry="2.8"></ellipse><path d="M4.5 5.5v13c0 1.6 3.4 2.9 7.5 2.9s7.5-1.3 7.5-2.9v-13"></path><path d="M4.5 12c0 1.6 3.4 2.9 7.5 2.9s7.5-1.3 7.5-2.9"></path>',
      sem: '<circle cx="12" cy="12" r="2.6"></circle><path d="M6.5 6.5a7.8 7.8 0 0 0 0 11M17.5 6.5a7.8 7.8 0 0 1 0 11"></path>',
      rrk: '<path d="M4 6h16M4 12h11M4 18h6"></path>'
    };
    return '<svg width="' + size + '" height="' + size + '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" style="display:block;">' + (P[type] || P.fwd) + '</svg>';
  }
  plMeta() {
    return {
      source: { name: 'Seed Set', prefix: 'SED', kind: 'database', label: 'Database', dot: 'teal',
                desc: 'The starting papers that every other step grows from.' },
      fwd:    { name: 'Forward Searcher', prefix: 'FWD', kind: 'network', label: 'Citation', dot: 'gold',
                desc: 'Follows every citation of each paper. The filter pipeline below shapes what is kept.' },
      bwd:    { name: 'Backward Searcher', prefix: 'BWD', kind: 'network', label: 'Citation', dot: 'gold',
                desc: 'Walks every reference of each paper. The filter pipeline below shapes what is kept.' },
      db:     { name: 'Database Searcher', prefix: 'DBS', kind: 'database', label: 'Database', dot: 'teal',
                desc: 'Writes and runs queries against paper databases, then screens what comes back.' },
      sem:    { name: 'Semantic Searcher', prefix: 'SEM', kind: 'semantics', label: 'Semantics', dot: 'purple',
                desc: 'Finds papers close in meaning to the current set, beyond the citation graph.' },
      rrk:    { name: 'Diversified Reranker', prefix: 'RRK', kind: 'reranker', label: 'Reranker', dot: 'rose',
                desc: 'Keeps the strongest papers while spreading them across clusters.' }
    };
  }
  setupProjMenu(root) {
    const wrap = root.querySelector('.tb-proj-wrap'); if (!wrap || wrap._w) return;
    wrap._w = 1;
    this._pjRoot = root;
    const btn = wrap.querySelector('.tb-project'), menu = wrap.querySelector('.tb-proj-menu');
    if (!btn || !menu) return;
    const closer = e => { if (wrap.contains(e.target)) return; close(); };
    const close = () => { menu.style.display = 'none'; btn.style.background = 'transparent'; document.removeEventListener('click', closer); };
    btn.addEventListener('click', () => {
      if (menu.style.display === 'block') { close(); return; }
      menu.style.display = 'block';
      btn.style.background = 'var(--row-hover)';
      menu.animate([{ opacity: 0, transform: 'translateY(-5px) scale(.97)' }, { opacity: 1, transform: 'none' }], { duration: 180, easing: 'cubic-bezier(.22,1.1,.36,1)' });
      setTimeout(() => document.addEventListener('click', closer), 0);
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
    menu.querySelectorAll('.tb-proj-item').forEach(it => it.addEventListener('click', () => {
      close();
      const nm = it.querySelector('.tb-proj-nm');
      document.dispatchEvent(new CustomEvent('kl-project', { detail: { name: nm ? nm.textContent : '' } }));
    }));
    menu.querySelectorAll('.tb-proj-new, .tb-proj-all').forEach(b => b.addEventListener('click', () => { close(); b.dispatchEvent(new CustomEvent('kl-home', { bubbles: true, composed: true, detail: { new: b.classList.contains('tb-proj-new') } })); }));
    document.addEventListener('kl-project', e => this.applyProject(e.detail && e.detail.name));
    this.paintProjAnchor();
  }
  applyProject(name) {
    const root = this._pjRoot; if (!root || !name) return;
    const nm = root.querySelector('.tb-proj-name');
    // project switch = context switch: the workspace fades through (same recipe as page transitions)
    if (nm && nm.textContent !== name) {
      const m = root.querySelector('main');
      if (m && m.animate) m.animate([
        { opacity: 1, transform: 'none' },
        { opacity: 0, transform: 'none', offset: .3 },
        { opacity: 0, transform: 'translateY(7px)', offset: .34 },
        { opacity: 1, transform: 'none' }
      ], { duration: 420, easing: 'cubic-bezier(.22,.61,.36,1)' });
    }
    if (nm) nm.textContent = name;
    root.querySelectorAll('.tb-proj-item').forEach(it => {
      const n = it.querySelector('.tb-proj-nm'), cur = !!n && n.textContent === name;
      it.setAttribute('data-cur', cur ? '1' : '0');
      const ck = it.querySelector('.tb-proj-check'); if (ck) ck.style.display = cur ? 'block' : 'none';
    });
  }
  paintProjAnchor() {
    const root = this._pjRoot; if (!root) return;
    const v = this.props.projAnchor ?? 'Folder tile';
    root.querySelectorAll('.tb-proj-item').forEach(it => {
      [['.tb-anch-l', 'Letter tile'], ['.tb-anch-m', 'Mark tile'], ['.tb-anch-f', 'Folder tile']].forEach(pair => {
        const el = it.querySelector(pair[0]); if (el) el.style.display = v === pair[1] ? 'flex' : 'none';
      });
    });
  }
  setupUserMenu(root) {
    const wrap = root.querySelector('.tb-user-wrap'); if (!wrap || wrap._w) return;
    wrap._w = 1;
    const btn = wrap.querySelector('.tb-user'), menu = wrap.querySelector('.tb-user-menu');
    if (!btn || !menu) return;
    const closer = e => { if (wrap.contains(e.target)) return; close(); };
    const close = () => { menu.style.display = 'none'; document.removeEventListener('click', closer); };
    btn.addEventListener('click', () => {
      if (menu.style.display === 'block') { close(); return; }
      menu.style.display = 'block';
      menu.style.animation = 'dlPop .18s cubic-bezier(.22,1.1,.36,1)';
      setTimeout(() => document.addEventListener('click', closer), 0);
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
    const st = wrap.querySelector('.tb-user-settings');
    if (st) st.addEventListener('click', () => { close(); st.dispatchEvent(new CustomEvent('kl-open-settings', { bubbles: true })); });
    const lo = wrap.querySelector('.tb-user-logout');
    if (lo) lo.addEventListener('click', () => { close(); lo.dispatchEvent(new CustomEvent('kl-logout', { bubbles: true, composed: true })); });
    this.klFsRow(wrap, close);
  }
  // "Enter full screen" — touch-only row (a home-screen launch is already chrome-less)
  klFsRow(wrap, close) {
    const st = wrap && wrap.querySelector('.tb-user-settings');
    if (!st || wrap._fsRow) return;
    const de = document.documentElement;
    const req = de.requestFullscreen || de.webkitRequestFullscreen;
    let touch = false; try { touch = matchMedia('(hover:none)').matches; } catch (e) {}
    if (!req || !touch || window.navigator.standalone) return;
    wrap._fsRow = 1;
    const row = st.cloneNode(true);
    row.classList.add('tb-user-fs');
    const ic = row.querySelector('svg');
    if (ic) ic.innerHTML = '<path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M8 21H5a2 2 0 0 1-2-2v-3M16 21h3a2 2 0 0 0 2-2v-3"></path>';
    const txt = Array.prototype.slice.call(row.childNodes).reverse().filter(n => n.nodeType === 3)[0] || row.appendChild(document.createTextNode(''));
    const sync = () => { txt.textContent = (document.fullscreenElement || document.webkitFullscreenElement) ? ' Exit full screen ' : ' Enter full screen '; };
    sync();
    st.parentNode.insertBefore(row, st);
    row.addEventListener('click', () => {
      if (close) close();
      const on = document.fullscreenElement || document.webkitFullscreenElement;
      try { on ? (document.exitFullscreen || document.webkitExitFullscreen).call(document) : req.call(de); } catch (e) {}
      setTimeout(sync, 500);
    });
  }
  dlLatestRun() {
    // Build only knows the latest completed run (mock: RUN-06); none before the first run.
    return (this.props.pageState || 'Has results') === 'Before first run' ? null : { id: 'RUN-06', date: 'yesterday', acc: 214 };
  }
  setupDownload(root) {
    const wrap = root.querySelector('.tb-dl-wrap'); if (!wrap || wrap._w) return;
    wrap._w = 1;
    const btn = wrap.querySelector('.tb-download'), menu = wrap.querySelector('.tb-dl-menu');
    if (!btn || !menu) return;
    const paint = () => {
      const run = this.dlLatestRun();
      menu.querySelectorAll('.tb-dl-item').forEach(b => b.style.display = run ? 'flex' : 'none');
      const none = menu.querySelector('.tb-dl-none'); if (none) none.style.display = run ? 'none' : 'flex';
      menu.querySelectorAll('.tb-dl-sep, .tb-dl-run, .tb-dl-date').forEach(el => el.style.display = run ? '' : 'none');
      if (run) {
        menu.querySelector('.tb-dl-run').textContent = String(run.id).replace(/^RUN-0*/i, 'Run ');
        menu.querySelector('.tb-dl-date').textContent = run.date || '';
        const n = menu.querySelector('.tb-dl-n'); if (n) n.textContent = this.kfmt(run.acc);
      }
    };
    const closer = e => { if (wrap.contains(e.target)) return; close(); };
    const close = () => { menu.style.display = 'none'; document.removeEventListener('click', closer); };
    btn.addEventListener('click', () => {
      if (menu.style.display === 'block') { close(); return; }
      paint();
      menu.style.display = 'block';
      menu.style.animation = 'dlPop .18s cubic-bezier(.22,1.1,.36,1)';
      setTimeout(() => document.addEventListener('click', closer), 0);
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
    menu.querySelectorAll('.tb-dl-item').forEach(b => b.addEventListener('click', () => {
      const sub = b.querySelector('.tb-dl-sub'); if (!sub || b._busy) return;
      b._busy = 1;
      const old = sub.innerHTML;
      sub.innerHTML = '<span style="color:var(--success); font-weight:600; display:inline-flex; transform-origin:left center; animation:ccPop .42s cubic-bezier(.22,1.2,.36,1);">✓ Preparing download…</span>';
      setTimeout(() => { sub.innerHTML = old; b._busy = 0; close(); }, 950);
    }));
  }
  plParamDefs(type) {
    if (type === 'sem') return [
      { key: 'anchors', group: 'Anchor', label: 'Anchor papers', num: true, chips: [5, 10, 25], def: 10,
        hint: 'How many of the current papers anchor the semantic search.' },
      { key: 'mode', group: 'Fetch', label: 'Query mode', seg: ['Combined', 'Per paper'], def: 'Combined',
        desc: { Combined: 'One recommendation query over all anchors together.', 'Per paper': 'A separate query per anchor paper; results are merged and de-duplicated.' } },
      { key: 'fetch', label: 'Recommendations to fetch', num: true, chips: [50, 100, 250], def: 100,
        hint: 'Papers pulled in before the filter pipeline below shapes what is kept.' }
    ];
    if (type === 'db') return [
      { key: 'topic', label: 'Research topic', ta: true, def: '', ph: 'e.g. Bounded gaps between primes',
        hint: 'The query agent writes database queries from this topic. Leave blank to use the project topic.' },
      { key: 'model', group: 'Query agent', label: 'Model', dd: true, def: 'Default from Settings (gemini-3.1-flash-lite)',
        hint: 'The LLM that writes and refines the database queries.' },
      { key: 'effort', label: 'Reasoning effort', seg: ['Low', 'Medium', 'High'], def: 'Low',
        desc: { Low: 'Fastest and cheapest; fine for well-scoped topics.', Medium: 'More query reformulation on harder topics.', High: 'Most reformulation and coverage; slowest and most expensive.' } },
      { key: 'budget', label: 'Search budget', num: true, chips: [5, 10, 20], def: 10, unit: 'iterations',
        hint: 'How many query / fetch / refine iterations the agent may run before it stops.' }
    ];
    if (type === 'rrk') return [
      { key: 'algo', group: 'Diversify', label: 'Clustering algorithm', seg: ['Community', 'Topic modeling'], def: 'Community',
        desc: { 'Community': 'Clusters papers by their citation relationships.', 'Topic modeling': 'Clusters papers by content semantics.' } },
      { key: 'gran', label: 'Cluster granularity', seg: ['High', 'Medium', 'Low'], def: 'Medium',
        desc: { High: 'Many small, tightly-focused clusters.', Medium: 'A balance of cluster count and size.', Low: 'A few large, broad clusters.' } },
      { key: 'rank', group: 'Rerank', label: 'Rank papers by', seg: ['Citation count', 'PageRank'], def: 'Citation count',
        desc: { 'Citation count': 'Raw citations: favors widely-cited work.', 'PageRank': 'Influence in the citation graph: favors structurally central work.' } },
      { key: 'keep', group: 'Select', label: 'Papers to pass on', num: true, chips: [100, 200, 500], def: 200,
        hint: 'Top-ranked papers, drawn across all clusters.' }
    ];
    return null;
  }
  plParamsOf(st) {
    const host = st;
    if (!host.params) {
      host.params = {};
      (this.plParamDefs(st.type) || []).forEach(d => { host.params[d.key] = d.def; });
    }
    return host.params;
  }
  plRenderParams(st) {
    const root = this.rootRef.current, box = root.querySelector('.cfg-params');
    if (!box) return;
    const defs = this.plParamDefs(st.type);
    if (!defs) { box.style.display = 'none'; return; }
    box.style.display = '';
    const body = box.querySelector('.cfg-params-body');
    body.innerHTML = '';
    const P = this.plParamsOf(st);
    let gn = 0;
    defs.forEach(d => {
      if (d.group) {
        gn++;
        const g = document.createElement('div'); g.className = 'cfg-pgroup';
        g.innerHTML = '<span class="g-n">' + gn + '</span><span class="g-t">' + d.group + '</span>';
        body.appendChild(g);
      }
      const f = document.createElement('div'); f.className = 'fc-field';
      f.innerHTML = '<span class="fc-t">' + d.label + '</span>';
      if (d.seg) {
        const seg = document.createElement('div'); seg.className = 'fc-seg'; seg.dataset.variant = 'Segmented';
        const desc = document.createElement('span'); desc.className = 'fc-d'; desc.style.margin = '7px 0 0';
        d.seg.forEach(opt => {
          const b = document.createElement('button'); b.type = 'button'; b.textContent = opt;
          if (P[d.key] === opt) b.dataset.on = '1';
          b.addEventListener('click', () => {
            P[d.key] = opt;
            seg.querySelectorAll('button').forEach(x => { x.dataset.on = x === b ? '1' : '0'; });
            if (d.desc) desc.textContent = d.desc[opt] || '';
          });
          seg.appendChild(b);
        });
        f.appendChild(seg);
        if (d.desc) { desc.textContent = d.desc[P[d.key]] || ''; f.appendChild(desc); }
      } else if (d.num) {
        const inp = document.createElement('input'); inp.type = 'number'; inp.min = '1'; inp.value = P[d.key];
        inp.addEventListener('input', () => { const v = parseInt(inp.value, 10); if (v > 0) P[d.key] = v; });
        f.appendChild(inp);
        const chips = document.createElement('div'); chips.className = 'fc-chips'; chips.style.marginTop = '8px';
        (d.chips || []).forEach(n => {
          const b = document.createElement('button'); b.type = 'button'; b.textContent = n + ' ' + (d.unit || 'papers');
          b.addEventListener('click', () => { inp.value = n; P[d.key] = n; });
          chips.appendChild(b);
        });
        f.appendChild(chips);
        if (d.hint) { const hn = document.createElement('div'); hn.className = 'fc-hint'; hn.style.marginTop = '8px'; hn.textContent = d.hint; f.appendChild(hn); }
      } else if (d.dd) {
        const dd = document.createElement('div'); dd.className = 'fc-dd fc-dd-model';
        const ddBtn = document.createElement('button'); ddBtn.type = 'button'; ddBtn.className = 'fc-dd-btn';
        const ddMenu = document.createElement('div'); ddMenu.className = 'fc-dd-menu';
        const setBtn = () => { ddBtn.innerHTML = '<span>' + P[d.key] + '</span><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"></path></svg>'; };
        const paintOpts = () => ddMenu.querySelectorAll('.fc-dd-opt').forEach(o => o.dataset.on = o.dataset.val === P[d.key] ? '1' : '0');
        const addOpt = (nm, sub, val) => { const o = document.createElement('button'); o.type = 'button'; o.className = 'fc-dd-opt fc-dd-opt2'; o.dataset.val = val; o.innerHTML = '<span class="fc-dd-txt"><span class="fc-dd-nm">' + nm + '</span>' + (sub ? '<span class="fc-dd-sub">' + sub + '</span>' : '') + '</span>'; o.addEventListener('click', e => { e.stopPropagation(); P[d.key] = val; setBtn(); ddMenu.style.display = 'none'; paintOpts(); }); ddMenu.appendChild(o); };
        addOpt('Default', 'gemini-3.1-flash-lite', 'Default from Settings (gemini-3.1-flash-lite)');
        this.llmModels().forEach(grp => { const g = document.createElement('div'); g.className = 'fc-dd-grp'; g.textContent = grp.g; ddMenu.appendChild(g); grp.items.forEach(it => addOpt(it[0], it[1], it[0])); });
        setBtn();
        ddBtn.addEventListener('click', e => { e.stopPropagation(); const open = ddMenu.style.display === 'block'; ddMenu.style.display = open ? 'none' : 'block'; if (!open) { paintOpts(); const closer = ev => { if (dd.contains(ev.target)) return; ddMenu.style.display = 'none'; document.removeEventListener('click', closer); }; setTimeout(() => document.addEventListener('click', closer), 0); } });
        dd.appendChild(ddBtn); dd.appendChild(ddMenu); f.appendChild(dd);
        if (d.hint) { const hn = document.createElement('div'); hn.className = 'fc-hint'; hn.style.marginTop = '8px'; hn.textContent = d.hint; f.appendChild(hn); }
      } else if (d.ta) {
        const ta = document.createElement('textarea'); ta.className = 'fc-ta'; ta.rows = 2;
        if (d.ph) ta.placeholder = d.ph;
        ta.value = P[d.key] || '';
        const fit = () => { ta.style.height = 'auto'; ta.style.height = Math.max(58, ta.scrollHeight) + 'px'; };
        ta.addEventListener('input', () => { P[d.key] = ta.value; fit(); });
        f.appendChild(ta);
        requestAnimationFrame(fit);
        if (d.hint) { const hn = document.createElement('div'); hn.className = 'fc-hint'; hn.style.marginTop = '8px'; hn.textContent = d.hint; f.appendChild(hn); }
      }
      body.appendChild(f);
    });
  }
  plNew(type, extra) {
    const o = { id: 'n' + (++this._plSid), type: type, filters: 0 };
    if (extra) for (const k in extra) o[k] = extra[k];
    return o;
  }
  plInit() {
    this._plSid = 0;
    const seed = this.plNew('source');
    const a = this.plNew('fwd', { filters: 5 }), b = this.plNew('bwd', { filters: 5 });
    const c = this.plNew('fwd', { filters: 5 }), d = this.plNew('bwd', { filters: 5 });
    const e = this.plNew('bwd', { filters: 5 }), f = this.plNew('fwd', { filters: 5 });
    const par = (branches) => ({ id: 'p' + (++this._plSid), type: 'parallel', branches: branches });
    this.pipe = [seed, par([[a], [b]]), par([[c], [d]]), par([[e], [f]])];
    this._plSel = a.id;
    this._plHist = [];
  }
  plSteps(seq) {
    const out = [];
    const walk = (s) => { (s || []).forEach(el => { if (el.type === 'parallel') (el.branches || []).forEach(walk); else out.push(el); }); };
    walk(seq || this.pipe);
    return out;
  }
  plIndexMap() {
    const map = {}; let k = 0;
    const walk = (s) => { (s || []).forEach(el => { if (el.type === 'parallel') (el.branches || []).forEach(walk); else map[el.id] = k++; }); };
    walk(this.pipe);
    return map;
  }
  plFind(id) { return this.plSteps().find(st => st.id === id) || null; }
  plCode(id, idx, M) {
    const st = this.plFind(id);
    if (!st || idx[st.id] == null) return '';
    return M[st.type].prefix + '-' + String(idx[st.id] + 1).padStart(2, '0');
  }
  plCount() { return this.plSteps().length; }
  // locate an element (step or parallel) — returns { seq, i, el, par }
  plLocate(id, seq, par) {
    seq = seq || this.pipe;
    for (let i = 0; i < seq.length; i++) {
      const el = seq[i];
      if (el.id === id) return { seq: seq, i: i, el: el, par: par || null };
      if (el.type === 'parallel') {
        for (const b of el.branches) { const r = this.plLocate(id, b, el); if (r) return r; }
      }
    }
    return null;
  }
  plCols(seq) { let m = 1; (seq || this.pipe).forEach(el => { m = Math.max(m, this.plElCols(el)); }); return m; }
  plElCols(el) {
    if (el.type !== 'parallel') return 1;
    return el.branches.reduce((n, b) => n + this.plCols(b), 0);
  }
  plCanAddBranch(parId) {
    const loc = this.plLocate(parId);
    if (!loc || loc.el.branches.length >= 2) return false;
    return this.plTry(p => { const l = this.plLocate(parId, p); l.el.branches.push([this.plNew('fwd')]); }, true);
  }
  plCanFanOut(id) {
    const loc = this.plLocate(id);
    if (!loc || loc.par) return false; // max 2 parallel branches: no fan-out from inside a parallel group
    return this.plTry(p => this.plApplyFanOut(p, id, this.plNew('fwd')), true);
  }
  // clone → mutate → column check. commit unless probeOnly.
  plTry(fn, probeOnly) {
    const before = JSON.stringify(this.pipe);
    const draft = JSON.parse(before);
    const keep = this.pipe;
    this.pipe = draft;
    let ok = true;
    try { fn(draft); } catch (e) { ok = false; }
    if (ok) ok = this.plCols(draft) <= 3;
    if (!ok || probeOnly) { this.pipe = keep; return ok; }
    this.pipe = keep;
    this._plHist.push({ pipe: before, sel: this._plSel });
    if (this._plHist.length > 40) this._plHist.shift();
    this.pipe = draft;
    return true;
  }
  plApplyFanOut(p, id, step) {
    const loc = this.plLocate(id, p);
    if (!loc) throw new Error('missing');
    if (loc.par) { loc.par.branches.push([step]); return; }
    loc.seq[loc.i] = { id: 'p' + (++this._plSid), type: 'parallel', branches: [[loc.el], [step]] };
  }
  plPrune(seq) {
    const out = [];
    for (const el of seq) {
      if (el.type !== 'parallel') { out.push(el); continue; }
      const branches = el.branches.map(b => this.plPrune(b)).filter(b => b.length);
      if (!branches.length) continue;
      if (branches.length === 1) { branches[0].forEach(x => out.push(x)); continue; }
      el.branches = branches;
      out.push(el);
    }
    return out;
  }
  // ================= pipeline: render =================
  plH(h) { const t = document.createElement('template'); t.innerHTML = h.trim(); return t.content.firstElementChild; }
  plEdge() {
    return this.plH('<svg class="pl-wire" width="14" height="26" viewBox="0 0 14 26" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"><path d="M7 0 V25"></path>' + this.plCap(7, 25) + '</svg>');
  }
  pl6Plus(id, aria) { return '<button class="v6d-plus" data-id="' + id + '" aria-label="' + (aria || 'Insert a step here') + '"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M12 5v14M5 12h14"></path></svg></button>'; }
  pl6Card(st, idx, M) {
    const m = M[st.type], no = String(idx[st.id] + 1).padStart(2, '0'), code = m.prefix + '-' + no;
    const t6 = st.type === 'source' ? 'sed' : st.type === 'db' ? 'dbs' : st.type;
    let cnt = '';
    if (st.type === 'source') { const sn = this._seeds ? this._seeds.length : 7; cnt = '<span class="v6d-cnt"><span class="v6d-out">' + sn + (sn === 1 ? ' paper' : ' papers') + '</span></span>'; }
    else if (st.type !== 'rrk') { const f = st.filters || 0; cnt = '<span class="v6d-cnt"><span class="v6d-out">' + f + (f === 1 ? ' filter' : ' filters') + '</span></span>'; }
    const link = '';
    const del = '<span class="v6d-tools"><button class="v6d-tool" data-act="del" aria-label="Delete this step"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 7h14M10 7V5h4v2M8 7l1 12h6l1-12"></path></svg></button></span>';
    const fan = (st.type !== 'source' && this.plCanFanOut(st.id)) ? '<span class="v6d-side"></span><button class="v6d-fan" data-id="' + st.id + '" aria-label="Run a step alongside this one"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M7 4v8a4 4 0 0 0 4 4h6"></path><path d="M14 13l3 3-3 3"></path></svg></button>' : '';
    const node = this.plH('<div class="v6d-node" data-kind="' + m.kind + '" data-type="' + t6 + '" data-id="' + st.id + '" data-title="' + m.name + '" data-code="' + code + '">'
      + '<span class="v6d-sh" aria-hidden="true"><i class="v6d-fa"></i></span>'
      + '<span class="v6d-ord">' + no + '</span>'
      + '<div class="v6d-l1"><span class="v6d-ico">' + this.plIconSvg(st.type, 13) + '</span><span class="v6d-name">' + m.name + '</span></div>'
      + '<div class="v6d-rule"><i></i>' + cnt + '</div>'
      + '<div class="v6d-meta"><span class="v6d-kind">' + m.label + '</span><span class="v6d-code">' + code + '</span><span class="v6d-sp"></span>' + link + del + '</div>'
      + fan + '</div>');
    if (st.id === this._plSel) { node.classList.add('is-sel'); node.setAttribute('data-sel', '1'); }
    return node;
  }
  pl6Par(el, idx, M, isLast) {
    /* no diamond at the open end: it would assert a junction that doesn't
       exist yet: the dashed Merge-branches pill is the one affordance. */
    const merge = isLast
      ? ''
      : '<button class="v6d-junc v6d-junc--merge" data-id="' + el.id + '" aria-label="Insert a step after this group"><i class="v6d-jdot"></i><span class="v6d-jplus"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M12 5v14M5 12h14"></path></svg></span></button>';
    const unit = this.plH('<div class="v6d-group-unit" data-id="' + el.id + '" data-last="' + (isLast ? '1' : '') + '">'
      + '<div class="v6d-group is-open" data-id="' + el.id + '">'
      + '<span class="v6d-fork"><i class="v6d-stem"></i><i class="v6d-bus"></i><i class="v6d-arrow v6d-arrow--l"></i><i class="v6d-arrow v6d-arrow--r"></i></span>'
      + '<span class="v6d-merge"><i class="v6d-bus"></i><i class="v6d-stem"></i></span>'
      + merge
      + '<div class="v6d-branches"></div></div></div>');
    const bs = unit.querySelector('.v6d-branches');
    el.branches.forEach(b => {
      const bd = this.plH('<div class="v6d-branch"></div>');
      this.pl6Seq(bd, b, idx, M, false);
      const lastStep = b[b.length - 1];
      bd.appendChild(this.plH('<div class="v6d-tail">' + this.pl6Plus(lastStep ? lastStep.id : '') + '</div>'));
      bs.appendChild(bd);
    });
    return unit;
  }
  pl6Tail(seq, isRoot) {
    const frag = document.createElement('div'); frag.style.display = 'contents';
    const last = seq[seq.length - 1], merging = !!last && last.type === 'parallel';
    if (last) frag.appendChild(this.plH('<div class="v6d-wire v6d-wire--tail' + (merging ? ' v6d-wire--tailm' : '') + '"></div>'));
    const icon = merging
      ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="18" r="3"></circle><circle cx="6" cy="6" r="3"></circle><path d="M6 21V9a9 9 0 0 0 9 9"></path></svg>'
      : '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M12 5v14M5 12h14"></path></svg>';
    frag.appendChild(this.plH('<button class="v6d-tailadd" data-after="' + (last ? last.id : '') + '" data-merge="' + (merging ? '1' : '') + '">' + icon + (merging ? 'Merge branches' : 'Add step') + '</button>'));
    if (isRoot) { frag.appendChild(this.plH('<div class="v6d-wire v6d-wire--end"></div>')); frag.appendChild(this.plH('<span class="v6d-term"></span>')); }
    return frag;
  }
  pl6Seq(host, seq, idx, M, isRoot) {
    seq.forEach((el, i) => {
      const prev = seq[i - 1];
      if (i > 0 && prev.type !== 'parallel' && el.type !== 'parallel') host.appendChild(this.plH('<div class="v6d-wire">' + this.pl6Plus(prev.id) + '</div>'));
      if (el.type === 'parallel') {
        if (prev) host.appendChild(prev.type !== 'parallel'
          ? this.plH('<div class="v6d-wire v6d-wire--pre">' + this.pl6Plus(prev.id) + '</div>')
          : this.plH('<div class="v6d-wire v6d-wire--link"></div>'));
        host.appendChild(this.pl6Par(el, idx, M, i === seq.length - 1));
      } else {
        if (prev && prev.type === 'parallel') host.appendChild(this.plH('<div class="v6d-wire v6d-wire--post"></div>'));
        host.appendChild(this.pl6Card(el, idx, M));
      }
    });
    if (isRoot) host.appendChild(this.pl6Tail(seq, isRoot));
  }
  pl6Ink(flow) {
    if (!flow || !flow.offsetWidth) return;
    const cs = getComputedStyle(flow);
    const T = (n, f) => { const v = parseFloat(cs.getPropertyValue(n)); return isNaN(v) ? f : v; };
    const SW = T('--v6d-w', 1.5), PR = T('--v6d-port', 7) / 2, A = T('--v6d-ah', 7) * .7071, R = T('--v6d-r', 9);
    const sc = flow.getBoundingClientRect().width / flow.offsetWidth || 1;
    const bx = el => el.getBoundingClientRect();
    const rl = (el, ref) => { const r = bx(el), q = bx(ref); return { y: (r.top - q.top) / sc, b: (r.bottom - q.top) / sc, cx: (r.left + r.width / 2 - q.left) / sc }; };
    const N = v => Math.round(v * 100) / 100;
    const ink = (el, w, h, d, tips) => {
      let s = el.querySelector(':scope > svg.v6d-ink');
      if (!s) { s = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); s.setAttribute('class', 'v6d-ink'); s.setAttribute('aria-hidden', 'true'); el.appendChild(s); el.classList.add('is-inked'); }
      s.setAttribute('viewBox', '0 0 ' + N(w) + ' ' + N(h));
      s.innerHTML = '<path d="' + d + '" fill="none" stroke="currentColor" stroke-width="' + SW + '"/>'
        + (tips || []).map(p => '<path d="M' + N(p[0] - A) + ' ' + N(p[1] - A) + ' L' + N(p[0]) + ' ' + N(p[1]) + ' L' + N(p[0] + A) + ' ' + N(p[1] - A) + '" fill="none" stroke="currentColor" stroke-width="' + SW + '" stroke-linecap="round" stroke-linejoin="round"/>').join('');
    };
    /* TERMINAL CONTRACT: every shaft leaves its source at y=0 (the opaque
       outlet dot / merge diamond / frame edge sits over it) and arrives
       either FLUSH at an edge, or as the ONE arrow glyph whose V-vertex sits
       exactly PR above the wire end: on the inlet ring's rim. */
    flow.querySelectorAll('.v6d-wire').forEach(wq => {
      const r = bx(wq), w = r.width / sc, h = r.height / sc;
      const toPort = !/--(pre|link|tail|end)/.test(wq.className);
      const y1 = toPort ? h - PR : h;
      ink(wq, w, h, 'M' + N(w / 2) + ' 0 V' + N(y1), toPort ? [[w / 2, y1]] : []);
    });
    flow.querySelectorAll('.v6d-group').forEach(gp => {
      const fork = gp.querySelector(':scope > .v6d-fork'), mg = gp.querySelector(':scope > .v6d-merge');
      const branches = gp.querySelector(':scope > .v6d-branches');
      if (!fork || !mg || !branches) return;
      const cols = [...branches.children].filter(b => b.classList.contains('v6d-branch'));
      if (cols.length) {
        const fr = bx(fork), fw = fr.width / sc, fh = fr.height / sc, busY = 30;
        const d = [], tips = [], xs = [];
        d.push('M' + N(fw / 2) + ' ' + N(rl(gp, fork).y) + ' V' + busY);
        cols.forEach(b => {
          const lx = rl(b, fork).cx; xs.push(lx);
          const first = b.firstElementChild;
          const arr = !!(first && first.classList.contains('v6d-node'));
          const endY = arr ? rl(first, fork).y - PR : rl(b, fork).y;
          const left = lx < fw / 2;
          d.push('M' + N(lx) + ' ' + N(endY) + ' V' + N(busY + R) + ' A' + R + ' ' + R + ' 0 0 ' + (left ? 1 : 0) + ' ' + N(lx + (left ? R : -R)) + ' ' + busY);
          if (arr) tips.push([lx, endY]);
        });
        xs.sort((a, b) => a - b);
        d.push('M' + N(xs[0] + R) + ' ' + busY + ' H' + N(xs[xs.length - 1] - R));
        ink(fork, fw, fh, d.join(' '), tips);
        const mr = bx(mg), mw = mr.width / sc, mh = mr.height / sc, mbusY = 14;
        const dm = [], mxs = [];
        const bb = rl(branches, mg).b, gb = rl(gp, mg).b;
        cols.forEach(b => {
          const lx = rl(b, mg).cx; mxs.push(lx);
          const left = lx < mw / 2;
          dm.push('M' + N(lx) + ' ' + N(bb) + ' V' + N(mbusY - R) + ' A' + R + ' ' + R + ' 0 0 ' + (left ? 0 : 1) + ' ' + N(lx + (left ? R : -R)) + ' ' + mbusY);
        });
        mxs.sort((a, b) => a - b);
        dm.push('M' + N(mxs[0] + R) + ' ' + mbusY + ' H' + N(mxs[mxs.length - 1] - R));
        dm.push('M' + N(mw / 2) + ' ' + mbusY + ' V' + N(gb));
        ink(mg, mw, mh, dm.join(' '), []);
      }
      gp.querySelectorAll(':scope > .v6d-branches > .v6d-branch > .v6d-tail').forEach(tl => {
        const r = bx(tl), w = r.width / sc, h = r.height / sc;
        ink(tl, w, h, 'M' + N(w / 2) + ' 0 V' + N(h), []);
      });
    });
  }
  plRender() {
    const root = this.rootRef.current; if (!root) return;
    const flow = root.querySelector('.pl-flow'); if (!flow) return;
    const M = this.plMeta(), idx = this.plIndexMap();
    if (!this.plFind(this._plSel)) { const first = this.plSteps()[0]; this._plSel = first ? first.id : null; }
    flow.innerHTML = '';
    if (root.dataset.pstyle === 'Flow chart (6d)') {
      this.pl6Seq(flow, this.pipe, idx, M, true);
      this.pl6Ink(flow);
      if (!this._plRO) this._plRO = new ResizeObserver(es => { const f = es[0] && es[0].target; if (f && f.isConnected && f.querySelector('.v6d-ink')) this.pl6Ink(f); });
      this._plRO.disconnect(); this._plRO.observe(flow);
    }
    else { if (this._plRO) this._plRO.disconnect(); this.plSeqInto(flow, this.pipe, idx, M, true); this.plWires(); }
    const sc = root.querySelector('.pl-scroll');
    if (sc && !this._plCentred) { this._plCentred = true; requestAnimationFrame(() => { sc.scrollLeft = Math.max(0, (sc.scrollWidth - sc.clientWidth) / 2); }); }
    this.plSyncHeader();
    this.plSyncConfig();
  }
  plSeqInto(host, seq, idx, M, isRoot) {
    seq.forEach((el, i) => {
      const prev = seq[i - 1];
      if (i > 0 && prev.type !== 'parallel' && el.type !== 'parallel') host.appendChild(this.plEdge());
      host.appendChild(el.type === 'parallel' ? this.plParEl(el, idx, M, i === seq.length - 1) : this.plCardEl(el, idx, M));
    });
    const last = seq[seq.length - 1];
    if (isRoot || (last && last.type === 'parallel')) host.appendChild(this.plTailEl(seq, isRoot));
  }
  plCardEl(st, idx, M) {
    const m = M[st.type], no = String(idx[st.id] + 1).padStart(2, '0'), code = m.prefix + '-' + no;
    let chip = '';
    if (st.type === 'source') { const sn = this._seeds ? this._seeds.length : 7; chip = '<span class="pb-chip">' + sn + (sn === 1 ? ' paper' : ' papers') + '</span>'; }
    else if (st.filters) chip = '<span class="pb-chip">' + st.filters + ' filters</span>';
    const link = '';
    const addS = '<button class="pb-add pb-add-s" data-id="' + st.id + '" aria-label="Add a step after this one"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 5v14M5 12h14"></path></svg></button>';
    const addP = (st.type !== 'source' && this.plCanFanOut(st.id))
      ? '<button class="pb-add pb-add-p" data-id="' + st.id + '" aria-label="Run a step alongside this one"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 5v14M5 12h14"></path></svg></button>' : '';
    const card = this.plH('<div class="pb-card" data-kind="' + m.kind + '" data-id="' + st.id + '" data-title="' + m.name + '" data-code="' + code + '">'
      + link
      + '<div class="pb-head"><span class="pb-ico">' + this.plIconSvg(st.type, 13) + '</span><span class="pb-kind">' + m.label + '</span><span class="pb-no">' + no + '</span></div>'
      + '<div class="pb-title">' + m.name + '</div>'
      + '<div class="pb-div"><span></span></div>'
      + '<div class="pb-foot"><span class="pb-code">' + code + '</span>' + chip + '</div>'
      + addS + addP + '</div>');
    if (st.id === this._plSel) card.setAttribute('data-sel', '1');
    return card;
  }
  plParEl(el, idx, M, isLast) {
    const unit = this.plH('<div class="pl-par" data-id="' + el.id + '" data-last="' + (isLast ? '1' : '') + '">'
      + '<div class="pl-fork"></div>'
      + '<div class="pl-par-row"></div>'
      + '<div class="pl-join"><span class="pl-joinsvg"></span>'
      + (isLast ? '' : '<button class="pl-join-add" data-id="' + el.id + '" data-tip="Add a step here"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 5v14M5 12h14"></path></svg></button>')
      + '</div></div>');
    const row = unit.querySelector('.pl-par-row');
    const box = this.plH('<div class="pl-group"><span class="pl-tag">Parallel</span><div class="pl-branches"></div></div>');
    const bs = box.querySelector('.pl-branches');
    el.branches.forEach(b => {
      const bd = this.plH('<div class="pl-branch"></div>');
      this.plSeqInto(bd, b, idx, M, false);
      bs.appendChild(bd);
    });
    row.appendChild(box);
    const inside = this.plSteps([el]).some(x => x.id === this._plSel);
    if (inside) { unit.setAttribute('data-has-sel', '1'); box.setAttribute('data-has-sel', '1'); }
    return unit;
  }
  plTailEl(seq, isRoot) {
    const wrap = this.plH('<div class="pl-tail"></div>');
    const last = seq[seq.length - 1];
    const merging = !!last && last.type === 'parallel';
    if (last && !merging) wrap.appendChild(this.plEdge());
    const icon = merging
      ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="18" r="3"></circle><circle cx="6" cy="6" r="3"></circle><path d="M6 21V9a9 9 0 0 0 9 9"></path></svg>'
      : '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 5v14M5 12h14"></path></svg>';
    wrap.appendChild(this.plH('<button class="pl-tailadd" data-after="' + (last ? last.id : '') + '" data-merge="' + (merging ? '1' : '') + '">' + icon
      + (merging ? 'Merge branches' : 'Add step') + '</button>'));
    if (isRoot) wrap.appendChild(this.plH('<svg class="pl-wire" width="14" height="26" viewBox="0 0 14 26" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" fill="none"><path d="M7 2 V19"></path><circle cx="7" cy="22" r="2.5" fill="currentColor" stroke="none"></circle></svg>'));
    return wrap;
  }
  plWires() {
    const root = this.rootRef.current; if (!root) return;
    if (root.dataset.pstyle === 'Flow chart (6d)') return;
    root.querySelectorAll('.pl-par').forEach(u => {
      const ur = u.getBoundingClientRect();
      const scale = ur.width && u.offsetWidth ? ur.width / u.offsetWidth : 1;
      const branches = [...u.querySelectorAll(':scope > .pl-par-row > .pl-group > .pl-branches > .pl-branch')];
      const centers = branches.map(b => {
        const c = b.querySelector('.pb-card');
        const r = (c || b).getBoundingClientRect();
        return Math.round((r.left - ur.left + r.width / 2) / scale);
      });
      if (!centers.length) return;
      const fork = u.querySelector(':scope > .pl-fork'), join = u.querySelector(':scope > .pl-join');
      fork.style.height = ''; join.style.height = '';
      fork.innerHTML = this.plWireSvg(u.offsetWidth, centers, 'fork', false);
      const tip = u.dataset.last === '1';
      join.querySelector(':scope > .pl-joinsvg').innerHTML = this.plWireSvg(u.offsetWidth, centers, 'join', tip);
      const btn = join.querySelector(':scope > .pl-join-add');
      if (btn) btn.style.top = '17px';
    });
  }
  plCap(x, y) {
    return '<path d="M ' + (x - 4.5) + ' ' + (y - 5.2) + ' L ' + x + ' ' + y + ' L ' + (x + 4.5) + ' ' + (y - 5.2) + '"></path>';
  }
  plWireSvg(w, centers, dir, tip) {
    const H = (dir === 'join' && !tip) ? 30 : 34, A = w / 2, R = 4;
    const lo = Math.min(A, ...centers), hi = Math.max(A, ...centers);
    const p = [], caps = [];
    if (dir === 'fork') {
      const busY = 13, base = 33;
      p.push('M ' + A + ' 0 V ' + busY);
      if (centers.length > 1) p.push('M ' + lo + ' ' + (busY + R) + ' Q ' + lo + ' ' + busY + ' ' + (lo + R) + ' ' + busY + ' H ' + (hi - R) + ' Q ' + hi + ' ' + busY + ' ' + hi + ' ' + (busY + R));
      centers.forEach(c => {
        const y0 = (centers.length > 1 && (c === lo || c === hi)) ? busY + R : busY;
        p.push('M ' + c + ' ' + y0 + ' V ' + base);
        caps.push(this.plCap(c, base));
      });
    } else {
      const busY = 17, base = 33;
      centers.forEach(c => {
        const y1 = (centers.length > 1 && (c === lo || c === hi)) ? busY - R : busY;
        p.push('M ' + c + ' 0 V ' + y1);
      });
      if (centers.length > 1) p.push('M ' + lo + ' ' + (busY - R) + ' Q ' + lo + ' ' + busY + ' ' + (lo + R) + ' ' + busY + ' H ' + (hi - R) + ' Q ' + hi + ' ' + busY + ' ' + hi + ' ' + (busY - R));
      p.push('M ' + A + ' ' + busY + ' V ' + (tip ? base : H));
      if (tip) caps.push(this.plCap(A, base));
    }
    return '<svg class="pl-wire" width="' + w + '" height="' + H + '" viewBox="0 0 ' + w + ' ' + H + '" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none">'
      + p.map(d => '<path d="' + d + '"></path>').join('') + caps.join('') + '</svg>';
  }
  plSyncHeader() {
    const root = this.rootRef.current; if (!root) return;
    const c = root.querySelector('.plh-count');
    const n = this.plCount();
    if (c) c.textContent = n + (n === 1 ? ' step' : ' steps');
    const u = root.querySelector('.panel-canvas [aria-label="Undo last pipeline change"]');
    if (u) { u.disabled = !this._plHist.length; u.style.opacity = this._plHist.length ? '1' : '.4'; u.style.cursor = this._plHist.length ? 'pointer' : 'default'; }
  }
  plSyncConfig() {
    const root = this.rootRef.current; if (!root) return;
    const M = this.plMeta(), idx = this.plIndexMap(), st = this.plFind(this._plSel);
    const T = root.querySelector('.cfgh-title'), C = root.querySelector('.cfgh-code'),
          D = root.querySelector('.cfgh-desc');
    if (!st) return;
    if (this._cfgSelWas !== st.id) {
      this._cfgSelWas = st.id;
      const scb = root.querySelector('.panel-config .cfg-scroll');
      if (scb && scb.animate && this._cfgSelInit) scb.animate([{ opacity: 0, transform: 'translateY(7px)' }, { opacity: 1, transform: 'none' }], { duration: 260, easing: 'cubic-bezier(.22,.61,.36,1)' });
      this._cfgSelInit = true;
    }
    const m = M[st.type];
    if (T) T.innerHTML = '<span style="display:inline-flex; width:16px; height:16px; vertical-align:-1.5px; margin-right:8px; color:var(--v-' + m.dot + '-dot);">' + this.plIconSvg(st.type, 16) + '</span>' + m.name;
    if (C) C.textContent = m.prefix + '-' + String(idx[st.id] + 1).padStart(2, '0');
    if (D) D.textContent = m.desc;
    this.plRenderParams(st);
    const isSrc = st.type === 'source';
    const delBtn = root.querySelector('.panel-config [aria-label="Delete step"]');
    if (delBtn) delBtn.style.display = isSrc ? 'none' : '';
    const seedBox = root.querySelector('.cfg-seed'), ph = root.querySelector('.cfg-pipe-head'), pw = root.querySelector('.cfg-pipe');
    root.setAttribute('data-src', isSrc ? '1' : '0');
    if (seedBox) seedBox.style.display = isSrc ? '' : 'none';
    if (ph) ph.style.display = isSrc ? 'none' : '';
    const tl = root.querySelector('.cfg-fp-tools');
    if (tl) tl.style.display = isSrc ? 'none' : '';
    if (pw) pw.style.display = isSrc ? 'none' : '';
    if (isSrc) this.seedRender();
  }
  // ================= pipeline: interaction =================
  scrollToTop(el) {
    let sc = el.parentElement;
    while (sc && sc !== document.body) {
      const ov = getComputedStyle(sc).overflowY;
      if ((ov === 'auto' || ov === 'scroll') && sc.scrollHeight > sc.clientHeight + 1) break;
      sc = sc.parentElement;
    }
    if (!sc || sc === document.body || sc.scrollTop < 2) return;
    const from = sc.scrollTop, t0 = performance.now(), DUR = 420;
    const step = now => {
      const p = Math.min((now - t0) / DUR, 1);
      sc.scrollTop = from * (1 - (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  // tween the enclosing scrollarea just far enough that an opened popover is fully
  // visible (menus near the panel bottom otherwise open into the clipped region)
  revealPop(menu) {
    let sc = menu.parentElement;
    while (sc && sc !== document.body) {
      const ov = getComputedStyle(sc).overflowY;
      if ((ov === 'auto' || ov === 'scroll') && sc.scrollHeight > sc.clientHeight + 1) break;
      sc = sc.parentElement;
    }
    if (!sc || sc === document.body) return;
    // offset-chain geometry: immune to popIn()'s in-flight transform and to preview scale
    const absY = el => { let y = 0; while (el) { y += el.offsetTop; el = el.offsetParent; } return y; };
    const menuBottom = absY(menu) - absY(sc) + menu.offsetHeight;
    const over = menuBottom - (sc.scrollTop + sc.clientHeight) + 10;
    if (over <= 1) return;
    const from = sc.scrollTop;
    const target = Math.min(Math.max(0, sc.scrollHeight - sc.clientHeight), from + over);
    if (target <= from) return;
    const t0 = performance.now(), DUR = 380;
    const step = now => {
      const p = Math.min((now - t0) / DUR, 1);
      sc.scrollTop = from + (target - from) * (1 - Math.pow(1 - p, 3));
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  centerScroll(el, done) {
    let sc = el.parentElement;
    while (sc && sc !== document.body) {
      const ov = getComputedStyle(sc).overflowY;
      if ((ov === 'auto' || ov === 'scroll') && sc.scrollHeight > sc.clientHeight + 1) break;
      sc = sc.parentElement;
    }
    if (!sc || sc === document.body) { if (done) done(); return; }
    // target-tracking tween: re-measure the live target every frame so late
    // reflow (list still growing) can never freeze us on a stale target
    const liveTarget = () => {
      const er = el.getBoundingClientRect(), sr = sc.getBoundingClientRect();
      const max = Math.max(0, sc.scrollHeight - sc.clientHeight);
      return Math.max(0, Math.min(max, sc.scrollTop + (er.top - sr.top) - (sc.clientHeight - er.height) / 2));
    };
    const from = sc.scrollTop, t0 = performance.now(), DUR = 620, HOLD = 500;
    sc._centering = (sc._centering || 0) + 1;
    const step = now => {
      const t = now - t0, target = liveTarget();
      if (t < DUR) {
        const p = t / DUR, e = 1 - Math.pow(1 - p, 3);
        sc.scrollTop = from + (target - from) * e;
      } else {
        const d = target - sc.scrollTop;
        if (Math.abs(d) > 0.5) sc.scrollTop += d * 0.25;
      }
      if (t < DUR + HOLD) { requestAnimationFrame(step); return; }
      sc._centering--;
      if (done) done();
    };
    requestAnimationFrame(step);
  }
  popIn(el) { if (el && el.animate) el.animate([{ opacity: 0, transform: 'translateY(-5px) scale(.97)' }, { opacity: 1, transform: 'none' }], { duration: 180, easing: 'cubic-bezier(.22,1.1,.36,1)' }); }
  collapseOut(el, done) {
    if (!el || !el.animate) { if (done) done(); return; }
    if (el._closing) return; el._closing = true;
    const cs = getComputedStyle(el);
    el.style.overflow = 'hidden'; el.style.pointerEvents = 'none';
    el.animate([
      { height: el.offsetHeight + 'px', opacity: 1, transform: 'none', paddingTop: cs.paddingTop, paddingBottom: cs.paddingBottom, borderTopWidth: cs.borderTopWidth, borderBottomWidth: cs.borderBottomWidth },
      { height: '0px', opacity: 0, transform: 'translateX(-12px)', paddingTop: '0px', paddingBottom: '0px', borderTopWidth: '0px', borderBottomWidth: '0px' }
    ], { duration: 260, easing: 'cubic-bezier(.4,0,.2,1)' }).onfinish = () => { if (done) done(); };
  }
  expandIn(el) {
    if (!el || !el.animate) return;
    const h = el.offsetHeight;
    el.style.overflow = 'hidden';
    el.animate([{ height: '0px', opacity: 0, transform: 'translateX(-8px)' }, { height: h + 'px', opacity: 1, transform: 'none' }], { duration: 280, easing: 'cubic-bezier(.22,.61,.36,1)' }).onfinish = () => { el.style.overflow = ''; };
  }
  plMenu(anchor, mode, id) {
    const root = this.rootRef.current;
    const canvas = root.querySelector('.pl-scroll'), menu = root.querySelector('.pl-addmenu');
    if (!menu) return;
    const M = this.plMeta(), idx = this.plIndexMap();
    const titles = { after: 'Add step after', along: 'Run alongside', tail: 'Add step', merge: 'Merge branches into' };
    // only hand-configured originals are reusable — a copy of a copy would snowball the list
    const reusable = this.plSteps().filter(x => ['fwd', 'bwd', 'db', 'sem'].includes(x.type) && !x.reused);
    const dotFor = t => 'var(--v-' + M[t].dot + '-dot)';
    let h = '<span>' + titles[mode] + '</span>';
    ['fwd', 'bwd', 'db', 'sem', 'rrk'].forEach(t => {
      h += '<button data-add="' + t + '"><span style="width:13px; height:13px; flex:none; display:flex; align-items:center; color:' + dotFor(t) + '">' + this.plIconSvg(t, 13) + '</span>' + M[t].name + '</button>';
    });
    if (reusable.length) {
      h += '<span>Copy an existing searcher</span>';
      reusable.forEach(x => {
        const code = this.plCode(x.id, idx, M);
        h += '<button data-reuse="' + x.id + '"><span style="width:13px; height:13px; flex:none; display:flex; align-items:center; color:' + dotFor(x.type) + '">' + this.plIconSvg(x.type, 13) + '</span>'
          + '<span class="m-txt">' + M[x.type].name + '<span class="m-sub">' + code + '&nbsp;&nbsp;copy its settings</span></span></button>';
      });
    }
    menu.innerHTML = h;
    menu._mode = mode; menu._id = id;
    const r = anchor.getBoundingClientRect(), cr = canvas.getBoundingClientRect(), mw = 214;
    menu.style.display = 'block';
    let left = r.left - cr.left + canvas.scrollLeft + r.width / 2 - mw / 2;
    left = Math.max(8, Math.min(left, canvas.scrollWidth - mw - 8));
    menu.style.left = left + 'px';
    menu.style.top = (r.bottom - cr.top + canvas.scrollTop + 8) + 'px';
    this.popIn(menu);
  }
  plHideMenu() { const root = this.rootRef.current; const m = root && root.querySelector('.pl-addmenu'); if (m) m.style.display = 'none'; }
  plMakeStep(pick) {
    if (pick.reuse) {
      const origin = this.plFind(pick.reuse);
      const step = this.plNew(origin.type, { filters: origin.filters });
      step.reused = 1;
      if (origin.params) { try { step.params = JSON.parse(JSON.stringify(origin.params)); } catch (e) {} }
      return step;
    }
    return this.plNew(pick.add);
  }
  plAdd(mode, id, pick) {
    const step = this.plMakeStep(pick);
    let ok = true;
    if (mode === 'along') ok = this.plTry(p => this.plApplyFanOut(p, id, step));
    else if (mode === 'tail' || mode === 'merge') ok = this.plTry(p => {
      if (!id) { p.push(step); return; }
      const l = this.plLocate(id, p);
      l.seq.splice(l.i + 1, 0, step);
    });
    else ok = this.plTry(p => { const l = this.plLocate(id, p); l.seq.splice(l.i + 1, 0, step); });
    if (ok) { this._plSel = step.id; this.plRender(); }
  }
  plDelete(id) {
    const st = this.plFind(id);
    if (!st || st.type === 'source') return;
    this.plDeleteNow(id);
  }
  plDeleteNow(id) {
    this.plTry(p => {
      const strip = (seq) => {
        const out = [];
        for (const el of seq) {
          if (el.id === id) continue;
          if (el.type === 'parallel') el.branches = el.branches.map(strip);
          out.push(el);
        }
        return out;
      };
      const next = this.plPrune(strip(p));
      p.length = 0; next.forEach(x => p.push(x));
    });
    this._plSel = null;
    this.plRender();
  }
  plDuplicate(id) {
    const st = this.plFind(id);
    if (!st || st.type === 'source') return;
    const copy = this.plNew(st.type, { filters: st.filters });
    if (st.params) { try { copy.params = JSON.parse(JSON.stringify(st.params)); } catch (e) {} }
    if (this.plTry(p => { const l = this.plLocate(id, p); l.seq.splice(l.i + 1, 0, copy); })) {
      this._plSel = copy.id;
      this.plRender();
    }
  }
  plUndo() {
    const h = this._plHist.pop();
    if (!h) return;
    this.pipe = JSON.parse(h.pipe);
    this._plSel = h.sel;
    this.plRender();
  }
  setupPipeline(root) {
    this.plInit();
    const canvas = root.querySelector('.pl-scroll');
    if (!canvas) return;
    this.plDragWire(canvas);
    new MutationObserver(() => { this._plCentred = false; this.plRender(); }).observe(root, { attributes: true, attributeFilter: ['data-pstyle'] });
    canvas.addEventListener('click', e => {
      if (this._plDragged) return;
      const menu = root.querySelector('.pl-addmenu');
      const item = e.target.closest('.pl-addmenu button');
      if (item) {
        e.stopPropagation();
        const pick = item.dataset.add ? { add: item.dataset.add } : { reuse: item.dataset.reuse };
        const mode = menu._mode, id = menu._id;
        this.plHideMenu();
        this.plAdd(mode, id, pick);
        return;
      }
      if (e.target.closest('.pl-addmenu')) return;
      const addS = e.target.closest('.pb-add-s');
      if (addS) { e.stopPropagation(); this.plMenu(addS, 'after', addS.dataset.id); return; }
      const addJ = e.target.closest('.pl-join-add');
      if (addJ) { e.stopPropagation(); this.plMenu(addJ, 'after', addJ.dataset.id); return; }
      const addP = e.target.closest('.pb-add-p');
      if (addP) { e.stopPropagation(); this.plMenu(addP, 'along', addP.dataset.id); return; }
      const tail = e.target.closest('.pl-tailadd');
      if (tail) { e.stopPropagation(); this.plMenu(tail, tail.dataset.merge ? 'merge' : 'tail', tail.dataset.after || null); return; }
      const d6 = e.target.closest('.v6d-tool[data-act="del"]');
      if (d6) { e.stopPropagation(); const c = d6.closest('.v6d-node'); if (c) this.plDelete(c.dataset.id); return; }
      const p6 = e.target.closest('.v6d-plus[data-id]');
      if (p6) { e.stopPropagation(); this.plMenu(p6, 'after', p6.dataset.id); return; }
      const f6 = e.target.closest('.v6d-fan[data-id]');
      if (f6) { e.stopPropagation(); this.plMenu(f6, 'along', f6.dataset.id); return; }
      const m6 = e.target.closest('.v6d-junc--merge[data-id]');
      if (m6) { e.stopPropagation(); this.plMenu(m6, 'after', m6.dataset.id); return; }
      const t6 = e.target.closest('.v6d-tailadd');
      if (t6) { e.stopPropagation(); this.plMenu(t6, t6.dataset.merge ? 'merge' : 'tail', t6.dataset.after || null); return; }
      this.plHideMenu();
      const card = e.target.closest('.pb-card, .v6d-node');
      if (card) {
        if (this._cfgRow) this.cancelConfig();
        this._plSel = card.dataset.id;
        this.plRender();
      }
    });
    document.addEventListener('click', e => { if (!e.target.closest('.pl-scroll')) this.plHideMenu(); });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') this.plHideMenu();
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      if ((e.key === 'Delete' || e.key === 'Backspace') && this._plSel) { e.preventDefault(); this.plDelete(this._plSel); }
      if ((e.key === 'z' || e.key === 'Z') && (e.metaKey || e.ctrlKey) && this._plHist.length) { e.preventDefault(); this.plUndo(); }
    });
    const del = root.querySelector('.panel-config [aria-label="Delete step"]');
    if (del) del.addEventListener('click', () => this.plDelete(this._plSel));
    const first = root.querySelector('.pl-first');
    if (first) first.addEventListener('click', e => {
      const opt = e.target.closest('[data-first-preset]');
      const scratch = e.target.closest('.pl-first-scratch');
      if (!opt && !scratch) return;
      this._plFirstDone = true;
      root.setAttribute('data-first', '0');
      if (opt) { this.plPreset(opt.dataset.firstPreset); return; }
      if (this.plTry(p => { const src = p.find(x => x.type === 'source') || this.plNew('source'); p.length = 0; p.push(src); })) {
        this._plSel = null;
        this._plCentred = false;
        this.plRender();
        this.pcToast('canvas', 'Starting from the seed set&nbsp;&nbsp;add steps with the plus button', true);
      }
    });
    const pb = root.querySelector('.plh-preset'), pm = root.querySelector('.plh-preset-menu');
    if (pb && pm) {
      pb.addEventListener('click', e => { e.stopPropagation(); const open = pm.style.display === 'block'; pm.style.display = open ? 'none' : 'block'; pb.setAttribute('aria-expanded', open ? 'false' : 'true'); if (!open) this.popIn(pm); });
      document.addEventListener('click', () => { if (pm.style.display !== 'block') pb.setAttribute('aria-expanded', 'false'); });
      pm.addEventListener('click', e => {
        e.stopPropagation();
        const o = e.target.closest('.plh-pre-opt');
        if (!o) return;
        pm.style.display = 'none';
        const kind = o.dataset.preset;
        const NM = { scout: 'Scout', survey: 'Survey', dragnet: 'Dragnet' };
        const n = this.plCount();
        this.cfgConfirm(pb, 'Start over with the <b>' + NM[kind] + '</b> preset? The current pipeline of <b>' + n + (n === 1 ? ' step' : ' steps') + '</b> will be replaced. Undo can bring it back.', 'Use ' + NM[kind], () => this.plPreset(kind));
      });
      document.addEventListener('click', () => { pm.style.display = 'none'; });
    }
    const undo = root.querySelector('.panel-canvas [aria-label="Undo last pipeline change"]');
    if (undo) undo.addEventListener('click', () => this.plUndo());
    window.addEventListener('resize', () => this.plWires());
    this.plRender();
  }
  renderVals() { return { rootRef: this.rootRef, pipelineStyle: this.props.pipelineStyle ?? 'Flow chart (6d)', cardLayout: this.props.cardLayout ?? 'Inline index' }; }
}
/* ===== END VERBATIM TRANSPLANT ===== */
