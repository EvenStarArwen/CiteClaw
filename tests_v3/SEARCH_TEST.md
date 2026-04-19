# V3 ExpandBySearch 测试日志

10 个 scenario,每个 scenario 用三个模型跑:
- **Gemma**: `gemma-4-31b` (Modal 128K B200,`reasoning_effort: medium`)
- **Grok**: `grok-4-1-fast-reasoning` (xAI, `reasoning_effort: medium`)
- **OpenAI**: `gpt-5.4-nano` (`reasoning_effort: medium`)

配置:
- `max_iter=5`, `max_subtopics=5`
- 无 seed paper 共享给 LLM,无 topic_description 下发到 worker (V3 默认)
- 只测 multi-agent query 设计,不跑 filter / citation graph
- 每个 scenario 记录: supervisor 初始 dispatch 的 sub-topic 数 / 最终补充的数 / 总运行时长 / union 大小 / 噪声比例估计

每个 scenario 的 data 在 `tests_v3/data/<scenario_id>_<model>/`;配置在 `tests_v3/configs/`;quality report 在 `tests_v3/reports/`。

## Scenario 列表

| # | ID | Topic | Domain |
|---|---|---|---|
| 1 | `prime_editing` | CRISPR prime editing for genetic disease correction | Biology |
| 2 | `opv_flex` | Organic photovoltaic materials for flexible solar cells | Materials |
| 3 | `spatial_txomics` | Spatial transcriptomics methods for tissue profiling | Biology |
| 4 | `gw_ligo` | Gravitational wave astronomy with LIGO / Virgo | Astronomy |
| 5 | `n2_reduction` | Electrocatalyst design for nitrogen reduction to ammonia | Chemistry |
| 6 | `lnp_delivery` | Lipid nanoparticle design for nucleic acid delivery | Biomedicine |
| 7 | `photocatalysis` | Hydrogen production via photocatalytic water splitting | Chemistry |
| 8 | `hydride_sc` | Room-temperature hydride superconductors under pressure | Physics |
| 9 | `jwst_exo` | Exoplanet atmosphere characterization via JWST | Astronomy |
| 10 | `sei_battery` | Battery solid electrolyte interphase formation | Materials |

---

## Scenario 1 — `prime_editing`

主题:CRISPR prime editing for correcting pathogenic genetic variants。Scope 明确限定 therapeutic 应用,排除 base editing / knockout / RNAi / CRISPR screens。

### 总览

| Model | Supervisor 初始 | 追加 | Workers 总数 | Union (deduped) | 运行时长 |
|---|---|---|---|---|---|
| Gemma 4-31B | 3 | +1 (`pe_agriculture`) | **4** | 6,422 | ~40 min |
| Grok 4-1-fast-reasoning | 5 | 0 | **5** | 13,925 | ~35 min |
| gpt-5.4-nano | 5 | 0 | **5** | 42,741 | ~25 min |

配置 `max_subtopics=5`,两个强模型(Grok / OpenAI)**直接顶到上限**,完全没有体现 "default toward fewer" 的 prompt 指令;Gemma 初始选了合理的 3 个,但后续 `add_sub_topics` 漂出 scope (见下)。

### 按模型分析

#### Gemma 4-31B

**好的地方**:
- 初始 decomposition 非常 clean:`pe_engineering` / `pe_delivery_safety` / `pe_therapeutics`,正好是 topic 描述里的三个自然 facet。命名有意义(不是 `1/2/3`)。
- 诊断质量高 —— 能精准说出问题。例子:pe_agriculture iter-4 自己发现 `pPE` 裸 acronym 匹配了 "Personal Protective Equipment",`pest*` 匹配了 "pesticide"。自己在 top-100 检查环节抓住了噪声。

**不好的地方**:
- **`add_sub_topics` 漂出 scope**:Supervisor 看到 pe_therapeutics 完成后,自己加了 `pe_agriculture`。但 topic_description 明确说 "therapeutic application",并没有 agriculture 范畴。这是**典型的 LLM 忽略 NOT 边界、只 pattern-match anchor 概念 ("prime editing") 就扩展**。
- **Query syntax 诡异**:worker pe_delivery_safety 把 `"prime edit* LNP"~10` 这种 proximity 带通配符的表达写进 query。Lucene 正式语法不支持这样写,能返回结果只能说 S2 prone 地忍受了。
- **Worker pe_therapeutics iter-0 用 `prime AND (editing OR editor*)`**:翻译后变成 `+prime +(editing | editor*)`。`prime` 当词根去 match,会 match "prime time / prime target / prime care" 等非基因编辑语境。Gemma 自己在 iter-1 更正为 `"prime editing"` 完整短语。
- **Worker pe_delivery_safety 剧烈震荡**:663 → 409 → 201 → 666 → 88。每 iter 都在 proximity 和 flat-AND 之间反复跳,最终 88 篇过窄。
- **Top-cited 严重噪声**:"Burnout and Self-Reported Patient Care" (2138c)、"Social Comparison in Everyday Life" (704c)、"Obsessive-Compulsive Disorder" (286c) 这些心理学/医学综述进了 top-20。应该是 pe_therapeutics 里 `therapeutic OR therapy OR treatment OR disease*` 这种过宽 facet 导致的。

#### Grok 4-1-fast-reasoning

**好的地方**:
- 真正遵循了 prompt 里 "pearl growing" 的术语,诊断里明确提 "pearl grow with 'compact prime' OR 'improved prime'"。
- 主动识别 bare acronym 风险 —— 多个 worker 在 iter-0 后都指出了 `PE` 匹配 "pulmonary embolism / preeclampsia / polyethylene" 的问题,并在下一 iter 去除。
- Union size 适中 (13,925),cluster 0 占 1990/2000 = **99.5% on-topic**,cluster 分布比 OpenAI 干净得多。
- Top-cited 比 OpenAI 命中率高:Anzalone 2019 (3575c) ✓、Anzalone 2020 综述 ✓、TwinPE ✓、Engineered pegRNAs ✓、prime editing in rice ✓ 都在 top 15 里。

**不好的地方**:
- **明显的 Lucene 语法错误**:worker 4 iter-0 写出 `"prime AND editing"` —— 把 AND 放进了引号内,搜的是字面短语 "prime AND editing",这是 quote 里有 operator 的典型新手错误。翻译层没有对这种 case 做防御。
- **Proximity 修饰符滥用**:worker 3 iter-2 到 iter-4 里 `"prime editing"~3 OR "prime editor*"~3` —— 对单词/单短语用 `~N` 没意义,只会让 query 不必要地复杂。
- **冗余 AND facets**:worker 2 写了 `(pegRNA* OR epegRNA*) AND (pegRNA* OR epegRNA* OR ...)` —— 同一个 facet 放在 AND 两侧,纯属无效计算。
- **部分 worker 剧烈震荡**:worker 2 从 266 → 21 → 13 → 180 → 2。结尾只剩 2 篇,基本等于 worker 失败。
- Supervisor 的 done summary 直接承认 "specificity/off-targets (10504, noisy)"、"therapeutics (2930, noisy)" —— 模型知道有噪声但没能力修复,只是打个标签。

#### gpt-5.4-nano

**好的地方**:
- 没有 policy rejection(说明 prime editing topic 不敏感)。
- 速度最快(~25 min 全跑完)。
- Set_strategy 的 5 个 sub-topic 描述都是完整英文句子,结构一致。
- 极长的 OR 列表 —— worker 里常见 15+ 个同义词并列(virus / viral vectors / AAV / adeno-associated / LNP / lipid nanoparticle / ...),说明它在按 prompt 里 "maximally enumerate" 的指令做事。

**不好的地方**:
- **Union 42,741 —— 严重过 broad**。cluster 0 (1969p) on-topic,但 top-20 高引里混入 SARS-CoV-2 Cas12a 检测 (2152c、633c、258c、271c、302c 等),还有整套乳腺癌、卵巢癌、HCC 的综述 —— 这些完全不是 prime editing。
- **标志性 acronym 灾难**:worker 4 iter-0 同时放 `"PE" OR "RT"` —— "PE" (polyethylene / 物理教育 / 肺栓塞) 和 "RT" (real-time / radiation therapy / 反转录) 是信息检索里最经典的两个陷阱词。OpenAI 自己在 iter-1 承认了并移除,但 iter-0 就已经污染了 100K+ 篇 paper 进 cache。
- **Subsumption 违反**:`"prime editing" OR "prime-editing" OR "CRISPR prime editing"` 三个全上。前两个是同义变体,最后一个被前两个严格 subsume(前两个已经匹配了所有包含 "prime editing" 的 paper)。prompt Step 2 里明确写了 subsumption 规则,但模型没遵守。
- **Query 复杂度失控**:worker 4 iter-3 的 query 长到 query_tree 的 count-only probe 都失败了 (`probe failure`),诊断里写 "final AND clause (sequencing/mitigation methods) is marked as a probe failure"。
- **Total 剧烈震荡**:worker 1: 5709 → 1399 → 5 → 9 → 367;worker 4: 202464 → 124 → 223 → 264 → 1235。几乎每 iter 都走极端,说明 diagnose→plan→rewrite 环里的 "tighten / loosen" 建议没有量化基准,只在 "极宽" 和 "极窄" 之间摆动。

### 跨模型对比

**Recall (top-cited canonical 命中)**:
- Anzalone 2019 (Nature) —— 原型 prime editing paper:**三家都找到**。
- Anzalone 2020 综述 (Nat Rev Genet) —— 基础参考:**Grok + Gemma ✓,OpenAI ✗** (被非相关 SARS 综述和肿瘤论文挤出 top-cited)。
- Engineered pegRNAs (Nelson 2021):**Grok + Gemma ✓**,OpenAI 不在 top-20 但未必缺席 union。
- twinPE (Anzalone 2022):**三家都找到**。

**Precision / 噪声比例**(enriched 2000 上,UMAP+HDBSCAN on SPECTER2,`size_factor=0.5` —— 真实的 cluster 分布,不是早期 k-means 的 artifact)

- **Gemma**:cluster 0 (1707p prime editing 主流) + cluster 1 (260p plant editing,来自 `pe_agriculture` 漂移) + ~33 篇 noise。On-topic ≈ 85%(plant 算 off-topic),是三家最干净的。
- **Grok**:极度污染。cluster 0 **360p = preeclampsia** (PE 名缩碰撞);cluster 1 337p = 治疗蛋白递送(非 editing,PE 歧义);cluster 5 123p 随机噪声;加上 cluster 3 plant (269p,对 therapeutic scope 来说 off-topic)。真正 prime-editing 核心 cluster 2+4+6+8 合计 666/2000 = 33%。**噪声 >50%**。
- **OpenAI**:cluster 0 (1655p 主流 on-topic) + cluster 1 (219p **SARS-CoV-2 CRISPR 检测** —— `RT` acronym 拉进的);cluster 2 (113p plant)。Noise ≈ 17%,介于 Gemma 和 Grok 之间。

早期用 k-means + TF-IDF 的 "99% on-topic" 结论是 **聚类算法 artifact**,不是真实精度。换成 UMAP+HDBSCAN on SPECTER2 embedding (size_factor=0.5 halve ExpandBySearch-specific 的 min_cluster_size) 后,各家差异变得非常明显。Grok 的 bare-`PE` 问题从 top-cited 单篇看不出有多严重,cluster 聚类才暴露出 700+ 篇被 PE=preeclampsia / PE=drug-delivery 拉进来。

**Union 大小的诡异对比**
- OpenAI 42K >> Grok 14K >> Gemma 6.4K
- 但 cluster 0 占比三家都 >98% —— union 越大不意味着信息密度越高,OpenAI 的 28K 增量大多是噪声长尾。这说明**单纯看 union 大小并不是好的 recall 指标**。

### 共性问题(所有模型都犯的)

1. **Supervisor 倾向顶到 max_subtopics**。两个强模型直接选了 5 个,Gemma 选了 3 但然后 add。V3.1 prompt 里 "default toward fewer" 是软约束,对这三家都没有落地。—— 这是 V3 supervisor prompt 需要**架构级**改动的信号(不是 patch rule)。

2. **对 bare short acronym 的警惕**都不够。`PE`/`RT` 这种公开已知的多义词在 iter-0 query 里都出现了。Step 2 "maximally enumerate OR" 的指令和 Step 4 "leakage check" 之间的张力没解决 —— 模型 literal 按 Step 2 堆同义词,Step 4 没有 hard force。

3. **诊断→行动脱节**。三家模型都能在 diagnose 阶段精准说出问题("too broad because X"),但**next query 的改动经常与诊断不一致**。Gemma worker pe_delivery_safety 就是最好的例子:iter-0 诊断"generic terms 过宽",iter-1 改成 proximity,iter-2 又 tighten proximity,iter-3 又回到宽 AND,iter-4 再次 proximity —— 像 random walk。

4. **Iteration 后期 total 趋近 0 或 仍震荡**。max_iter=5 给了充足预算,但没有单调收敛的行为,说明 diagnose_plan 缺少 "距离理想值 X 篇还差多少" 的量化反馈。

### 可能的修改建议(给下一 scenario 观察后再定是否 apply)

**暂不改动 prompts**,但记下几个观察:

1. Supervisor 的 decomposition 决策需要**硬约束**而非软约束 —— 可能的方向:在 set_strategy 前先插入一个显式判断 turn("回答:这个 parent topic 是否有 shared anchor?"),把 anchor 的文字写出来,再据此决定 subtopic 数。现在的 prompt 只是 "think about this first",三家都跳过了这一步。
2. 翻译层 `to_lucene()` 应该**主动拒绝**引号内的 AND/OR (如 Grok 的 `"prime AND editing"`) 和无意义的 proximity (如 Grok 的 `"single_phrase"~N`),让模型看到 syntax error 而不是 silently 运行一个错意的 query。
3. Step 2 里 "maximally expand" 的指令可能需要对 short acronym(≤3 char)加一个硬例外 —— 即使 step 4 说去 filter,iter-0 就已经从 200K+ 论文里 sample 了 top-cited,先产生的污染就是用户最先看到的。

---

## Scenario 1 — `prime_editing` 重跑 (post-diagnostic-overhaul)

中间落地了一系列修复:V3.1 supervisor anchor-shared 判断、union → last-iter、translator 保护 quoted strings、per-OR-alternative 计数、pre-S2 syntax checker(8 个检查 + WORKER_SYNTAX_ERROR retry)、in-cluster query tree + forced per-item noise diagnosis、S2 auto-stem 的 prompt 更新(删 `word*` 推荐、加 <4 char acronym 硬例外)、wildcard-in-phrase 与 operator-in-phrase 两个 syntax 检查、LLM consecutive-failure tracker、topic_model scipy 错误 guard、in-cluster combo render 上限 40。

### 数量变化

| Model | 旧轮 初始 / 追加 / Union | 新轮 初始 / 追加 / Union | Workers 总 |
|---|---|---|---|
| Gemma | 3 + 1 = 4 workers, union **6,422** | **3 + 0 = 3 workers, union 463** | 3 |
| Grok | 5 + 0 = 5 workers, union **13,925** | **1 + 0 = 1 worker, union 1,076** | 1 |
| OpenAI | 5 + 0 = 5 workers, union **42,741** | **1 + 0 = 1 worker, union 49** | 1 |

Supervisor 行为彻底变了:Grok 和 OpenAI 都从 5 顶格降到 1(anchor-shared 判断真正落地);Gemma 初始仍选 3 但没像上轮那样多加 `pe_agriculture` 漂出 scope。

### 聚类干净度(UMAP+HDBSCAN on SPECTER2, size_factor=0.5)

**Gemma** — 18 clusters on 463 enriched papers, **接近 100% on-topic**:
18 个 clusters 全部是 prime editing 细分方向 —— retinal gene editing、liver in-vivo、cardiac RBM20/PLN、CD34+/HSPCs、CFTR 囊性纤维、DMD exon、KRAS、SCID 基因治疗、eVLP 递送、Alzheimer APOE4 等。**零 PE=preeclampsia / PPE / plantar / property accounting 类的 collision**。上轮 41% 噪声 → 现在 ~0%。

**Grok** — 14 clusters on 1,076 enriched papers, **~75% on-topic**:
cluster 1 (251p, plant editing) 仍有残留 —— plant prime editing 在 topic 的 therapeutic scope 之外但确实是 prime editing 的应用。其余 13 个 cluster 全部 on-topic(prime editing / pegRNA design / gene therapy / CFTR / DMD / retinal / cardiovascular / Alzheimer / KRAS 等)。**零 PE=preeclampsia / phosphatidylethanolamine / pulmonary embolism 类的 collision**。上轮 >50% 噪声 → 现在 ~25%(剩余 drift 不是 acronym 碰撞而是 scope 问题)。

**OpenAI** — 16 clusters on 49 enriched papers, **100% on-topic**:
每个 cluster 2-8 篇,全部是 prime editing 细分:SBDS、SMA exon skipping、F508del CFTR、ELANE neutrophil editing、ABCA4 BRET、OptiPrime 机器学习、PEDAR deletion、template-jumping PE 等。**零 `RT`=reverse transcriptase / SARS-CoV-2 / 癌症综述 类的 collision**。上轮 17% 噪声 → 现在 0%,但 recall 偏低(49 vs Gemma 463)。

### Top-cited 对比

新的三家 top-cited 都是货真价实的 prime editing 论文:Anzalone 2019 原型("Search-and-replace genome editing"),DMD/CFTR/IRD 的 prime editing 应用,Enhanced pegRNA,PASSIGE,engineered pegRNA,split PE with dual-AAV,twin prime,mouse liver/heart in vivo 等。上轮那种 SARS-CoV-2 Cas12、burnout 社科综述、preeclampsia、肺栓塞之类的"高引噪声"彻底消失。

### syntax_check 触发统计

| code | Gemma | Grok | OpenAI |
|---|---|---|---|
| `or_subsumed`(warning) | 3 | 2 | **21** |
| `bare_short_acronym`(warning) | 4 | 0 | 0 |
| `wildcard_in_phrase`(error) | 0 | 0 | 0 |
| `operator_word_in_phrase`(warning) | 0 | 0 | 0 |

OpenAI 每 iter 都堆 `"prime editing" OR "prime-editing" OR "prime editing system"` 这种 subsumption 违反,检查每次都报,但属于 warning,没阻塞。Gemma 的 4 次 `bare_short_acronym` 都是在 therapeutic facet 里放了 `rat`(3 字母)—— 警告触发符合规则。由于这轮三家都没写 bare `PE` / `RT` / `PASTE`、也没写 quoted-phrase-with-wildcard 或 AND-inside-quote,新加的两个 error 检查没机会触发,但**模型行为本身已经规避了这些陷阱** —— 说明 prompt 更新生效。

### 修复了 SEARCH_TEST 里记的哪些问题

| 上轮记录的问题 | 状态 |
|---|---|
| PE / RT / PASTE / PPE 裸缩写 collision | ✓ 三家都消失 |
| Supervisor 顶到 max_subtopics | ✓ Grok & OpenAI 从 5 → 1;Gemma 3 未追加 |
| Gemma pe_agriculture scope drift | ✓ 这次没有 |
| Grok `"prime AND editing"`(AND 在引号内) | ✓ 这次没写 |
| Grok 冗余 AND facet (同一 OR 组两侧) | ✓ 这次没有 |
| 迭代剧烈震荡(5→500K→9→50K) | 🟢 这次单调收敛(Gemma 2413→521→255→204→105;OpenAI 750→83→23→40→49)|
| cluster 0 k-means 伪 99% 干净 | ✓ 换 UMAP+HDBSCAN 后真实 cluster 分布 |
| Union 作为 recall 指标误导 | ✓ 现在返回 last-iter 而非 union |
| 诊断 → 行动脱节 | 🟢 可见单调收敛,间接证据 |

### 仍存问题(observation,非 blocker)

1. **OpenAI recall 偏低(49 papers)**。worker 在 iter 1 后 query 过窄(750→83),到 iter 4 只有 49 篇。max_iter=5 给了 5 次机会但 agent 倾向越收越紧。没有"你离理想召回 X 篇还差多少"的量化反馈;这正是上轮 "共性 #4" 留的口子,user 指示暂不加(不同领域没有通用值)。
2. **Grok worker 1 iter 2 propose 返回空 query**,worker 只完成 2 iter 就止步(iter 0 的 142K 被 iter 1 的 1076 覆盖,最终就是 iter 1)。不是我们的 bug,是模型响应异常;新加的 consecutive-failure tracker 此时不会触发(tracker 只看空响应,iter 2 propose 返回了空但 iter 0-1 的中间 LLM 调用都成功,consec 计数被重置)。
3. **Grok plant editing drift (251p/1076)**。plant prime editing 是合法的 prime editing 应用,但 topic 明确 scope 在"therapeutic".这不是 acronym collision,是 agent 对 topic 描述里 "NOT" 边界的 pattern-match 不够严;属于 supervisor/worker 共同的 scope interpretation 问题,单靠 syntax check 抓不到。
4. **OpenAI subsumption 违反 21 次**(warning-only)。`or_subsumed` 作为 warning 不阻塞,OpenAI 稳定忽略。如果希望真正压这个,需要把它升级成 error,或者由 translator 层静默删除冗余 alternative。

### 小结

上轮三家一眼望去都是灾难(PE 拉进 preeclampsia、RT 拉进 SARS-CoV-2 RT、PPE 拉进 Personal Protective Equipment、OpenAI union 42K 里大半是无关论文),本轮三家的 last-iter 输出都是**纯粹的 prime editing 相关 paper**,差别只剩 recall(OpenAI 49 < Gemma 463 < Grok 1076)和一个 scope-drift 残余(Grok plant editing)。Diagnostic tools + 两轮 prompt 更新 + syntax check + anchor-shared supervisor 叠加起来的综合效果显著。

---
