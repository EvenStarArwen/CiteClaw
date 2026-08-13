# 复盘日志增强提案（B7）

**状态：提案，未实施。** 本文档不改任何代码。产品负责人在下方"签字栏"逐条勾选后，
才会派 agent 动工。所有 engine 改动都属于**后续批次**。

**要回答的两个问题**（产品负责人原话）：

- **Q-A**：「为什么这个 run 没搜到某篇文章？」
- **Q-B**：「为什么搜出这么多无关的？」

本提案的判据只有一条：**拿着磁盘上的 run 目录（不重跑、不连网、不看内存），
能不能把这两个问题答到"哪一步、哪个 filter、什么参数"的粒度。**

今天的答案是**不能**。下面先说清楚差在哪，再给 9 条补充建议，每条都标了
「落在哪 / 多大 / 复盘时怎么用」。

---

## 0. 先说结论

| | |
|---|---|
| 现在磁盘上有什么 | 一份**结果**（收了谁、丢了谁、总账），没有**过程** |
| 缺的最要命的一样 | **阶段归属**——每篇论文是在**哪一步的哪个 filter** 被杀的，磁盘上完全没有 |
| 最划算的一条 | **R4 formula 子查询判定**：数据已经算出来了，代码里当场丢掉，加回来 0 额外成本 |
| 全套加起来多大 | 小 run（runs/data，486 拒绝）**+250 KB**；大 run（demo RUN-37 规模，17,440 拒绝）**+1.9 MB**，其中 90% 是 R3 的 rejections 扩容 |
| 参照系 | 同一个 run 目录里 `cache.db` 已经 **28.7 MB**、`citeclaw.log` 已经 **2.0 MB**。本提案全套 ≤ 现有目录的 **6%** |
| 顺手的收获 | R2/R7/R8 正好补上新 UI `stats.json` 契约里 recon 标为"缺"的 per-step `found/kept/rej` + `{calls,graph,reco,tokens}` + 时间戳 |

⚠️ **两条安全/事实问题已单独升级到 `escalations.md`（E-LOG-01 / E-LOG-02），
其中 E-LOG-01 是密钥泄漏，建议优先于本提案处理。**

---

## 1. 现状审计：一个 run 跑完，磁盘上到底留下了什么

实测样本：`runs/data/`（21 篇 accepted / 486 rejected，唯一完整的真实 CLI run）、
`runs/webui/e8cd1f0f4a72/` 和 `cf7278602b6d/`（web/live 的 run）、
`runs/web/20260716_145857_c65b6a/`（旧 web/backend，已死）。

### 1.1 文件清单与写入者

| 文件 | 写入者 | 里面有什么 | 实测体积 |
|---|---|---|---|
| `literature_collection.json` | `steps/finalize.py:262` | 接受的论文全量元数据 + `summary.budget` 总账 + `depth/source/verdict` 三个分布 | ~4.9 KB/篇 |
| `literature_collection.bib` | `finalize.py:270` | BibTeX | 小 |
| `citation_network.graphml` | `finalize.py:151-154` | 图 + 边级 `contexts/intents/is_influential` | 中 |
| `collaboration_network.graphml` | `finalize.py:181-186` | 作者图（Q8 已永久砍掉，但仍在写） | 中 |
| `run_state.json` | `finalize.py:275` | `collection_ids / rejected_ids / seen_ids / queue_ids` 四个 **ID 列表** + `budget`（按 category / model / s2 by_type 拆得相当细） | 50 KB |
| `rejections.json` | `finalize.py:25-70` | `{paper_id: {categories:[...], title, year}}` | **228 B/条** |
| `shape_summary.{txt,json}` | `pipeline.py:449-465` | 每步 `in / out / delta_collection / stats{}` | ~1 KB |
| `cache.db` | `cache.py` | 9 张表；含 `search_queries(query_json, result_json)` 和 `llm_response_cache` | 28.7 MB |
| `citeclaw.log` | `logging_config` / `run_manager.py:1054` | DEBUG 全量文本 | 2.0 MB |
| `pipeline_config.json` | **仅 web/live**：`run_manager.py:48-61` | `Settings.model_dump()`，部分字段脱敏 | 5–10 KB |

**注意：`pipeline_config.json` 只有 web/live 这条路会写。CLI 跑的 run
（也就是 `runs/data/`、`runs/test_data/` 这些真实语料）磁盘上完全没有配置记录**——
连 `screening_model` 都只能从 graphml 的 `citeclaw_screening_model` 属性里抠。

### 1.2 拿现有文件试着回答 Q-A / Q-B

**Q-A「为什么没搜到 X」** —— 现状能走到第 2 步就断：

1. X 在 `run_state.json.seen_ids` 里吗？→ ✅ 能查。不在 = 从没被枚举到；在 = 被筛掉了。
2. 如果被筛掉了：在 `rejections.json` 里查到 `{"categories": ["citation"]}` → ❌ **到此为止**。
   哪一步？（第 2 步 ExpandForward 还是第 6 步 ExpandBackward？）哪个 filter 实例？
   （`cit_base` beta=30 还是 Route 分支里的 beta=60？）具体因为什么？（引用数 3，阈值要 11？）
   **三个都答不了。**
3. 如果从没被枚举到：❌ **完全答不了**。是 query 没匹配上？是它的 anchor 论文有 5,348 个
   citer 而 `max_citations: 30` 只取了引用数最高的 30 篇？是那个 anchor 压根没被 Rerank 选中？
   磁盘上没有任何一个字节能区分这三种。

**Q-B「为什么这么多无关的」** —— 现状只能看到总账：

- `rejection_counts` 聚合（`citation: 241, year: 106, llm_title_llm: 60, similarity: 49,
  llm_abstract_llm: 25, not_not_pure_app.inner: 5`）→ 能看出哪个桶量大。
- ❌ 看不出**哪一层在放水**：某个 filter 全程 0 拒绝（= 形同虚设）在聚合里和"没配"长得一样。
- ❌ 看不出 formula 里是哪一条在放水：`llm_title_llm` 这个桶把
  `(q_ml | q_stats) & !q_survey` 的三条子查询**压成了一个名字**。
  而同一个 run 的 `run_state.json.budget.by_category` 里明明有
  `llm_title_llm::q_ai`、`::q_survey`、`::q_benchmark` 的分账——
  **token 账本记了子查询，判定结果没记。**

### 1.3 已经算出来、但代码里当场丢掉的数据（"甜品"的来源）

这是本提案的核心发现：**下面每一项都已经在内存里，写盘只是加一行。**

| 数据 | 在哪算出来的 | 丢在哪 |
|---|---|---|
| 每篇拒绝的**人话原因**（`reason` 字符串，不只是桶名） | `filters/runner.py:302-314` 写进 `ctx.rejection_details` | 只给 live API 用，**从不落盘**；`_write_rejections_json` 读的是另一个只有桶名的 `rejection_ledger` |
| formula **每条子查询**的 per-paper 判定 | `screening/llm_runner.py:257-280` `sub_verdicts[qname][paper_id]` | 算完取 Boolean 合并值，`sub_verdicts` 直接出作用域 |
| ExpandBySearch 发出的**完整 query**（自然语言 + S2 语法 + S2 报的 total + 本轮新增 ID） | `agents/iterative_search.py:321-325` `AgentTurn` | 返回给调用方后，只有 `last_query[:200]` 进了 stats（`expand_by_search.py:133`），其余丢弃 |
| ExpandForward 每个源论文的 **citer 总数 / 截断后数 / 去重后新数** | `expand_forward.py:66-84`（`citers`、`citers[:max]`、`unseen`） | stats 只吐 `{"accepted": N}`（`expand_forward.py:154-157`） |
| 每步耗时 | `progress.py:336, 408` Dashboard 算了用来显示 | 只画在终端上，不落盘 |
| 每步的 budget 增量 | `BudgetTracker` 全程累加（`budget.py:373 to_dict`） | 只有 run 级总数落盘，没有 per-step 差分 |
| 种子标题解析到了**哪一篇** | `steps/resolve_seeds.py:132, 206` `search_match` 的返回 | stats 只吐 5 个聚合计数（`resolve_seeds.py:268-273`），换成了哪篇论文不记 |

---

## 2. 提案清单

每条给四样东西：**是什么 / 挂在哪 / 多大 / 复盘时怎么用**。
优先级 **P0 = 产品负责人点名必做**，**P1 = 强烈建议**，**P2 = 锦上添花**。

---

### R1 — 「这个 run 到底跑的是什么」：配置快照 `run_manifest.json`  · P0（点名）

**是什么。** 把一个 run 的完整身份卡片落盘，CLI 和 web 两条路都写。三部分：

1. **完整配置**：`Settings.model_dump(mode="json")` 脱敏后的全量（含 `blocks` / `pipeline`
   原样 YAML 结构、`seed_papers`、三个预算上限、`llm_batch_size/concurrency`、
   `structured_output_enabled`、镜像开关等）。
2. **展平的 filter 生效表**（新增，关键）：把 block 图展平成一张叶子清单，每行是
   *这个 filter 实例实际生效的参数*：
   ```
   [{ "path": "pipeline[2].screener.layers[1]",
      "step_idx": 2, "step": "ExpandBackward",
      "type": "CitationFilter", "name": "cit_base",
      "params": {"beta": 30},
      "resolved_model": null, "resolved_effort": null },
    { "path": "pipeline[2].screener.layers[3].queries.q_survey",
      "type": "LLMFilter::sub", "name": "topic_llm::q_survey",
      "params": {"scope": "title_abstract", "prompt": "…"},
      "resolved_model": "grok-4-1-fast-non-reasoning", "resolved_effort": "" }]
   ```
   YAML 里的 `blocks` 是带别名和 Route 分支的**图**——同一个 block 可能在三个步骤里被复用，
   per-filter 的 `model:` / `reasoning_effort:` 覆盖还会重走一遍 routing
   （`screening/llm_runner.py:75-97`）。**能拿来 diff 两个 run 的是展平后的清单，不是 YAML。**
3. **运行环境**：`started_at / ended_at / status / stop_reason`、engine git sha、
   `citeclaw.__version__`、Python 版本、S2 走的是官方 API 还是自建 mirror。

**挂在哪。** `pipeline.run_pipeline` 开头（`pipeline.py:359-385`，`_ensure_*` 注入之后、
step 循环之前——注入过的步骤必须体现在快照里），写 `cfg.data_dir/run_manifest.json`。
脱敏逻辑复用 `web/live/backend/run_manager.py:31-45 _scrub_secrets`，**但要先修 E-LOG-01**，
并且应该下沉进 engine，让 CLI run 也有。展平表复用 `pipeline._block_contains_similarity_filter`
（`pipeline.py:86-120`）已有的那套 block 递归下降写法。

**多大。** 实测五个真实配置的 `model_dump` JSON：
`config.yaml` 5.3 KB / `config_bio.yaml` 5.3 KB / `config_seg.yaml` 5.8 KB /
`config_sdl_mimo.yaml` 9.0 KB / `config_rna.yaml` 10.3 KB（54 个种子）。
展平表按 ~200 B × ~20 个叶子 = 4 KB。环境块 < 1 KB。
**合计 ~11–16 KB/run。**

**复盘时怎么用。** 这是所有复盘的第 0 步。今天用户拿到一个三周前的 CLI run 目录，
连"当时 `beta` 设的几"都得靠猜。有了这个：
- Q-B 的第一诊断动作 = 把两个 run 的展平表 diff 一下，看是哪个参数变了。
- 和 R2 的 PRISMA 表 join，直接得到「`cit_base` 在 beta=30 下，第 4 步砍掉 1,526 篇」。
- 是 UI「Edit filter in Build」跳转（`Runs.dc.html` step drill 的
  `rdDef` filter 定义弹层）唯一可能的数据源。

---

### R2 — 「每一步进去多少、出来多少、谁砍的」：PRISMA 逐阶段计数 `prisma.json`  · P0（点名）

**是什么。** 一个两层的漏斗表。

*run 级（PRISMA 主图的四个框）*：
```
identified        跨所有检索步骤枚举到的候选总数（去重前）
deduplicated      去掉 ctx.seen 重复后的唯一候选数
screened          真正送进 filter 的数量
excluded          按 filter 分桶的排除数（= 现有 rejection_counts，但带阶段）
included          最终 collection 大小
```

*step 级（真正解决 Q-B 的那层）*：每个步骤一行，行内是**顺序的阶段列表**：
```json
{ "step_idx": 4, "step": "ExpandBackward", "step_key": "ExpandBackward#2",
  "found": 2767,            // 检索原始返回（截断前）
  "truncated_to": 2767,     // max_citations 截断后（见 R9）
  "novel": 2701,            // ctx.seen 去重后
  "screened": 2701,
  "kept": 102,
  "stages": [
    {"name":"year_layer","type":"YearFilter","in":2701,"out":2035,"cut":666},
    {"name":"cit_base","type":"CitationFilter","in":2035,"out":456,"cut":1579},
    {"name":"kw","type":"AbstractKeywordFilter","in":456,"out":221,"cut":235},
    {"name":"topic_llm","type":"LLMFilter","in":221,"out":110,"cut":111,
     "sub":[{"q":"q_ml","cut":12},{"q":"q_survey","cut":99}]}   // ← 来自 R4
  ]}
```

**挂在哪。** 两个钩子，都很浅：

1. `pipeline.run_pipeline` 的 step 循环（`pipeline.py:387-419`）在 `step.run()` 前后设置
   `ctx.current_step = (idx, name)`。**这是整个提案的地基**——今天 filter 层完全不知道自己
   跑在哪一步，R3 也依赖它。同名步骤要带序号（`ExpandBackward#2`），因为一个 pipeline
   里同一个步骤会出现好几次（demo RUN-37 就是 3×FWD + 3×BWD）。
2. `filters/runner.py:69-98 _apply_sequential` 的 for 循环里已经有
   `in_count = len(passed)` 和每层跑完的 `passed`——把
   `(layer_name, type, in, out)` append 到 `ctx.prisma_ledger[step_key]` 即可。
   `_apply_any` / `_apply_route` / `_apply_not` 同理（各自已有 passed/rejected）。
3. `found`（截断前原始数）要 expand 步骤补吐：`expand_forward.py:76-84` 的
   `citers` / `citers[:max]` / `unseen` 三个数**已经在手上**，只是
   `stats={"accepted": N}` 没带（`expand_forward.py:154-157`）。
   ExpandBy* 家族的 `_expand_helpers.py:249-261` 已经吐了
   `{raw_hits, hydrated, novel, accepted, rejected}`——**这条已经对了，照抄给 Forward/Backward 就行。**
4. `Finalize` 写 `prisma.json`（`finalize.py` 现有 write 序列旁边加一个）。

**多大。** 纯整数。7 步 × ~8 阶段 × ~60 B ≈ **4 KB**。
就算 40 步的大 pipeline 也 < 25 KB。**基本是免费的。**

**复盘时怎么用。**
- **Q-B 的正面答案**：哪一层在放水，一眼看出来——`cut: 0` 的 filter 就是形同虚设的。
  今天在聚合的 `rejection_counts` 里，"配了但没砍到" 和 "根本没配" 长得一模一样。
- 「第 6 步 BWD 找回 5,223 篇只留了 62 篇」——比值异常本身就是信号
  （anchor 选错了 / 那批 citer 是个综述的引用列表）。
- **顺手补上 UI 契约**：demo 的 step drill（`Runs.dc.html` `rnStepDone` / `rdSquares`，
  数据在 `run-mock.js` 的 `PIPELINE`）要的正是每步
  `{in, found, kept, rej, cuts[6]}` + 阶段行
  「`Fetch reference papers` → `Enrich metadata` → `Enrich abstracts` →
  `Basic screening` (in/passed) → `LLM title screening` → `LLM abstract screening`」。
  recon §2.5 把 per-step `found/kept/rej` 列为**后端缺口**——R2 就是那个缺口的填法。

  ⚠️ **一处需要 design 决策**：demo 的 `cuts` 是**定长 6 桶**
  `[year, citation, keyword, LLM-title, LLM-abstract, duplicate]`，
  而真实 pipeline 的 filter 是任意组合、任意条数、可嵌套 Route/Any/Not。
  建议后端吐**有序不定长的 stages 列表**（如上），由前端折叠到它的 6 桶。
  这条要进 design 批次（已记入 `design-lane-worklist` 的建议见 §5）。

---

### R3 — 「这篇到底是谁砍的」：拒绝原因带阶段归属（扩 `rejections.json`）· P1（强烈建议）

**是什么。** 今天一条拒绝长这样（实测 `runs/data/rejections.json`）：
```json
"795dc87b…": {"categories": ["year"], "title": "UniRef clusters: …", "year": 2014}
```
建议改成：
```json
"795dc87b…": {
  "cut_at": {"step_idx": 4, "step": "ExpandBackward#2",
             "filter": "year_layer", "type": "YearFilter",
             "category": "year", "reason": "year 2014 < min 2018"},
  "also_cut_by": ["citation"],          // 其他分支/其他步骤也拒过（Parallel 会有）
  "source": "backward", "cites": 4210,
  "title": "UniRef clusters: …", "year": 2014
}
```

**挂在哪。** 三处，都很浅：
- `reason` 这个人话字符串**已经在内存里**：`filters/runner.py:302-314` 写进
  `ctx.rejection_details`（首次拒绝 wins，含 `reason` / `category` / `source` / `cites`），
  上限 `MAX_REJECTION_DETAILS = 20_000`（`runner.py:37`）。它只喂给 live 侧的
  "Rejected" 面板（`snapshots.py:143-194`），**从不落盘**。
- `step` 归属来自 R2 的 `ctx.current_step`——`record_rejections`（`runner.py:280`）拿得到 `fctx.ctx`。
- `_write_rejections_json`（`finalize.py:25-70`）现在读的是只有桶名的 `rejection_ledger`；
  改成以 `rejection_details` 为主、`rejection_ledger` 补 `also_cut_by`。

**多大。** 实测现状 **228.3 B/条**（110,935 B / 486 条）。
新增 `cut_at` 展开 + `source` + `cites` ≈ **+90 B/条**。

| run 规模 | 现在 | 加完 |
|---|---|---|
| `runs/data`（486 拒绝） | 111 KB | **~155 KB** |
| demo RUN-37 规模（17,440 拒绝） | ~3.9 MB | **~5.5 MB** |

**给产品负责人的三个瘦身选项**（选一个，或都不选）：
- **(a) 全留**（推荐）。5.5 MB 在一个已经有 28.7 MB `cache.db` 的目录里不值一提。
- **(b) 便宜规则不存 title**。`year` + `citation` 两桶通常占 80% 体量
  （runs/data 实测 347/486 = 71%），title 随时能从 `cache.db.paper_metadata` 反查。
  这样反而**比现在还小**（~2 MB）。
- **(c) 拆两个文件**：`rejections.json`（可浏览，沿用 20k 上限，全字段）+
  `rejection_counts.json`（精确总数，不受上限影响）。上限外的论文只进计数不进明细，
  和现在 `rejection_counts` / `rejected` 的语义一致。

**复盘时怎么用。**
- **Q-A 的正面答案**。今天用户看到 `["citation"]`，不知道是"第 2 步那个 beta=30 的"
  还是"Route 里 preprint 分支那个 beta=60 的"——而这两者的修法完全相反。
- `also_cut_by` 顺手暴露一类真实的坑：Parallel 的一个分支拒了、另一个分支收了，
  论文最终在 collection 里。live 侧已经在过滤这种（`snapshots.py:166`
  `if pid not in collection`），磁盘上没有对应信息。
- ⚠️ **这条同时修一个事实错误**：公开版下载包的 README
  （`web/public/backend/runs_fs.py:39-41`）白纸黑字写着 `rejections.json` 含
  "the human reason it was screened out"——**今天不含**。见 `escalations.md` E-LOG-02。

---

### R4 — 「formula 里是哪一条在误杀」：子查询判定落盘 · P1（**性价比第一**）

**是什么。** LLMFilter 的 formula 模式（`formula: "(q_ml | q_stats) & !q_survey"`）
把 N 条子查询的判定合并成一个 bool。建议把**合并前的 per-sub-query 判定**记下来：
```json
"c1e3f2d5…": {"cut_at": {"filter": "topic_llm", …},
              "sub_verdicts": {"q_ml": false, "q_stats": false, "q_survey": true}}
```

**挂在哪。** 一行。`screening/llm_runner.py:257-280 _dispatch_formula` 里的
`sub_verdicts: dict[str, dict[str, bool]]` **已经是完整的 per-query × per-paper 判定矩阵**，
第 279 行算完 `formula.evaluate(values)` 之后整个变量就出作用域了。
命名规范也是现成的：第 261 行子 filter 已经叫 `f"{name}::{qname}"`，
而 `BudgetTracker` 早就按这个名字分账——`runs/data/run_state.json` 里实测有
`llm_title_llm::q_ai`（3,707 tokens / 6 calls）、`::q_survey`、`::q_benchmark`、
`::q_biology`、`::q_software`。**token 账本记了子查询，判定结果没记，这是个纯遗漏。**

**多大。** 每条被 LLM 拒的论文一个紧凑 map，~40 B。
`runs/data` 实测 486 条拒绝里只有 85 条是 LLM 的（`llm_title_llm` 60 + `llm_abstract_llm` 25）
→ **3.4 KB**。demo 规模约 1,200 条 LLM 拒绝 → **48 KB**。

**复盘时怎么用。** 直接回答"我这个布尔式到底是哪一项在惹事"：
- `q_survey` 命中率 40% → 综述判定太激进，把方法论文当综述杀了（Q-A）。
- `q_ml` 命中率 95% → 这一条形同虚设，等于没筛（Q-B）。

**这是全提案里 value-per-byte 最高的一条：数据已经算出来了，写盘不产生任何额外
LLM 调用、任何额外 API 调用、任何额外计算。**

---

### R5 — 「这一步到底问了什么、拿回多少」：检索 query 日志 `queries.jsonl` · P1

**是什么。** append-only 的 JSONL，每一次对外检索一行：
```json
{"t": 1785970712.4, "step_idx": 2, "step": "ExpandForward#1",
 "kind": "citations", "anchor": "dc32a984…", "params": {"max_citations": 30},
 "reported_total": 5348, "returned": 5348, "kept_after_trunc": 30,
 "novel_after_seen": 27, "cached": false, "elapsed_ms": 412}
{"t": 17859709…, "step_idx": 5, "step": "ExpandBySearch#1",
 "kind": "search_bulk",
 "query_natural": "(protein structure prediction OR folding) AND deep learning NOT survey",
 "query_s2": "(protein structure prediction|folding) deep learning -survey",
 "params": {"limit": 1000, "sort": null, "pages": 3},
 "reported_total": 18422, "returned": 2841, "novel_after_seen": 2799, "cached": false}
```

**挂在哪。** 三个调用点，其中第一个的数据结构**已经存在**：
- `agents/iterative_search.py:321-325` 每轮构造 `AgentTurn(iteration, raw_response,
  query_natural, query_s2, total, new_ids)`——**完整得不能再完整**，返回给
  `expand_by_search.py` 之后只有 `last_query[:200]` 进了 stats（`expand_by_search.py:132-133`），
  `raw_response` / `total` / `new_ids` 全丢。
- `steps/expand_forward.py:66` / `expand_backward.py` 的 per-source 抓取。
- `steps/_expand_helpers.py:181` 的 `enrich_batch` 之前（ExpandBy* 家族共用）。

最省事的实现：`ctx.query_log.append(...)`，Finalize 一次性 dump，或直接边跑边 append
（jsonl 天生适合，且 run 崩了也留得住——这点比 Finalize 才写的文件强）。

**多大。** ~150 B/行（ExpandBySearch 的行带完整 query 串，~350 B）。
一次 snowball run 大约 = 源论文数 × 方向数：300 源 × 2 = 600 行 = **~90 KB**。

**复盘时怎么用。** 这是 **Q-A 最关键的一块拼图**：「X 没搜到」有两种完全不同的死法，
建议给用户的动作正好相反——

| 死法 | 证据 | 该改什么 |
|---|---|---|
| **从没被枚举到** | X 不在 `seen_ids`，且它的 anchor 那一行显示 `reported_total: 5348 → kept_after_trunc: 30` | 提 `max_citations`，或换 anchor（加 Rerank） |
| **枚举到了被杀了** | X 在 `seen_ids`，`rejections.json` 有 `cut_at` | 改那个 filter 的参数 |
| **query 根本没覆盖** | ExpandBySearch 那行的 `query_s2` 里没有 X 的关键词 | 改 `topic_description` |

**没有 R5，这三种在磁盘上长得一模一样。**

*补充事实*：`cache.db.search_queries` 表（`cache.py:61-66`）已经存了
`query_json` + `result_json`，所以 query **内容**今天部分可反查——
但没有 run 归属、没有步骤归属、没有顺序，且只有走 search 的那类。

---

### R6 — 「种子是不是一开始就歪了」：种子溯源 `seeds.json` · P1

**是什么。** 每个配置里的种子一行：
```json
{"input": {"title": "Highly Accurate Protein Structure Prediction"},
 "resolution": "search_match",
 "resolved_id": "dc32a984…",
 "matched_title": "Highly accurate protein structure prediction with AlphaFold",
 "title_similarity": 0.93,
 "siblings": ["arXiv:2106.xxxxx"], "sibling_reason": "external_ids DOI↔ArXiv",
 "status": "resolved"}
```
未解析的记 `{"status": "unresolved", "failure_reason": "search_match returned no paperId"}`。

**挂在哪。** `steps/resolve_seeds.py` 已经全算出来了——第 132 / 206 行的
`ctx.s2.search_match(title)` 返回、第 212/218/226 行的三种失败分支、
`include_siblings` 的兄弟发现逻辑。**但 stats 只吐 5 个聚合数**
（`resolve_seeds.py:268-273`：`input_seeds / primaries_resolved / siblings_added /
unresolved_titles / total_resolved`）。`LoadSeeds` 那边知道最后哪些真的 hydrate 成功了。

**多大。** ~250 B/种子。`config_rna.yaml` 有 54 个种子 → **13 KB**。可忽略。

**复盘时怎么用。** 标题解析是 silent 失灵的重灾区：`search_match` 拿回一篇
"看着很像"的错论文，整个雪球从错误的起点滚出去，**跑完一切正常，结果全是无关的**——
这是 Q-B 一个很常见但今天完全不可见的根因。今天磁盘上只有 `unresolved_titles=0`
（"都解析成功了"），看不出哪一条被换成了什么。`resolve_seeds.py:118-123` 的注释
自己也承认这个风险（"search_match will return the wrong paper"）。

---

### R7 — 「钱和 API 花在哪一步」：per-step 预算增量 · P2

**是什么。** `BudgetTracker` 的总账已经很细了（`run_state.json.budget` 实测有
`by_category` / `by_model` / s2 `by_type{batch,citations,metadata,references,search_match}`
各自的 api/cached 拆分）——**但全是 run 级**。建议在每步前后各取一次快照，存差分：
```json
{"step_idx": 4, "step": "ExpandBackward#2",
 "llm": {"calls": 12, "in": 41200, "out": 8100, "reasoning": 0, "cost_usd": 0.031,
         "cache_hits": 3},
 "s2": {"api": 206, "cached": 1,
        "by_type": {"references": {"api": 206, "cached": 1}}}}
```

**挂在哪。** `pipeline.run_pipeline` 的 step 循环已经把 `step.run()` 夹在中间
（`pipeline.py:396-398`）。前后各调一次 `ctx.budget.to_dict()`（`budget.py:373`）做差。
**engine 内部一行不用改**，纯粹是 runner 层的加法。

**多大。** ~400 B/步 × 7 步 = **~3 KB**。

**复盘时怎么用。**
- 「这个 run 花了 $2.3，其中 $1.9 在第 6 步」——直接定位该砍哪一步。
- s2 缓存命中率骤降 = 那一步在做全新的工作（或者缓存 key 变了）。
  实测 `runs/webui/e8cd1f0f4a72`：225 次 API / 15 次命中（冷跑）；
  `runs/data`：21 次 API / 927 次命中（几乎全走缓存）——这个对比只有 run 级，
  拆到步级才知道是哪一步在打网。
- **顺手补 UI 契约**：新 UI `stats.json` 每步要 `{calls, graph, reco, tokens}`
  （`run37/stats.json` 实测 `FWD-04: {calls:7, graph:7, reco:0, tokens:28764}`），
  recon §2.5 明确标为"现状没有 per-step token/调用计量"。R7 就是它。

---

### R8 — 「跑了 13 分钟，花在哪」：per-step 计时 · P2（几乎免费）

**是什么。** 每步加 `started_at` / `ended_at` / `elapsed_sec`；run 级加
`started_at` / `ended_at` / `status` / `stop_reason`。

**挂在哪。** `steps/shape_log.py:49-51 ShapeLog.record` 加两个 float 字段
（`shape_summary.json` 每行就多两个 key）；run 级的进 R1 的 manifest。
`progress.py:336, 408` 早就在算 elapsed 用来显示，只是不存。

**多大。** ~50 B/步。**全 run < 1 KB。**

**复盘时怎么用。** 不直接答 Q-A/Q-B，但它是所有性能类复盘的门槛，
而且今天 **engine 落盘里根本没有任何时间戳**——`web_run.json`
的 `created_at/started_at/completed_at` 是已死的 `web/backend` 写的，
web/live 的时间只活在内存 `RunState` 里。新 UI 的
`stats.json` 要 `startedAt / endedAt / elapsed`（实测 RUN-37: 781 s），
今天没有来源。

---

### R9 — 「哪里被悄悄截断了」：截断与提前终止台账 · P1（防 silent fail）

**是什么。** 一个小清单，记下这个 run 里**每一处悄悄丢过数据的地方**：
```json
[{"kind": "max_citations", "step": "ExpandForward#1", "anchor": "dc32a984…",
  "available": 5348, "taken": 30, "dropped": 5318, "order": "citation_count desc"},
 {"kind": "budget_cap", "step_idx": 7, "detail": "max_papers_total 2000 reached"},
 {"kind": "rejection_detail_cap", "recorded": 20000, "total": 24118},
 {"kind": "already_searched", "step": "ExpandBySearch#2",
  "detail": "fingerprint 4f2a… 已跑过，本步整个跳过"}]
```

**挂在哪。** 所有数值在切断的地方就是现成的：
- `expand_forward.py:76-77` `citers.sort(...)` + `citers[:self.max_citations]`
- `pipeline.py:435` 的 `budget.is_exhausted / max_papers_total` break
- `filters/runner.py:37, 302` 的 `MAX_REJECTION_DETAILS`
- `agents/iterative_search.py:299, 308` 的 `max_papers_per_iteration` cap
- `expand_by_search.py:87-96` 的 fingerprint 短路（**整步跳过，今天只有一行 INFO log**）
- `snapshots.py:18` 的 `_MAX_GRAPH_NODES = 700`（只影响展示，仍应记）

**多大。** 大多数 run **< 2 KB**。`max_citations` 那类是每 anchor 一行，
可以只记 `dropped > 0` 的（或直接并进 R5 的 query log，天然同源）。

**复盘时怎么用。** **这是 Q-A 最阴险的一类答案。**
用户问「为什么 X 没搜到」，真相是：X 的 anchor 有 5,348 个 citer，
`max_citations: 30` 按引用数倒序只取了前 30，X 排第 900。
**这条信息今天在磁盘上一个字节都没有**，用户会一路怀疑是 filter 的锅，
去调 filter，然后什么也调不出来。

和 decisions-ledger Round 2 的 `missing-states` 精神一致：**demo 只做了 happy path，
要防 silent fail**——R9 是 silent fail 在后端的对应物。

---

## 3. 体积总账

| | 小 run（`runs/data` 规模：21 收 / 486 拒） | 大 run（demo RUN-37 规模：354 收 / 17,440 拒） |
|---|---|---|
| R1 配置快照 | 11–16 KB | 11–16 KB |
| R2 PRISMA | 4 KB | 8 KB |
| R3 拒绝归属（**增量**） | +44 KB | +1.6 MB |
| R4 子查询判定 | 3.4 KB | 48 KB |
| R5 query 日志 | ~30 KB | ~90 KB |
| R6 种子溯源 | 1–13 KB | 1–13 KB |
| R7 per-step 预算 | 3 KB | 3 KB |
| R8 计时 | < 1 KB | < 1 KB |
| R9 截断台账 | < 2 KB | < 2 KB |
| **新增合计** | **~110 KB** | **~1.8 MB** |
| 同目录现有 | `cache.db` 28.7 MB + `citeclaw.log` 2.0 MB + 其余 ≈ **31 MB** | 更大 |
| **占比** | **0.35%** | **< 6%** |

若选 R3 的瘦身方案 (b)，大 run 的合计反而降到 **~0.3 MB**。

**一条顺带的观察**（不属提案，供参考）：现在的 `citeclaw.log` 2.0 MB 里，
绝大多数是 `Cache HIT/MISS [paper_metadata] <hash>` 这类 DEBUG 行，信息密度极低
（实测 tail 20 行里 18 行是缓存日志）。R5 的 `queries.jsonl` 是它的结构化替代品；
R5 落地后可以考虑把缓存那几行降级，**净效果可能是 run 目录变小**。

---

## 4. 和新 UI 契约的对应关系

本提案不是凭空发明字段，其中三条正好补上 recon 已经标记为"后端缺口"的东西：

| 新 UI 要的（`web/design-reference/uploads/run37/stats.json` + `run-mock.js` `PIPELINE`） | 现状 | 本提案 |
|---|---|---|
| 每步 `found / kept / rej` | ❌ recon §2.5 标为缺 | **R2** |
| 每步 `cuts[6]`（year/citation/keyword/LLM-title/LLM-abstract/duplicate） | ❌ | **R2**（吐不定长 stages，前端折叠；需 design 裁决） |
| 每步 `stats{calls, graph, reco, tokens}` | ❌ recon §2.5 标为缺 | **R7** |
| `startedAt / endedAt / elapsed` | ❌ engine 无时间戳 | **R8** |
| step drill 的 `rdDef` filter 定义弹层 + "Edit filter in Build" 跳转 | ❌ 无生效参数落盘 | **R1** 的展平 filter 表 |
| hero 的 PRISMA 计数 `Accepted / Rejected / Steps done` | ✅ 现有 `rejection_counts` 够 | — |
| `stats.rejections{}` 分桶 | ✅ 现有 `rejection_counts` 就是 | — |

---

## 5. 需要产品负责人/design 裁决的三点

1. **PRISMA 的桶形状**（R2）。demo 写死 6 桶，真实 pipeline 是任意 filter 树。
   建议：后端吐真实的有序 stages，前端映射。**要不要给 design agent 出稿**，
   在 step drill 里展示"任意条数的 filter 阶段"？（建议进 design-lane 批次。）
2. **`rejections.json` 的瘦身取舍**（R3 的 a/b/c）。默认建议 (a) 全留。
3. **R1 是否下沉进 engine**。今天 `pipeline_config.json` 只有 web/live 写，
   CLI run 完全没配置记录。下沉 = engine 改动（本期禁止），
   不下沉 = CLI run 永远复盘不了。建议**下沉**，排进后续批次。

---

## 6. 建议的落地批次（若获批）

| 批次 | 内容 | 理由 |
|---|---|---|
| **第一批** | R2 的地基（`ctx.current_step`）+ **R4** + **R3** | R4 是一行、零成本、性价比第一；R3 依赖 `current_step`；三者一起就把 Q-A 的"被谁杀的"答全了 |
| **第二批** | **R2** 完整 + **R7** + **R8** | 一起做，因为都挂在 `run_pipeline` 的同一个 step 循环上；顺手补齐新 UI 的 `stats.json` |
| **第三批** | **R5** + **R9** | 一起做，因为 `max_citations` 截断和 query 日志天然同源 |
| **第四批** | **R1** + **R6** | R1 需要 engine 下沉的裁决；R6 独立、极小 |

每批都应该**先补一个真实 run 的回归样本**（跑 `configs/config.yaml` 的 stub 模式，
落一个 fixture 进 `tests/`），否则这些字段会像 `pipeline_config.json` 一样悄悄只在一条路上写。

---

## 7. 签字栏

| 编号 | 一句话 | 大小（大 run） | 建议 | 批准？ |
|---|---|---|---|---|
| R1 | 配置快照 + 生效 filter 展平表 | 16 KB | P0（点名） | ☐ |
| R2 | PRISMA 逐阶段计数 | 8 KB | P0（点名） | ☐ |
| R3 | 拒绝原因带阶段归属 + 人话 reason | +1.6 MB | P1 | ☐ |
| R4 | formula 子查询判定 | 48 KB | **P1 · 性价比第一** | ☐ |
| R5 | 检索 query + 命中数日志 | 90 KB | P1 | ☐ |
| R6 | 种子溯源 | 13 KB | P1 | ☐ |
| R7 | per-step 预算/缓存增量 | 3 KB | P2 | ☐ |
| R8 | per-step 计时 | 1 KB | P2 | ☐ |
| R9 | 截断与提前终止台账 | 2 KB | P1（防 silent fail） | ☐ |

**另需裁决**：§5 的三点；以及 `escalations.md` 的 **E-LOG-01（密钥泄漏，建议优先）**
和 **E-LOG-02（下载包 README 与实际文件不符）**。

---

*本文档由 B7 lane 于 2026-08-13 起草。审计样本：`runs/data/`、`runs/test_data/`、
`runs/webui/{e8cd1f0f4a72, cf7278602b6d, c48376f3714e}`、`runs/web/20260716_145857_c65b6a`。
代码引用基于分支 `wiring/omniknowledge-0813`。未修改任何代码。*
