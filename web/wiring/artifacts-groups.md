# artifacts-groups.md — 分组产物 schema（Leiden communities + 主题图 2D 坐标）

> 中文摘要：引擎现在会把**每篇论文的 Leiden community、每篇论文的主题图二维坐标、
> 每个 group 的名字/关键词/中心点、以及各划分算法的 modularity 与 NMI/ARI**
> 落盘到 run 目录。本文档是这批文件的唯一 schema 契约：前端 Explore 屏（topic map、
> community cards、ⓘ "How this was computed"）按此读取。
>
> Owner: B5 (engine)。改字段名/分隔符前必须先改这里。
> 设计基准 fixture：`web/design-reference/uploads/run37/`（只读真相）。

---

## 0. TL;DR — 谁写、什么时候写

| | |
|---|---|
| 写入者 | `citeclaw.output.groups.write_group_artifacts(ctx)` |
| 调用点 | `Finalize` step（在 graphml 之后、rejections 之前），异常被 catch 成 WARNING |
| 前置条件 | pipeline 里跑过至少一个 `Cluster` step，且结果落在 `ctx.clusters` |
| 无 Cluster step 时 | **一个文件都不写**，返回 `[]`。这是正常配置，不是错误 |
| 迭代运行 | 文件名自动加 `.expN`（`iteration > 1` 时），与 `literature_collection.expN.json` 一致 |
| 网络 / LLM | 零。全部从 `ctx.clusters` 里已有的 `ClusterResult` 派生 |

写出的文件：

```
<run_dir>/
  paper_groups.csv          每篇论文一行：坐标 + 各种 group id
  topics.csv                每个 topic 一行（embedding 空间分组）
  communities_leiden.csv    每个 community 一行（citation 空间分组）
  community_methods.csv     每种图划分算法一行：形状 + 质量指标
  groups.json               机读索引：跑了什么、用了什么参数、出了几组
```

---

## 1. 算法选型与既成事实

### 1.1 Leiden = `leidenalg`（不是 igraph 内置，也不是 graspologic）

`src/citeclaw/cluster/leiden.py`，注册名 `leiden`。

选 `leidenalg` 的**实证理由**（不是偏好）：在 run37 fixture 的 354 节点 / 1239 边
引用图上，

```
leidenalg.find_partition(g, ModularityVertexPartition, seed=42, n_iterations=2)
```

**逐标签复现** fixture 的 `community_leiden` 列 —— 12 个社区，size
`[57, 54, 52, 49, 37, 31, 30, 27, 14, 1, 1, 1]`，modularity 0.5558，
NMI = ARI = 1.0000。也就是说 fixture 本身就是这个调用产出的，它因此从"形状参考"
升级成了**golden test**（`tests/test_cluster_leiden.py::TestRun37Fixture`）。

> ⚠️ **复现的前提是顶点顺序相同。** Leiden 的 RNG 轨迹依赖顶点遍历顺序，
> 而生产路径 `build_citation_graph()` 是按 `paper_id` **排序**建图，
> fixture 用的是 `accepted.csv` 的行序（= 接收顺序）。同一张图换个顶点顺序，
> leidenalg 会落到另一个同样合法的局部最优（实测 12 组、Q=0.5628、最大组 74，
> 对比 fixture 12 组、Q=0.5558、最大组 57 —— 两者都非退化，我们这条甚至
> modularity 更高）。所以：
> * golden test 直接在 `graph.json` 的顶点序上调 `leiden_partition`，
>   钉住的是**算法 + seed + 顶点序**三件套；
> * 真实 run 的社区划分**不会**逐标签等于 fixture，只保证 run-to-run 稳定
>   （`build_citation_graph` 的排序是确定性的）；
> * writer 层另有一个测试，把 fixture 自己的 label + 坐标喂进来，
>   验证产出的 `communities_leiden.csv` / `topics.csv` / `community_methods.csv`
>   与 fixture **逐格相同**（31 个 centroid 里 2 个 round-half 平局差 0.01）。

被否掉的两个候选：

| 候选 | 结论 | 依据 |
|---|---|---|
| igraph 内置 `Graph.community_leiden` | **降级为 fallback** | 同一张图上落到明显不同的划分：ARI 0.53 / NMI 0.74 vs fixture，最大社区 79（fixture 57）。且不暴露 RNG seed，跨 igraph 构建不可复现 |
| `graspologic` | 否 | Rust 移植，非参考实现；且会把 numpy/scipy/sklearn/networkx/gensim 一整套拖进 core 安装 |

依赖已 pin 进 `pyproject.toml` **core** deps（不是 extras）：`leidenalg>=0.10,<1`。
理由：Leiden 是新 UI 的默认分组算法，且它只是 `python-igraph`（已是 core dep）
之上的一层薄 C++ 扩展。许可证：`leidenalg` 是 GPL-3.0+，`python-igraph` 已经是
GPL-2.0+，**没有引入新的许可证类别**。

`leidenalg` 缺失时会 fallback 到 igraph 内置实现并打 **WARNING**；
`groups.json` 里 `quality.backend_is_leidenalg` 会是 `0.0`，
所以 fallback 永远不会冒充参考实现。

### 1.2 Community id 按 size 降序重编号

Leiden 的 id 经 `citeclaw.cluster.base.relabel_by_size()` 重排：`0` = 最大社区，
`1` = 次大，依此类推；同 size 时按成员里最小的 paper_id 决胜（纯函数，不依赖 dict
迭代顺序）。这是为了对上 UI 的 `C00`/`C01` 卡片排序。

**排序发生在整个 `ctx.collection` 上，然后才裁到 signal**——所以 `C00` 的含义是
"本语料最大的社区"，不是"本 signal 里最大的"。

> ⚠️ 只有 Leiden 这么做。`walktrap` / `louvain` 保留 igraph 原始 id，
> 它们既有的 graphml 列输出**逐字节不变**。

### 1.3 2D 坐标 = 第二次 UMAP，clustering 仍在 5D

现状（改动前）：`TopicModelClusterer` 把 SPECTER2 降到 `n_components=5` 喂
HDBSCAN，中间结果**不落盘**。

改动：**5D 那条路一行没动**（cluster 归属逐字节不变），另外跑一次
`n_components=2` 的 UMAP 专供显示，结果挂在
`ClusterResult.coords_2d: dict[paper_id, (x, y)]`。这是 BERTopic 的标准做法：
密度聚类在 5D 更稳，人看的是 2D。代价约 1 秒 / 354 篇。
`TopicModelClusterer(emit_coords_2d=False)` 可关。

`random_state` 被 pin 住（默认 42），UMAP 因此单线程且可复现 —— 同一语料每次跑
出同一张图，用户"右边那一坨"的空间记忆不会因为重跑而失效。

### 1.4 坐标归一化：1000 单位方框，保持长宽比

`normalize_to_box()`：两轴共用**同一个** scale（不剪切），长轴恰好填满
`[25, 975]`，短轴居中于 500。

这是从 fixture 反推出来的：run37 的 `accepted.csv` 里 `x ∈ [25.0, 975.0]`、
`y` 中心恰为 500.0 且跨度更短 —— 即单一等比缩放，而非逐轴 min-max 拉伸。
对上它意味着引擎输出可以直接进 UI 坐标系，不需要任何 rescale 垫片。

坐标保留 2 位小数（跨平台 float repr 稳定）。

---

## 2. `paper_groups.csv`

每篇论文一行，按 `id` 升序（稳定 diff）。

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | str | S2 paperId |
| `x` | float / **空** | 主题图横坐标，`[25, 975]`。**没有 SPECTER2 向量的论文这里是空字符串，不是 0** |
| `y` | float / **空** | 同上 |
| `topic` | int / 空 | topic id；`-1` = HDBSCAN noise。无 topic model 时整列不存在 |
| `topic_label` | str | topic 的 LLM/c-TF-IDF 名字；noise 与未命名为空 |
| `community_leiden` | int / 空 | Leiden 社区 id（size 降序重编号，从 0 连续） |
| `community_<algo>` | int / 空 | 每个跑过的图划分算法各一列（`walktrap` / `louvain` / …） |

**空单元格 = "没算"，不是 0。** 写 0 会把论文钉在主题图左上角、把方法钉在
modularity 零，两者都会被读成真实测量值。

### 与 fixture 的差异（重要）

fixture 把这些列合并进了 `accepted.csv`：

```
id,title,authors,venue,year,cites,accepted_at_step,url,x,y,topic,topic_label,
community_louvain,community_leiden,community_walktrap,community_infomap,community_label_propagation
```

B5 **只产出自己拥有的列**，写进独立的 `paper_groups.csv`：

* `title/authors/venue/year/cites/url` — 已经在 `literature_collection.json` 里，不重复落盘；
* `accepted_at_step` — 属于 Q4「步骤归属」工作项（改引擎事件协议），不在 B5 范围；
* `community_infomap` / `community_label_propagation` — **这两个算法引擎里不存在**，
  见 §5 未做项。

→ 下游若要还原 fixture 形状的 `accepted.csv`，按 `id` join 本文件即可。

---

## 3. `topics.csv` / `communities_leiden.csv`

两张表**列名与分隔符和 fixture 完全一致**（有测试逐字节比对表头：
`tests/test_output_groups.py::test_headers_are_byte_identical_to_the_design_fixture`），
唯一差别是第一列叫 `topic_id` 还是 `community_id`。

| 列 | 说明 |
|---|---|
| `topic_id` / `community_id` | int，组 id |
| `label` | `ClusterMetadata.label`（naming pipeline 产出；`mode: none` 时为空） |
| `description` | `ClusterMetadata.summary` |
| `size` | **从 `membership` 重算**，不信 metadata（naming 可能没跑） |
| `keywords` | c-TF-IDF 关键词，`"; "` 分隔 |
| `centroid_x` / `centroid_y` | 成员坐标的**算术平均**（非中位数），2 位小数；无坐标时为空 |
| `representative_paper_ids` | `"; "` 分隔 |
| `representative_paper_titles` | `" \| "` 分隔（标题里的 `\|` 会被替换成 `/`） |

行序：**size 降序**（UI 卡片渲染顺序），同 size 按 id 升序。
**noise（`-1`）不是一行** —— 它是"没有组"，UI 从 `paper_groups.csv` 直接把未归组
论文画成灰点。fixture 同此：20 个 topic 值里 19 个成行。

`centroid` 用平均值是实测结论：run37 全部 19 个 topic + 12 个 community 的
`centroid_x/y` 与成员坐标平均值**两位小数全等**（中位数对不上）。

> **community 的 centroid 也用主题图坐标。** 社区自己没有 embedding；
> 它在图上的位置就是其成员在 *embedding 空间*的位置。这解释了为什么
> community 的点在主题图上看起来是散的（fixture 实测 silhouette −0.079），
> 而 topic 是聚的（+0.447）—— 不是 bug，是两种空间。

多次 Cluster（例如 Parallel 分支内一次 + 全语料一次）产出同算法两个结果时，
第一个用裸文件名 `communities_leiden.csv`，之后的加 `store_as` 后缀
（`communities_leiden_<store_as>.csv`），列名同理。

---

## 4. `community_methods.csv`

ⓘ "How this was computed" 面板的数据源。**每个图划分算法一行**（topic model 不占行，
它是被比较的基准）。

| 列 | 说明 |
|---|---|
| `method` | `leiden` / `louvain` / `walktrap` / … |
| `n_communities` | 非 noise 组数 |
| `modularity` | 由 clusterer 自己算并放进 `ClusterResult.quality`（它知道自己划分的是哪张图）。没报的算法留**空**，不在这里凭空重算 |
| `largest_community` | 最大组 size |
| `largest_fraction` | `largest / 论文总数`，4 位小数 |
| `singletons` | size == 1 的组数 |
| `median_size` | `int(statistics.median(sizes))`（截断；fixture 实测如此：leiden sizes 中位数 30.5 → 30） |
| `nmi_vs_topic_model` | 4 位小数；无 topic model 时空 |
| `adjusted_rand_vs_topic_model` | 同上 |

### NMI / ARI 的实现与约定

纯 Python 实现在 `citeclaw.cluster.agreement`，**不依赖 sklearn** —— 因为
sklearn 只在可选的 `topic_model` extras 里，而这两个数字必须每个 run 都有。

* NMI 用**算术平均**归一化（= sklearn `average_method` 默认值）；
* ARI 是标准配对计数公式，**允许为负**（比随机还差是有意义的信息，不 clamp）。

**约定（load-bearing）**：在两个划分都覆盖的论文上比较，且 topic model 的
`-1` noise 桶**当作普通标签**参与。这是 fixture 的做法 —— 五种算法的
NMI/ARI 与 fixture 的 `community_methods.csv` **四位小数全等**（
`tests/test_cluster_agreement.py::TestRun37Fixture`）。
改成剔除 noise 会把 leiden 的 NMI 从 0.4897 推到 0.5026，所以这是约定而非口味。

---

## 5. `groups.json`

机读索引，给后端 API / 调试用：

```jsonc
{
  "coords_2d_papers": 354,
  "coord_space": {"box": 1000.0, "pad": 25.0, "origin": "top_left"},
  "topics":  {"store_as": "...", "file": "topics.csv", "algorithm": "topic_model",
              "n_topics": 19, "n_noise": 12, "n_papers": 354},
  "communities": [{"store_as": "...", "file": "communities_leiden.csv",
                   "algorithm": "leiden", "n_communities": 12, "n_papers": 354,
                   "quality": {"modularity": 0.5558, "n_communities": 12.0,
                               "seed": 42.0, "resolution": 1.0,
                               "backend_is_leidenalg": 1.0}}],
  "papers":  {"file": "paper_groups.csv", "n_papers": 354},
  "methods": {"file": "community_methods.csv", "rows": [ /* 同 §4 各列 */ ]}
}
```

`coord_space.origin = "top_left"` 记录的是 fixture 的 y 轴朝向；UI 侧
`topic-viz.js` 会自己再归一化到 640 单位并翻转 y（见 recon 报告），本文件
不替它做这件事。

---

## 6. 未做 / 已知差异清单（B5 交付时的诚实清单）

1. **`infomap` / `label_propagation` 两个 clusterer 不存在。** fixture 的
   `community_methods.csv` 有 5 行、`accepted.csv` 有 5 个 community 列；
   引擎现在能产 3 种（`leiden` / `louvain` / `walktrap`）。igraph 内置
   `community_infomap` / `community_label_propagation`，补齐各约 40 行，
   但**新 UI 只消费 leiden**，故本期不做。
2. **`accepted_at_step` 列不产。** 属 Q4「步骤归属改事件协议」工作项。
3. **`topics.csv` 的 `description` 是 LLM 产出**，fixture 里那些
   plain-language 描述是人写的；naming `mode` 不含 `llm` 时该列为空。
4. **2D 坐标只有跑了 `topic_model` 才有。** 只跑图划分的 run，
   `paper_groups.csv` 的 `x`/`y` 整列为空，`centroid_x/y` 也为空 ——
   UI 需要有"本 run 没有主题图"的状态（已记入 `missing-states.md` 的关注面）。
5. **坐标不跨 run 稳定。** 同一语料同一 seed 稳定，但语料变了（`--continue-from`
   加了论文）UMAP 会重排整张图。fixture 无此问题因为它是单次 run。
   若产品需要"增量运行时地图不跳"，需要 UMAP `transform()` 的增量投影，
   属独立工作项。

---

## 7. 测试入口

| 文件 | 覆盖 |
|---|---|
| `tests/test_cluster_leiden.py` | `relabel_by_size`、clusterer 行为、**run37 golden test（逐标签复现 fixture）**、354 节点非退化断言 |
| `tests/test_cluster_agreement.py` | NMI/ARI 解析解、与 sklearn 逐位对齐、**复现 fixture 的 5 组指标** |
| `tests/test_topic_coords.py` | `normalize_to_box` 几何性质、真 UMAP 投影的有限性/确定性/结构保持、5D 路径不受影响 |
| `tests/test_output_groups.py` | 全部文件契约（列名/行序/分隔符/空值语义）、表头与 fixture 逐字节比对、**354 节点端到端** |

全部使用固定 seed；不联网，不调 LLM。
