# OmniKnowledge 接线决策台账（Round 1，2026-08-13）

## 已拍板
- Q1 技术路线：倾向重写成工程代码（待第二轮确认保真方法论后最终定）
- Q2 登录/部署：选 B（邀请码方向）；但**本期登录页不接**（保持随便输都能进），Home 要接真数据
- Q3 语料版本链：**A 全做**（六种精炼操作全上）；建议独立 sub-agent 队伍 + 指定 phase 和 evaluation
- Q4 步骤归属：**A 改引擎事件协议**记录逐步归属；times-hits 需改 UI 逻辑（跑完才定型）；relevance & LLM confidence 后端没有→**暂时全写 1.00**
- Q5 Agent 面板：**不接**，保持三栏但做成"死的"；用户发送任何内容→报错（措辞参考 Claude Code 风格，我拟稿）；派专职 design agents 处理接线所需的补充设计（按 design-system，截图验证，标准："世界顶级设计师会采纳吗"）
- Q6 分组算法：**A 真实现 Leiden**
- Q7 引用语句：**B 变体**——只要 citation statements 原句，不要 section 分类/TLDR；派单独 agent 从 Semantic Scholar 等数据源验证覆盖度与质量（针对 354 篇 corpus）
- Q8 作者协作网络：**永久砍掉**（用户认为鸡肋）；外部图上传、任意运行浏览也不要
- Q9 Home：**B 轻量项目**（项目=文件夹），文献库砍掉
- Q10 流水线编辑器：**A 接受子集**（刻意为 general users 牺牲灵活性），复杂配置只读+保存不丢数据
- Q11 旧能力回归：design agents 塞回**暂停/继续 + 上限提额弹窗**（全屏暗淡+居中弹窗，复用 UI 里已有的弹窗样式，不许新造）；live log 砍、心跳砍、花费本期不接
- Q12 预算/供应商：**A 全兑现**（三个预算真执行 + 接入 Anthropic + 成本预估）
- Q13 改名：**A 只改用户可见**（内部 kl/CiteClaw 前缀不动）
- Q14 分支 wiring/omniknowledge-0813（基于 webapp-public）OK；死代码不管

## 最终报告必须提醒用户的"未接清单"（持续累积）
1. 登录页未接（纯装饰，任何输入都能进）
2. Agent 对话面板 + 文献综述整套未接（用户后续专门开发）
3. relevance & LLM confidence 硬编码 1.00（用户还没想清楚语义）
4. 文献库（跨项目论文总表+收藏夹+转种子集）整栏砍掉
5. 花费仪表盘未接（后端有数据，UI 无入口）
6. live log、心跳、上限提额倒计时原版交互——砍掉
7. 作者协作网络——永久砍掉
8. 引用语句 section 分类/TLDR——不做（只做原句）

## Round 2 已拍板（2026-08-13）
- Q15 路线：**A 重写**，但所有动手活交给 agents，orchestrator 本人不上手。对拍基准视口：**1600×900 + iPad Pro 12.9**。源头真相 = "KnowledgeLab iPad Demo.html" 渲染出来的样子（Claude Design 内部可能 link 多个文件，以该文件所见为准）。变体锁定：pipelineStyle='Flow chart (6d)'，networkPalette='Neutral ink'，collectionStyle='Cover rows'（已核实均为该文件默认值）。**响应式布局功能不许丢**。
- Q16 登录页：确认不接，随便输入都能进；目前仅 local deployment。
- Q17 历史运行：**A 导入为版本 1**。追加需求：丰富磁盘 run 日志（debug 用途、不大量占磁盘）——至少记录 pipeline+filter configuration、PRISMA flow counts（用户写作 PARSIMA，已核实新 UI 有 PRISMA flow 概念、代码库零命中）；agent 审计现有日志后提"甜品级"补充清单，供用户复盘"为什么这个 run 没搜到某文章/搜出一堆无关文章"。
- Q18 times-hits：**A 实时累计值**，跑完定格。
- Q19 报错措辞：**B 技术版** "Assistant unreachable — no backend is wired to this panel."（以后部署可复用）。追加需求：wiring 全程专门记录**缺失的 UI 状态**（demo 只做了 happy path，要防 silent fails），汇总成 missing-states register。
- Q20 Home 右栏：**文献库栏保留但做成死的**（点击无反应），后续再设计 UX。
- Q21 UI 改动清单：**全部批准**（其中第 8 项 Home 右栏改为"保留死栏"方案）。

## Round 3 已拍板（开工令，2026-08-13）
- 试点页改为 **Build**（用户判断其接线最直接），设三道闸：静态对拍 → 交互对拍 → 真数据。Settings 回归普通排期。
- 新增 **import 系统专项测试线**：专门 agent 测试用户可能上传的各种文件格式（PDF、BibTeX、RIS、DOI 列表等），建测试样本库（含畸形文件）。
- **文件格式图标**：交 design agent 评估——若 demo 的 import UI 没有各格式图标，则出提案进 UI 改动签字批次。
- 用户已明确说"开工，没有其他问题了"。

## 更新后的未接清单（最终报告必须提醒）
1. 登录页未接（纯装饰，任何输入都能进；仅 local deployment）
2. Agent 对话面板 + 文献综述整套未接（发送→"Assistant unreachable"报错；用户后续专门开发）
3. relevance & LLM confidence 硬编码 1.00（用户还没想清楚语义）
4. 文献库栏：保留外观但完全不可交互，UX 待重新设计
5. 花费仪表盘未接（后端有数据，UI 无入口）
6. live log、心跳、上限提额倒计时原版交互——砍掉（提额弹窗改为复用现有弹窗样式的新设计）
7. 作者协作网络——永久砍掉
8. 引用语句 section 分类/TLDR——不做（只做原句；覆盖度侦察进行中）
