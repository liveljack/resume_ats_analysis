# 简历质量评分 + 简历-岗位匹配度评分系统 — 设计方案

> 目标：对简历质量打 1-10 分，对简历与岗位（JD）匹配度打 1-10 分，中英双语，无自有标注数据（用开源数据集冷启动），混合方案（LLM 评分 + ML 语义匹配 + 规则校验），目标准确率 >80%。
>
> 本方案基于最新学术论文与开源实现的对抗式检索验证，详见文末「参考来源」。

---

## 1. 可行性结论

**结论：可行，>80% 准确率在混合方案下现实可达——但需明确"准确率"的度量口径。**

关键依据：

1. **无标注冷启动路径成立**。Self-Taught Evaluator（arXiv:2408.02831）证明：在完全无人工标注的情况下，通过合成数据自迭代，LLM-judge 在 RewardBench 上从 75.4 → 88.3（多数投票 88.7），超越 GPT-4 评判、媲美有标注训练的顶配奖励模型。直接背书"无自有数据 + >80%"场景。
2. **多语种语义匹配有成熟 SOTA**。BGE-M3（arXiv:2402.03216）单模型支持 100+ 语言（含中英）、8192 token 上下文（一份简历/JD 单次嵌入无需分块），同时输出 dense/sparse/ColBERT 三路向量，混合检索效果最佳。MIT 许可，可商用。
3. **双塔 + 交叉编码器两阶段是业界标准**（sbert.net 官方推荐架构）。Bi-encoder 做召回，Cross-encoder（bge-reranker-v2-m3）做精排，直接产出匹配分。
4. **LLM-as-judge 可靠性非自动获得**，需刻意设计（一致性、偏差缓解、场景适配）。prompt 工程 + 校准是达到 80% 的关键工作量，而非调参。

### 1.1 关于"准确率 >80%"的现实约束

简历评分本质是主观任务，没有绝对 ground truth。可操作的口径有三种，建议组合使用：

| 口径 | 说明 | 难度 |
|---|---|---|
| **排序一致性** | Spearman/Kendall 与人工排序 | 最稳，>80% 易达成 |
| **二分类达标判定** | 达标/不达标与人工共识 F1 | >80% 可达 |
| **绝对分数误差** | 1-10 分 ±1 内命中率 | 最难，需大量校准 |

**建议以「排序一致性 + 二分类达标 F1」作为主指标承诺 >80%，绝对分误差作为辅助。**

---

## 2. 系统架构（混合三层 + 融合）

```
                ┌─────────────────────────────────────────────┐
  简历(PDF/文本) │  Layer 0: 解析与结构化                          │
  + JD(文本)    │  pyresparser/OpenResume(英) + 中文NER(NCRF++)  │
  ─────────────▶│  → 结构化字段 + 原始段落                        │
                └───────────────┬─────────────────────────────┘
                                │
                ┌───────────────▼─────────────────────────────┐
                │  Layer 1: 规则/特征层 (廉价、确定性、可解释)      │
                │  简历质量维度:                                  │
                │   - 量化指标占比(含数字的bullet比例)             │
                │   - STAR格式覆盖、动词力度                      │
                │   - 关键词覆盖(JD词在简历的命中率)               │
                │   - 排版完整性(联系方式/教育/经历段齐全度)        │
                │   - 长度/分段合理性                              │
                │  → rule_quality_score (0-10 锚定)              │
                └───────────────┬─────────────────────────────┘
                                │
                ┌───────────────▼─────────────────────────────┐
                │  Layer 2: ML 语义匹配层 (中英、推理便宜)         │
                │  BGE-M3 双塔: cosine(emb(resume), emb(JD))     │
                │  BGE-reranker-v2-m3 交叉编码: 精排匹配分         │
                │  → ml_match_score (0-10)                       │
                └───────────────┬─────────────────────────────┘
                                │
                ┌───────────────▼─────────────────────────────┐
                │  Layer 3: LLM-as-Judge (准确率上限、可解释)      │
                │  GLM-4.6 / Claude，rubric-based prompt:        │
                │   - 质量评分: 按维度 rubric 打分 + 理由          │
                │   - 匹配评分: 技能/经验/资历维度对齐 + 理由        │
                │  self-consistency: 3次采样多数投票              │
                │  → llm_quality_score, llm_match_score          │
                └───────────────┬─────────────────────────────┘
                                │
                ┌───────────────▼─────────────────────────────┐
                │  Fusion: 加权融合 (权重用少量人工锚定集校准)       │
                │  quality = w1·rule + w2·llm  (ML不参与质量)     │
                │  match   = w3·rule_kw + w4·ml + w5·llm         │
                │  初值: quality(0.3,0.7) match(0.2,0.3,0.5)     │
                └─────────────────────────────────────────────┘
```

### 2.1 为什么三层而非单 LLM

- **规则层**：提供确定性锚点和可解释性，防止 LLM 漂移。
- **ML 层**：提供低成本语义相似度，可批量预筛、降本。
- **LLM 层**：提供准确率上限和自然语言理由。
- **融合层**：让权重可被少量人工样本校准——这是无标注场景下逼近 80% 的关键旋钮。

---

## 3. 数据集清单（冷启动）

| 数据集 | 用途 | 语言 | 来源 |
|---|---|---|---|
| **PJF14** (Joint Repr. Learning for Person-Job Fit, ACL 2022) | 简历-JD 匹配训练/评估 | 英 | aclanthology.org/2022.acl-long.314/ |
| **ESC: English Skill Corpus** (LREC 2022) | 技能-岗位匹配 | 英 | aclanthology.org/2022.lrec-1.250/ |
| **arXiv:2204.10967** 对比/双编码匹配方法 | 方法+数据 | 英 | arxiv.org/abs/2204.10967 |
| **C-MTEB** (31 个中文测试集) | 嵌入模型选型评估 | 中 | BAAI |
| **NCRF++ 中文简历 NER** (~1000 标注简历) | 中文简历解析 | 中 | github.com/jiesutd/NCRFpp |
| **HuggingFace resume-job matching 模型** (如 jjbrosolo/bert-resume-matching) | 预训练匹配模型迁移 | 英 | huggingface.co/models?search=resume+job+matching |
| **HuggingFace resume NER 模型** | 实体抽取 | 英 | huggingface.co/models?search=resume+ner |
| **MTEB** (arXiv:2210.07316) / **MMTEB** (arXiv:2502.13595) | 嵌入模型基准选型 | 多语 | github.com/embeddings-benchmark/mteb |

### 3.1 验证中发现的重要纠偏（避免踩坑）

- ⚠️ 某些资料称 MTEB 含 "JobFit pair-classification 任务"——**经核查不实**，MTEB 仓库无 JobFit/resume 任务，不能用作现成匹配基准，需自建评估集。
- ⚠️ NCRF++ 仓库自带 sample_data 是 CoNLL 2003 英文 NER，**不是**中文简历；中文简历数据需另行获取（可基于 NCRF++ 工具自标或找衍生数据集）。
- ⚠️ OpenResume 仅支持**单栏英文**简历解析，不提供匹配评分——只贡献解析层。
- ⚠️ pyresparser 仅提取结构化字段（姓名/邮箱/技能/学历等），**无质量评分、无匹配、仅英文**。
- ⚠️ Resume-Matcher 的"匹配"实际是**关键词集合交集百分比**（非 embedding 余弦），且其简历评分功能在中文 README 中标注为**"开发中"**——不能直接拿来用，但其规则层（sections_preserved、jd_keywords_present 等确定性不变量）值得借鉴。

---

## 4. 模型选型

| 角色 | 选型 | 理由 |
|---|---|---|
| 嵌入（中英） | **BAAI/bge-m3** | 100+ 语言、8192 token、dense+sparse+ColBERT、MIT、C-MTEB 顶尖 |
| 精排交叉编码器 | **BAAI/bge-reranker-v2-m3** | 多语种、轻量、与 bge-m3 配套 |
| LLM judge | GLM-4.6（已有 ANTHROPIC_AUTH_TOKEN）/ Claude Sonnet 4.6 | 已有 GLM 接入；Claude 作高准确率备选 |
| 中文 NER | NCRF++ (CharLSTM+WordLSTM+CRF, F1 91.2) | 开源 SOTA、config 驱动 |
| 英文解析 | pyresparser + OpenResume | 成熟、可改造 |
| 编排 | LangChain | 模型无关，可换 embedding/LLM |

---

## 5. 达到 >80% 的关键工程手段

1. **Rubric-based LLM 评分**（Prometheus, arXiv:2310.05470）：给 LLM 明确分维度 rubric（如"量化指标：每段经历至少1个数字=满分"），而非开放式打分。这是从 LLM 拿到稳定分数的最大杠杆。
2. **Self-consistency**（arXiv:2203.11171）：同 prompt 采样 3 次取多数/均值，显著降方差。
3. **Peer rank / 多 judge 讨论**（PRD, arXiv:2304.02585）：高难样本多 judge 交叉。
4. **少量人工锚定集校准融合权重**：准备 50-100 条人工打分样本（覆盖 1-10 各档），用其拟合 fusion 权重并作为回归测试基线。这是把"无标注"变成"弱标注"的最低成本路径。
5. **合成数据自迭代**（Self-Taught Evaluator）：用 LLM 生成"好/差简历"对比对，蒸馏 judge，无需人工标签。
6. **规则层做硬约束兜底**：如联系方式缺失/经历段空白直接扣分，避免 LLM 漏判结构性硬伤。

---

## 6. 评估方法（无 ground truth 时）

- **人工抽样一致性**：50-100 条样本，3 位标注者打分，计算系统分与人工均值的 Spearman ρ（目标 ρ>0.8）和 ±1 分命中率（目标 >80%）。
- **二分类达标**：定义阈值（如质量≥6、匹配≥6 为"通过"），与人工共识比 F1。
- **跨语言一致性**：同一简历中英版本评分应接近（漂移检测）。
- **回归测试集**：锚定集纳入 CI，每次 prompt/权重变更跑回归，防退化。
- **LLM-judge 元评估**（arXiv:2411.15594）：用专门基准量化 judge 本身可靠性。

---

## 7. 评分维度细则（Rubric 草案）

### 7.1 简历质量评分（1-10）维度

| 维度 | 权重 | 满分要点 |
|---|---|---|
| 量化指标 | 0.25 | 经历描述含数字/百分比/规模，体现影响 |
| STAR 覆盖 | 0.20 | 情境-任务-行动-结果要素齐全 |
| 动词力度 | 0.10 | 使用强动词（led/owned/drove）而非弱动词（helped/worked） |
| 关键词覆盖 | 0.15 | 岗位相关技能词出现率 |
| 排版完整性 | 0.15 | 联系方式/教育/经历/技能段齐全、无空白 |
| 长度合理性 | 0.10 | 1-2 页、段落长度适中、无冗余 |
| 无硬伤 | 0.05 | 无错别字/时间断层/虚假信息迹象 |

### 7.2 简历-岗位匹配评分（1-10）维度

| 维度 | 权重 | 满分要点 |
|---|---|---|
| 硬技能匹配 | 0.35 | JD 必备技能在简历的命中与熟练度 |
| 软技能匹配 | 0.10 | JD 提到的软技能佐证 |
| 经验年限 | 0.20 | 达到 JD 要求年限比例 |
| 行业/领域 | 0.15 | 相关行业背景对齐 |
| 资历层级 | 0.10 | 级别（初级/资深/管理）与 JD 匹配 |
| 语义相关性 | 0.10 | BGE-M3+reranker 语义相似度 |

> 权重初值由人工锚定集回归校准。

---

## 8. 项目结构建议

当前仓库 `resume_analysis` 仅有 `main.py`/`pyproject.toml`。建议结构：

```
src/
  parsing/        # Layer 0: PDF→文本→结构化（中英）
  features/       # Layer 1: 规则特征与 rubric 打分
  matching/       # Layer 2: bge-m3 双塔 + reranker
  judging/        # Layer 3: LLM rubric 评分 + self-consistency
  fusion/         # 加权融合 + 校准
  eval/           # 锚定集 + 一致性评估
data/
  anchor_set/     # 50-100 人工样本
  datasets/       # PJF14/ESC 等
tests/
```

---

## 9. 实施阶段

| 阶段 | 内容 | 产出 |
|---|---|---|
| P0 | 解析层（中英 PDF→结构化） | `src/parsing/` 可用 |
| P1 | 规则层（rubric 特征 + 质量基线分） | 可解释 baseline 质量分 |
| P2 | ML 匹配层（bge-m3 双塔 + reranker） | 匹配 baseline 分 |
| P3 | LLM judge 层（rubric prompt + self-consistency） | 高准确率质量/匹配分 |
| P4 | 融合校准（50-100 锚定集拟合权重） | 融合分 |
| P5 | 评估闭环（一致性 + 二分类 + 回归 CI） | >80% 验证报告 |

先把规则 + ML 跑通拿到 baseline，再叠加 LLM 提上限，最后用锚定集校准融合权重并完成评估闭环。

---

## 10. 参考来源

**学术**
- Self-Taught Evaluator — arxiv.org/abs/2408.02831
- A Survey on LLM-as-a-Judge — arxiv.org/abs/2411.15594
- LLM-as-a-Judge Literature Survey (2025) — arxiv.org/abs/2503.01745
- Benchmarking LLM-as-a-Judge for Document-Level Eval — arxiv.org/abs/2411.04214
- LLM-as-a-Judge: Biases, Calibration, Meta-Eval — arxiv.org/abs/2406.12641
- Prometheus (rubric judge) — arxiv.org/abs/2310.05470
- Prometheus 2 — arxiv.org/abs/2405.01535
- Self-Consistency — arxiv.org/abs/2203.11171
- PRD Peer Rank — arxiv.org/abs/2304.02585
- BGE-M3 — arxiv.org/abs/2402.03216
- Sentence-BERT — arxiv.org/abs/1908.10084
- Joint Repr. Learning for Person-Job Fit (PJF14) — aclanthology.org/2022.acl-long.314/
- ESC Skill Corpus — aclanthology.org/2022.lrec-1.250/
- Contrastive resume-job matching — arxiv.org/abs/2204.10967
- MTEB — arxiv.org/abs/2210.07316
- MMTEB — arxiv.org/abs/2502.13595

**开源实现**
- sentence-transformers — github.com/UKPLab/sentence-transformers
- FlagEmbedding (BGE) — github.com/FlagOpen/FlagEmbedding
- BAAI/bge-m3 — huggingface.co/BAAI/bge-m3
- Resume-Matcher — github.com/srbhr/Resume-Matcher
- OpenResume — github.com/xitanggg/open-resume
- pyresparser — github.com/OmkarPathak/pyresparser
- NCRF++ — github.com/jiesutd/NCRFpp
- PaddleNLP text-matching — github.com/PaddlePaddle/PaddleNLP
- LangChain — github.com/langchain-ai/langchain

---

## 附录 A. 数据集可用性核实（2026-06 实测）

对每个候选数据集做了真实可下载性验证（GitHub API / Kaggle API / curl HTTP 状态码）。

### 总体结论

- ✅ **简历文本/解析**类数据充足（中英都有，可下载）
- ✅ **简历-JD 匹配**类有学术框架提供（RecBole-PJF），但需去 TIANCHI/Kaggle 下载原始文件再跑预处理脚本
- ❌ **简历质量评分**无任何公开带分数据集 —— 最大缺口，必须靠 rubric + LLM 合成自标
- ⚠️ HuggingFace 上的若干 "resume-job-matching" 数据集因网络拦截未能直连核实，WebSearch 给出的名称（crcode/、AIAgents42/、casszhao/ 等）**未经证实**，需手动到 huggingface.co 搜索确认

### 已验证可用

| 数据集 | 规模 | 语言 | 用途 | 可达性 |
|---|---|---|---|---|
| Kaggle: snehaanbhawal/resume-dataset | 65MB, ~2400 份 | 英 | LiveCareer 简历，24 职业类别 | Kaggle API ✅ |
| Kaggle: suriyaganesh/resume-dataset-structured | 39MB, 54k 份 | 英 | 结构化简历字段 | Kaggle API ✅ |
| Kaggle: rhythmghai/resume-screening-dataset-200k-candidates | 3.7MB, 20万 | 英 | 候选人筛选含匹配标签 | Kaggle API ✅ |
| Kaggle: 另有 5+ 个小数据集 | KB~MB | 英 | 各类辅助 | Kaggle API ✅ |
| GitHub: florex/resume_corpus | 89MB+63MB | 英 | 多标签职业分类简历+skills词表，有同行评议 | GitHub 直下 ✅ |
| GitHub: fengyh3/Chinese_Resume_NER | train 1.3MB (BMES字级标注) | 中 | 中文简历NER | GitHub 直下 ✅ |
| GitHub: RUCAIBox/RecBole-PJF (70★) | 框架+预处理脚本 | 中+英 | person-job fit 交互+文本；zhilian(TIANCHI 31623)+kaggle job-recommendation | 需下原始文件跑脚本 ✅ |
| C-MTEB / MTEB / MMTEB | 基准 | 中+多语 | 嵌入模型选型评估（非训练数据） | 公开 ✅ |

### 不可用 / 待核实

- **DPGNN / SHPJF / JRMPM**：仅代码，data 目录为空，须自备数据，clone 不可得数据。
- **NCRF++**：工具可用，但自带 sample_data 是 CoNLL 英文 NER，非中文简历。
- **OpenResume / pyresparser / Resume-Matcher**：是工具非数据集。
- **HuggingFace 数据集**（crcode/、AIAgents42/、casszhao/、AlgoliaEscalation/）：未核实，疑似部分为幻觉，需本地实查。

### 关键缺口与对策

1. 简历质量分无公开数据 → rubric + LLM-as-judge 对简历语料自标 + 50 条人工锚定集校准。
2. 中英匹配数据 → RecBole-PJF (zhilian 中 + kaggle 英)；不足部分用 LLM 合成 JD-简历对。
3. 中文解析 → fengyh3/Chinese_Resume_NER 微调中文 NER。
4. 模型选型评估 → C-MTEB 量化中文嵌入质量。

### 冷启动组合

- 质量评分训练/校准：Kaggle snehaanbhawal + florex/resume_corpus（英）+ 自建中文小样本 → LLM rubric 自标 + 50 条人工锚定
- 匹配评分训练：RecBole-PJF (zhilian+kaggle) 对比学习微调 bge-m3；不足用 LLM 合成对
- 中文解析：fengyh3/Chinese_Resume_NER
- 模型选型评估：C-MTEB
