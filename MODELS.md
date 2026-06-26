# 模型详解：BAAI/bge-m3 与 BAAI/bge-reranker-v2-m3

> 本系统匹配层的设计目标后端。两者都是开源（MIT 许可）、可商用，基于同一 XLM-RoBERTa-large 骨架，但**职责不同、配合使用**（两阶段 Retrieve & Re-Rank）。

---

## 1. 两个模型是什么

### BAAI/bge-m3 —— 双塔编码器（Bi-Encoder）

- **架构**：`XLMRobertaModel`（纯编码器，输出向量）
- **职责**：把任意文本编码成 1024 维稠密向量（同时还能输出 sparse 词权重向量、ColBERT 多向量）。简历和 JD **分别**编码后算余弦相似度。
- **核心能力**：
  - **多语种**：100+ 语言（中、英、日、韩…），跨语言检索无需为每种语言单独训模型。
  - **长文本**：最大 8192 token（一份完整简历或 JD 可一次性编码，无需切片）。
  - **多功能**：dense（稠密语义）/ sparse（词法关键词）/ ColBERT（细粒度交互）三路向量同模型产出，可混合检索。
- **为什么适合简历匹配**：简历-JD 是跨语言（中英混合）+ 长文本 + 语义相似任务，正好踩在 bge-m3 的三个强项上。C-MTEB/MTEB 多语种检索榜 SOTA。

### BAAI/bge-reranker-v2-m3 —— 交叉编码器（Cross-Encoder）

- **架构**：`XLMRobertaForSequenceClassification`（编码器 + 一个分类头）
- **职责**：输入**一个 (query, passage) 文本对**（如 (JD, 简历)），输出一个相关性分数。简历和 JD **拼在一起**进模型，做深层交互，精度比双塔高，但**无法预计算**、每次都要跑一对。
- **核心能力**：多语种、轻量、部署快、推理相对交叉编码器里算快的。
- **用途**：只对双塔召回的 top-K（如 top-50）做精排，不用于全量。

### 两者关系（两阶段 Retrieve & Re-Rank，sbert.net 推荐架构）

```
全部简历 (N 万)
   │  bge-m3 双塔：JD 向量 vs 预计算的简历向量，cosine 召回
   ▼
top-50 候选
   │  bge-reranker-v2-m3 交叉编码器：(JD, 简历) 逐对精排
   ▼
top-10 精排结果 + 最终匹配分
```

- **双塔（bge-m3）**：快、可预计算简历向量入库、支持海量召回。精度中等。
- **交叉编码器（reranker）**：慢、不可预计算，但精度高。只用在少量候选上。
- 二者结合 = 速度 + 精度。对应本系统 `src/matching/matcher.py` 的 `MatchScorer`（双塔 `score` + 可选 `Reranker.score`）。

---

## 2. 技术规格（取自模型 config.json 实测）

| 指标 | bge-m3 | bge-reranker-v2-m3 |
|---|---|---|
| 骨架 | XLM-RoBERTa-large | XLM-RoBERTa-large |
| 架构 | XLMRobertaModel | XLMRobertaForSequenceClassification |
| 参数量 | **~5.68 亿** | **~5.68 亿** |
| hidden_size | 1024 | 1024 |
| num_hidden_layers | 24 | 24 |
| num_attention_heads | 16 | 16 |
| intermediate_size | 4096 | 4096 |
| vocab_size | 250,002 | 250,002 |
| max_position_embeddings | 8194（**8192 token**） | 8194 |
| 输出 | 1024 维 dense（+sparse +ColBERT） | 单个相关性分数 |
| 输入 | 单文本 | 文本对 (query, passage) |

参数量推导：权重文件 `pytorch_model.bin` / `model.safetensors` 均为 **2,271,145,830 字节** ≈ 2.27GB（fp32）→ 2.27e9 / 4 = **5.68 亿参数**，与 XLM-RoBERTa-large 一致。

### 权重文件体积（实测 Content-Length）

**bge-m3 仓库：**
| 文件 | 体积 | 是否必需 |
|---|---|---|
| `pytorch_model.bin` | 2.27 GB | ✅ PyTorch 推理必需 |
| `onnx/model.onnx_data` | 2.27 GB | ❌ 仅 ONNX Runtime 部署才需要 |
| `onnx/model.onnx` | 0.7 MB | ❌ 同上 |
| `colbert_linear.pt` | 2.1 MB | ✅ 用 ColBERT 多向量检索时需要 |
| `sparse_linear.pt` | 3.5 KB | ✅ 用 sparse 词法检索时需要 |
| tokenizer 等 | ~16 MB | ✅ |

- **最小下载（仅 PyTorch + dense）**：~2.3 GB
- **完整仓库（含 ONNX）**：~4.6 GB（ONNX 与 pytorch 是同一份权重的两种格式，**重复**，按需取一）

**bge-reranker-v2-m3 仓库：**
| 文件 | 体积 | 是否必需 |
|---|---|---|
| `model.safetensors` | 2.27 GB | ✅ |
| tokenizer 等 | ~16 MB | ✅ |
- **最小下载**：~2.3 GB

> ⚠️ 当前 `scripts/download_models.py` 走 ModelScope 默认拉全量（含 ONNX），bge-m3 会下到 ~4.6GB。若只用 PyTorch，可跳过 ONNX 省 2.27GB（见文末优化建议）。

---

## 3. 部署配置需求

### 3.1 内存/显存占用估算

权重内存（单模型）：

| 精度 | 权重内存 | 说明 |
|---|---|---|
| fp32 | 2.27 GB | 默认；CPU 推理常用 |
| fp16 | 1.13 GB | GPU 推荐；`use_fp16=True` |
| int8 量化 | ~0.57 GB | 极致省显存（需额外量化工具） |

推理峰值 = 权重 + 激活。激活主要来自 **8192 token 长序列的自注意力**（O(n²)）：
- 8192 token 朴素注意力 = 24层×16头×8192² ≈ 257 亿元素 → 朴素实现会爆显存。
- FlagEmbedding/torch 2.x 用 **FlashAttention / 分块**避免完整注意力矩阵，8192 token 实际峰值额外 ~2-4GB（fp16）。
- 简历/JD 通常 1000-3000 token，峰值更小。

### 3.2 三档部署配置推荐

#### 档位 A：CPU-only（开发/低频离线批处理）
- **内存**：16 GB RAM（最低 8 GB，两个模型 fp32 同时驻留 ≈ 4.5GB + 激活 + 系统）
- **CPU**：8 核以上
- **磁盘**：20 GB 空闲（模型 4.6GB + torch 依赖 ~3GB + 数据）
- **GPU**：无
- **速度**：bge-m3 短文本 ~0.1-0.5s/条，8192 token 长文本 ~2-5s/条；reranker 逐对 ~0.3-1s/对
- **适用**：本项目的批量评估、小规模简历筛选、CI 测试

#### 档位 B：单卡 GPU（生产推荐，性价比最优）
- **显存**：**8 GB VRAM**（最低）/ 12-16 GB（推荐）
- **典型卡**：RTX 3060 12GB / RTX 4060 8GB / **T4 16GB** / A10 24GB
- **内存**：16 GB RAM
- **磁盘**：20 GB
- **部署**：两模型同时 fp16 加载（2×1.13GB=2.26GB 权重），剩 ~6-14GB 给 8192 上下文 + 批处理
- **速度**：bge-m3 短文本 ~数百条/秒，长文本 ~数十条/秒；reranker top-50 精排 <1s
- **适用**：中小流量在线服务、每日处理数千-数万简历

#### 档位 C：高吞吐生产
- **显存**：24-40 GB VRAM
- **典型卡**：RTX 3090/4090 24GB / A10G 24GB / **A100 40GB**
- **内存**：32 GB RAM
- **部署**：两模型 fp16 + 大 batch + 长上下文 + 高并发
- **速度**：bge-m3 上千条/秒，reranker 并发精排
- **适用**：大型招聘平台级流量

### 3.3 一句话选型

| 场景 | 推荐 |
|---|---|
| 本地开发/跑评估脚本 | **CPU 16GB RAM** 即可（慢但能跑） |
| 想免费验证 GPU 效果 | **Google Colab T4 16GB**（免费） |
| 自建生产服务器（性价比） | **1× T4 16GB** 或 **RTX 3060 12GB**（~$200-300） |
| 高并发生产 | **A10 24GB** 或 **A100 40GB** |

### 3.4 两个模型同时部署的总资源

| 资源 | CPU-only | GPU 8GB | GPU 16GB | GPU 24GB |
|---|---|---|---|---|
| 权重驻留 (fp16) | 2.26 GB RAM | 2.26 GB VRAM | 2.26 GB VRAM | 2.26 GB VRAM |
| 权重驻留 (fp32) | 4.54 GB RAM | — | — | — |
| 推理峰值 (8192 ctx) | +2-4 GB | +2-4 GB | +2-4 GB | +2-4 GB |
| **总需求** | **8-16 GB RAM** | **8 GB VRAM** ⚠️紧凑 | **16 GB VRAM** ✅舒适 | **24 GB VRAM** 宽裕 |

> 8GB VRAM 能装下两模型 fp16，但 8192 长上下文 + 大 batch 会紧张；12-16GB 是舒适区。

---

## 4. 推理速度参考（量级估算）

| 操作 | CPU | T4 16GB GPU |
|---|---|---|
| bge-m3 编码短文本（<512 token） | ~0.1-0.5 s | ~5-20 ms |
| bge-m3 编码 8192 token 长文档 | ~2-5 s | ~0.1-0.3 s |
| reranker 单对 (JD, 简历) | ~0.3-1 s | ~20-50 ms |
| reranker top-50 精排 | ~15-50 s | ~1-3 s |

> 简历匹配场景：简历向量可**预计算入库**（一次性），在线只需算 1 个 JD 向量 + 余弦，毫秒级；reranker 只对 top-K 跑。

---

## 5. 优化建议

1. **跳过 ONNX 省一半磁盘/下载时间**：PyTorch 推理只需 `pytorch_model.bin`，`scripts/download_models.py` 可加 `allow_patterns` 过滤掉 `onnx/*`（bge-m3 从 4.6GB → 2.3GB）。
2. **fp16**：GPU 上务必 `use_fp16=True`，显存减半、速度翻倍，精度损失可忽略。
3. **简历向量预计算**：用 bge-m3 把全部简历向量算一次存入向量库（FAISS/Chroma），在线只算 JD 向量做 ANN 检索 → 毫秒级召回万级简历。
4. **reranker 只用 top-K**：交叉编码器贵，只对双塔召回的 top-50 跑。
5. **CPU 部署用 ONNX Runtime**：若纯 CPU 生产，可加载 onnx 版本 + int8 量化，速度比 PyTorch fp32 快 2-4 倍。
6. **长文本切分**：虽支持 8192，但超长简历可按段编码后均值池化，省显存。

---

## 6. 许可与商用

- 两个模型均 **MIT 许可**，可免费商用。
- 依赖 `FlagEmbedding`（MIT）与 `torch`（BSD）。
- 无 API 调用费用（本地推理）。

---

## 7. 在本项目中激活

```bash
# 1. 下载（ModelScope 镜像，跳过 ONNX 见优化建议）
python scripts/download_models.py

# 2. 装依赖
uv sync --extra embedding

# 3. 激活（指向本地目录 + 切后端）
export MATCH_BACKEND=bge-m3
export BGE_M3_MODEL_NAME="$(pwd)/models/bge-m3"
export RERANKER_MODEL_NAME="$(pwd)/models/bge-reranker-v2-m3"

# 4. 验证
python -c "from src.matching.embedder import get_embedder; e=get_embedder('bge-m3'); print(e.embed(['机器学习工程师','ML engineer']).shape)"

# 5. 跑跨域准确率（tfidf 下 37.6% → 语义嵌入通常 80%+）
python scripts/occupation_match_eval.py --source livecareer --top-k 24 --per-occ 40
```

详见 [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md)（下载方法）。
