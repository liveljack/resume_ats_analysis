# 模型下载指南：bge-m3 / bge-reranker-v2-m3（高精度匹配后端）

本系统的匹配层有三个后端（`src/matching/embedder.py`）：

| 后端 | `MATCH_BACKEND` | 精度 | 体积 | 依赖 |
|---|---|---|---|---|
| TF-IDF（默认 fallback） | `tfidf` | 低 | 0 | 无 |
| Sentence-Transformers | `sentence-transformers` | 中 | ~470MB | `sentence-transformers` |
| **BGE-M3（设计目标）** | `bge-m3` | **高** | ~2.3GB | `FlagEmbedding` |

要达到设计目标的 **>80% 跨域匹配准确率**，必须启用 `bge-m3`（语义嵌入能区分「厨师 vs 律师」这类词法匹配器无法区分的语义近邻类）。本指南讲解如何下载模型并激活。

---

## 0. 为什么需要这篇指南

`huggingface.co` 在中国大陆及部分网络环境**无法直接访问**（本机测试 HTTP 000）。但有两个镜像可达：

| 镜像 | 域名 | 本机测试 | 说明 |
|---|---|---|---|
| **ModelScope 魔搭** | modelscope.cn | ✅ HTTP 200 | 阿里达摩院，BAAI 模型官方镜像，**中国大陆首选** |
| **hf-mirror** | hf-mirror.com | ✅ HTTP 200 | HuggingFace 全量镜像，`HF_ENDPOINT` 指向即可 |
| HuggingFace 官方 | huggingface.co | ❌ HTTP 000 | 需科学上网 |

> 推荐 **ModelScope**（国内最快、bge 系列由 BAAI 官方同步）。下面的「方式一」是首选。

---

## 需要下载的模型

| 模型 | 用途 | 体积 | 默认配置变量 |
|---|---|---|---|
| `BAAI/bge-m3` | 双塔嵌入（中英 100+ 语种，8192 token） | ~2.3GB | `BGE_M3_MODEL_NAME` |
| `BAAI/bge-reranker-v2-m3` | 交叉编码器精排（可选，提升上限） | ~2.3GB | `RERANKER_MODEL_NAME` |
| `paraphrase-multilingual-MiniLM-L12-v2` | 轻量替代（ST 后端，先试水用） | ~470MB | `ST_MODEL_NAME` |

> 先下 `bge-m3` 即可跑通；reranker 是可选的精排增强。

---

## 方式一：ModelScope（推荐，中国大陆可达）

### 1) 一键脚本（本项目已提供）

```bash
# 下载 bge-m3 + reranker 到 ./models/
python scripts/download_models.py

# 或只下 bge-m3
python scripts/download_models.py --only bge-m3 --source modelscope
```

脚本会自动 `pip install modelscope`，调用 `snapshot_download` 把模型存到 `./models/bge-m3/` 与 `./models/bge-reranker-v2-m3/`，并在结束时打印要导出的环境变量。

### 2) 手动命令（等价操作）

```bash
# 安装 modelscope
uv pip install modelscope      # 或 pip install modelscope

# 下载到本地目录
python -c "
from modelscope import snapshot_download
snapshot_download('BAAI/bge-m3', local_dir='./models/bge-m3')
snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='./models/bge-reranker-v2-m3')
"
```

### 3) 命令行（modelscope CLI）

```bash
pip install modelscope
modelscope download --model BAAI/bge-m3 --local_dir ./models/bge-m3
```

---

## 方式二：hf-mirror.com（HuggingFace 镜像）

设置 `HF_ENDPOINT` 后，所有 `huggingface_hub` / `huggingface-cli` 命令都会走镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 安装 huggingface_hub
uv pip install huggingface_hub

# 方式 A：Python
python -c "
import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download(repo_id='BAAI/bge-m3', local_dir='./models/bge-m3')
"

# 方式 B：CLI
pip install -U huggingface_hub
huggingface-cli download BAAI/bge-m3 --local-dir ./models/bge-m3
```

本项目脚本也支持：`python scripts/download_models.py --source hf-mirror`

---

## 方式三：HuggingFace 官方（需能直连 huggingface.co）

适合有科学上网或身处可直连 HF 的地区：

```bash
uv pip install huggingface_hub
huggingface-cli login          # 公开模型可不登录，私有模型需 token
huggingface-cli download BAAI/bge-m3 --local-dir ./models/bge-m3
```

或用 `git clone`（需安装 [git-lfs](https://git-lfs.com)）：

```bash
brew install git-lfs && git lfs install
git clone https://huggingface.co/BAAI/bge-m3 ./models/bge-m3
```

---

## 激活：让代码用本地模型

**代码已支持「模型名 = 本地路径」**，无需改代码，只需设环境变量：

```bash
# 安装重依赖（FlagEmbedding 含 torch）
uv sync --extra embedding

# 指向本地模型目录 + 切换后端
export MATCH_BACKEND=bge-m3
export BGE_M3_MODEL_NAME="$(pwd)/models/bge-m3"
export RERANKER_MODEL_NAME="$(pwd)/models/bge-reranker-v2-m3"   # 可选

# 验证加载
python -c "from src.matching.embedder import get_embedder; e=get_embedder('bge-m3'); print(e.backend, e.embed(['你好','hello']).shape)"
```

> 关键点：`BGEM3FlagModel(local_path)` 与 `SentenceTransformer(local_path)` 都接受本地目录，所以把 `BGE_M3_MODEL_NAME` 指向 `./models/bge-m3` 即可，代码会自动从本地加载、不再访问网络。

---

## 验证端到端效果

下载并设置环境变量后，重跑之前在 tfidf 下不达标的跨域测试，应越过 80%：

```bash
export MATCH_BACKEND=bge-m3
export BGE_M3_MODEL_NAME="$(pwd)/models/bge-m3"

# 之前 tfidf：37.6% → 语义嵌入通常 80%+
python scripts/occupation_match_eval.py --source livecareer --top-k 24 --per-occ 40

# flox 8 类：之前 78.5% → 应 >85%
python scripts/occupation_match_eval.py --source florex --top-k 12 --per-occ 50 --max-total 20000

# 演示
python main.py demo
```

---

## 轻量替代：先试 Sentence-Transformers（470MB）

如果只想先快速验证语义嵌入的收益、不想下 2.3GB，可用 ST 多语种小模型：

```bash
uv sync --extra embedding
export HF_ENDPOINT=https://hf-mirror.com     # 若 HF 不可达
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').save('./models/st-multilingual')
"
export MATCH_BACKEND=sentence-transformers
export ST_MODEL_NAME="$(pwd)/models/st-multilingual"
python scripts/occupation_match_eval.py --source livecareer --top-k 24 --per-occ 40
```

---

## 常见问题

| 问题 | 解决 |
|---|---|
| `huggingface.co` 连接超时 | 用方式一（ModelScope）或方式二（hf-mirror），不要直连 HF |
| `modelscope` 下载慢/中断 | `snapshot_download` 自带断点续传，重跑即可；或换 hf-mirror |
| `No module named FlagEmbedding` | `uv sync --extra embedding`（会装 torch，约 800MB） |
| 显存不足 / CPU 慢 | bge-m3 在 CPU 可跑（慢）；或用轻量 ST 方案；`use_fp16` 在 CPU 会自动关 |
| 磁盘空间 | bge-m3 + reranker 共 ~5GB，确保 `./models/` 所在盘有空间 |
| 想离线部署到无网机器 | 在有网机器下到 `./models/`，整个目录拷过去，设 `BGE_M3_MODEL_NAME` 指向即可 |

---

## 一句话流程

```bash
python scripts/download_models.py                 # 1. 下载（ModelScope 镜像）
uv sync --extra embedding                          # 2. 装重依赖
export MATCH_BACKEND=bge-m3                        # 3. 切后端
export BGE_M3_MODEL_NAME="$(pwd)/models/bge-m3"    # 4. 指本地路径
python scripts/occupation_match_eval.py --source livecareer --top-k 24 --per-occ 40   # 5. 验证 >80%
```
