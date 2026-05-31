---
stage: 9
title: "毕业项目 · 端到端 AI 产品"
level: "综合"
duration: "3~4 周"
goal: "独立完成一个打通数据→RAG/Agent→评估→部署→监控全链路的真实 AI 产品，并能在简历与面试中讲清楚"
prereq: "完成 stage-01~08 全部内容"
stack: ["Python", "FastAPI", "向量数据库", "LangGraph/自研编排", "Langfuse", "Docker", "pytest"]
---

## 第 1 节：如何选题——真实价值优先，而非技术炫技

### 📖 知识点

毕业项目（Capstone）的第一道、也是最容易翻车的坎，是**选题**。绝大多数人会犯两个相反的错误：要么选一个"又一个 ChatGPT 套壳聊天机器人"，毫无辨识度；要么贪大求全，想做一个"颠覆行业的 AI 平台"，三周后烂尾。好的 Capstone 选题遵循一个核心原则——**真实价值优先**：它要解决一个**具体的人**在**具体场景**下的**真实痛点**，哪怕这个痛点很小。

怎么判断一个选题靠不靠谱？用四个问题筛：

- **真实性（Real）**：这个痛点是真的吗？最好你自己或身边人就是用户。"我每周要读 30 篇 arXiv 论文，想要个能按我研究方向筛选+摘要+追问的助手"——这是真痛点；"做一个万能 AI 助手"——这是伪需求。
- **AI 必要性（AI-native）**：这个问题**非 LLM 不可**吗？如果用传统规则/搜索就能解决，那 AI 只是噱头。要选那些**涉及非结构化文本理解、需要推理或多步规划**的问题——这正是 LLM 不可替代的地方。
- **可行性（Feasible）**：数据拿得到吗？三四周做得完吗？要能切出一个"窄而深"的最小闭环，而不是"宽而浅"的半成品。
- **可展示性（Demoable）**：能不能跑给别人看、并讲清楚价值？面试官没耐心看你解释，要让产品自己说话。

> **专家心法**：评判 Capstone 价值的不是"用了多少种时髦技术"，而是"有没有人愿意用第二次"。一个能让一个真实用户每天打开的小工具，远胜过一个技术堆砌但没人用的"大平台"。窄而深，永远优于宽而浅。

### 💻 代码示例

选题不是写代码，但你可以用一份**结构化的选题评估表**强迫自己想清楚。把它当成项目的第一个产出物。

```python
# select_topic.py —— 用打分卡量化对比候选选题，避免凭感觉拍板
from dataclasses import dataclass

@dataclass
class TopicScore:
    name: str
    real_pain: int      # 真实痛点强度 1-5（你/身边人是否真受其苦）
    ai_native: int      # 非 LLM 不可的程度 1-5（传统方法能否替代）
    feasible: int       # 三四周可完成度 1-5（数据/范围/复杂度）
    demoable: int       # 可展示与讲述性 1-5

    @property
    def total(self) -> float:
        # ai_native 与 real_pain 加权更高：决定项目"有没有意义"的是它们
        return self.real_pain * 1.5 + self.ai_native * 1.5 + self.feasible + self.demoable


candidates = [
    TopicScore("个人 arXiv 论文追踪+问答助手", real_pain=5, ai_native=5, feasible=4, demoable=5),
    TopicScore("万能 AI 客服平台",            real_pain=3, ai_native=3, feasible=2, demoable=3),
    TopicScore("企业制度文档智能问答",          real_pain=4, ai_native=4, feasible=4, demoable=4),
]

for t in sorted(candidates, key=lambda x: x.total, reverse=True):
    print(f"{t.total:5.1f}  {t.name}")
# 输出按总分排序，最高分通常就是你该做的那个
```

### 📚 推荐资料

- [Y Combinator: How to Get Startup Ideas (Paul Graham)](https://www.paulgraham.com/startupideas.html) — "做你自己想要的东西"，选题第一原则的源头。
- [Emerging Architectures for LLM Applications (a16z)](https://a16z.com/emerging-architectures-for-llm-applications/) — 看清 LLM 应用的典型形态，帮你判断选题落在哪类架构。
- [Build a Large Language Model Application (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/) — 大量真实小项目案例，激发选题灵感。

### 🛠 动手实践

列出 3 个你真实想做的候选选题，用上面的打分卡逐项打分并写下理由。然后为最高分的那个写一段 100 字的"电梯演讲"：用户是谁、痛点是什么、你的产品如何解决、为什么非 AI 不可。验收标准：你能向一个非技术朋友讲清楚这个产品，并让他说出"这个我可能会用"。

## 第 2 节：端到端架构设计——把前 8 个阶段串成一条链

### 📖 知识点

Capstone 的技术内核，是把你过去 8 个阶段学的零件组装成一个**完整的数据流闭环**。一个生产级 AI 产品的标准链路是：

**数据上游 → 索引/检索（RAG）→ 编排/推理（Agent）→ 评估 → 部署 → 监控**，且监控的反馈要能回流去改进前面每一环。

逐段拆解它们如何咬合：

- **数据上游（stage 数据工程）**：数据从哪来、怎么清洗、怎么切分（chunking）。这是 RAG 质量的天花板——**垃圾进，垃圾出**。文档解析、分块策略、元数据抽取都在这层决定。
- **索引/检索（stage RAG）**：embedding 模型选型、向量库、检索策略（向量+关键词混合检索、rerank）。检索召回不到，后面再强的模型也白搭。
- **编排/推理（stage Agent）**：是简单的"检索+生成"，还是需要多步规划、调工具、循环反思的 Agent？复杂度按需选择，**别为了用 Agent 而用 Agent**。
- **评估（stage-07）**：建立离线评估集，定义质量指标（忠实度 faithfulness、相关性、答案正确率），作为迭代的标尺和 CI 门禁。
- **部署（stage-08）**：FastAPI 服务化 + Docker 容器化。
- **监控（stage-08）**：Langfuse 追踪 + 成本/护栏，线上反馈回流。

设计架构时最重要的判断是**复杂度匹配**：从最简单的能跑通的链路（朴素 RAG）起步，**只在评估数据告诉你"这里不够好"时才加复杂度**（加 rerank、加 Agent、加多轮）。这叫"演进式架构"，是对抗过度工程的唯一良药。

> **专家心法**：先画数据流图，再写一行代码。把"一条数据/一个请求"从入口到出口走一遍，在每个箭头上问"这一步可能怎么失败、怎么测、怎么观测"。架构图画得清楚的人，代码通常也写得干净——因为他想清楚了边界。

### 💻 代码示例

把整条链路用接口（Protocol）先定义出来，让架构边界显式、各模块可独立替换与测试。这是"面向接口编程"在 AI 系统里的体现。

```python
# architecture.py —— 先定义契约，再填实现。每一层都可独立 mock 与替换
from typing import Protocol, runtime_checkable
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    metadata: dict          # 来源、页码、时间戳等，溯源与过滤都靠它

@dataclass
class Answer:
    text: str
    sources: list[Chunk]    # 答案必须可溯源，这是 RAG 产品可信度的关键
    trace_id: str           # 关联 Langfuse trace，打通监控

@runtime_checkable
class Retriever(Protocol):
    """检索层契约：可换成向量检索、混合检索、带 rerank 的检索。"""
    def search(self, query: str, top_k: int = 5) -> list[Chunk]: ...

@runtime_checkable
class Generator(Protocol):
    """生成层契约：可换模型、换 Prompt、换成多步 Agent。"""
    def generate(self, query: str, context: list[Chunk]) -> str: ...

class RAGPipeline:
    """编排层：把各层组装起来。依赖的是接口而非具体实现，便于测试与演进。"""
    def __init__(self, retriever: Retriever, generator: Generator, guardrail, tracer):
        self.retriever = retriever
        self.generator = generator
        self.guardrail = guardrail      # stage-08 的护栏
        self.tracer = tracer            # stage-08 的可观测性

    def answer(self, query: str) -> Answer:
        ok, reason = self.guardrail.check_input(query)
        if not ok:
            return Answer(text=f"请求被拦截：{reason}", sources=[], trace_id="")
        with self.tracer.trace(query) as t:          # 监控贯穿全程
            chunks = self.retriever.search(query)     # 数据/检索层
            raw = self.generator.generate(query, chunks)  # 推理层
            safe = self.guardrail.mask_pii(raw)           # 输出护栏（方法名与 stage-08 对齐）
            return Answer(text=safe, sources=chunks, trace_id=t.id)
```

### 📚 推荐资料

- [Emerging Architectures for LLM Applications (a16z)](https://a16z.com/emerging-architectures-for-llm-applications/) — 业界 LLM 应用参考架构，看清各层职责。
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 用图来编排有状态、多步、可循环的 Agent，复杂编排首选。
- [Architecture Decision Records (ADR)](https://adr.github.io/) — 把"为什么这样选型"记录下来的轻量方法，面试讲项目时极有用。
- [What is RAG? (AWS)](https://aws.amazon.com/what-is/retrieval-augmented-generation/) — RAG 链路各环节的清晰梳理。

### 🛠 动手实践

为你的 Capstone 画一张端到端架构图（用 Excalidraw / Mermaid 均可），标出六个环节、数据在每个箭头上的形态、以及每一环的失败模式与观测点。再用 Protocol/接口把核心模块的契约写出来（先不实现）。验收标准：架构图能让别人在 2 分钟内看懂你的数据如何流动，且接口定义清晰到可以分给不同人并行开发。

## 第 3 节：技术选型决策——在约束下做权衡，并记录理由

### 📖 知识点

工程的本质是**在约束下做权衡**，技术选型最能体现一个工程师的成熟度。Capstone 里你会面对一连串选择：用哪个模型？哪个向量库？要不要上 Agent 框架？自研还是用 LangChain？这里没有标准答案，**只有针对你的约束（成本、延迟、数据隐私、团队熟悉度、可维护性）的合理权衡**。

几个高频决策点和判断框架：

- **模型选型**：闭源 API（GPT/Claude，质量高、省心、按量付费、数据出境）vs 开源自托管（Llama/Qwen，可私有化、固定成本、需运维 GPU）。判断锚点：有没有数据隐私硬要求？流量规模够不够摊薄自托管成本？
- **向量库**：原型期 Chroma/FAISS（本地、零成本）；要持久化与过滤上 pgvector（复用 Postgres，运维简单）；超大规模才考虑 Milvus/专用云服务。**别一上来就上重型方案**。
- **是否用 Agent 框架**：LangChain/LlamaIndex 抽象多、起步快但黑盒、调试难；自己用原生 SDK + 少量胶水代码，可控、透明但要多写代码。**学习/小项目建议至少自己手写一遍核心循环**，理解原理后再决定是否用框架。
- **编排复杂度**：能用朴素 RAG 解决就别上 Agent；能用单次调用解决就别上多轮。

关键不在于"选了什么"，而在于**你能否说清"为什么这么选、放弃了什么、什么情况下你会改选"**。这就是 ADR（架构决策记录）的价值，也是面试官真正想听的。

> **专家心法**：面试时被问"你为什么用 X"，最差的回答是"因为大家都用"或"教程里是这么写的"。最好的回答是："在我的约束下，X 和 Y 我都评估过，X 在 A 维度更优、代价是 B，而 B 对我的场景影响小——如果未来流量涨 10 倍，我会重新评估切到 Y。" 选型的深度，藏在你被放弃的那个选项里。

### 💻 代码示例

把选型决策写成机器可读、可对比、可追溯的 ADR。代码项目里建一个 `docs/adr/` 目录长期维护它。

```python
# adr.py —— 用代码结构表达"架构决策记录"，强迫自己写清权衡
from dataclasses import dataclass, field

@dataclass
class ADR:
    id: int
    title: str
    context: str                       # 我面临什么问题、有哪些约束
    decision: str                      # 我决定怎么做
    alternatives: list[str]            # 我评估过、但放弃的选项
    consequences: str                  # 这个决定的代价与风险
    revisit_when: str                  # 什么信号出现时我会重新评估

adr_vectorstore = ADR(
    id=1,
    title="向量库选用 pgvector 而非 Milvus",
    context="个人项目，文档约 2 万 chunk，需按 metadata 过滤，希望运维成本极低。",
    decision="复用一个 Postgres 实例 + pgvector 扩展，检索与业务数据同库。",
    alternatives=["Chroma（无法持久化与复杂过滤）", "Milvus（功能强但运维重，规模用不上）"],
    consequences="单机性能上限较低；若 chunk 数破百万需迁移。换来的是零额外运维。",
    revisit_when="文档量级超过 50 万 chunk，或 P95 检索延迟 > 200ms。",
)

def render(adr: ADR) -> str:
    return (f"# ADR-{adr.id}: {adr.title}\n\n"
            f"## 背景\n{adr.context}\n\n## 决策\n{adr.decision}\n\n"
            f"## 备选方案\n" + "\n".join(f"- {a}" for a in adr.alternatives) +
            f"\n\n## 代价\n{adr.consequences}\n\n## 何时重新评估\n{adr.revisit_when}\n")

if __name__ == "__main__":
    print(render(adr_vectorstore))  # 输出可直接存为 docs/adr/0001-vectorstore.md
```

### 📚 推荐资料

- [Architecture Decision Records (ADR)](https://adr.github.io/) — ADR 模板与最佳实践合集。
- [pgvector](https://github.com/pgvector/pgvector) — 在 Postgres 里做向量检索，中小项目性价比之选。
- [Chroma](https://docs.trychroma.com/) / [FAISS](https://faiss.ai/) — 原型期最轻量的向量检索方案。
- [LangChain vs LlamaIndex vs 自研](https://www.pinecone.io/learn/langchain/) — 帮助判断是否需要 Agent 框架的讨论。

### 🛠 动手实践

为你的 Capstone 写至少 3 篇 ADR：模型选型、向量库选型、是否使用编排框架。每篇都必须包含"备选方案"和"何时重新评估"两节。验收标准：把任意一篇 ADR 给同行看，他能理解你的权衡逻辑，且无法简单地用"换成 X 不就行了"来反驳你——因为你已经把 X 列在备选里并说明了放弃理由。

## 第 4 节：项目工程化——让代码配得上"作品集"三个字

### 📖 知识点

Demo 代码和作品集代码的差距，往往比功能本身更能暴露你的工程水平。面试官 clone 下你的 repo，30 秒内就能从目录结构、README、配置管理判断你是不是一个靠谱的工程师。Capstone 必须**当成一个真正的软件项目来工程化**，至少做到四点：

- **清晰的代码结构**：按职责分层（数据/检索/生成/服务/评估），而不是一个 2000 行的 `main.py`。目录结构本身就是架构的表达。
- **配置与密钥管理**：所有可变项（模型名、阈值、API 地址）走配置文件 + 环境变量，**密钥绝不进 Git**（用 `.env` + `.gitignore` + `.env.example`）。这是 12-Factor 原则，也是 stage-08 的延续。
- **测试**：LLM 输出不能精确断言，但**确定性逻辑必须有单测**（数据清洗、分块、护栏、解析），核心链路要有评估集回归。有测试的 repo 立刻显得专业。
- **一份杀手级 README**：这是你作品集的门面。要有：一句话价值主张、架构图、quickstart（让人 5 分钟跑起来）、技术选型说明（链接到 ADR）、效果演示（截图/GIF/在线 demo）、已知局限与未来计划。

为什么工程化值得你花时间？因为**作品集项目的受众是"未来的雇主"，不是"现在的你"**。一个功能普通但工程规范、文档清晰、一键可跑的项目，给人的信任感远超一个功能炫酷但 clone 下来跑不起来、README 只有一行的项目。**可复现性，是专业度的第一信号**。

> **专家心法**：写完代码后，找一台干净的机器（或新建虚拟环境），严格按自己的 README 从零跑一遍。90% 的人会在这一步发现"原来我有三个没写进文档的隐藏步骤"。能让陌生人一次跑通的项目，已经赢过了大多数候选人。

### 💻 代码示例

一个清晰、可直接套用的端到端项目目录骨架与配置管理实现。

```text
capstone-arxiv-assistant/
├── README.md                  # 作品集门面：价值/架构图/quickstart/demo/局限
├── pyproject.toml             # 依赖与项目元数据
├── .env.example               # 列出需要的环境变量（不含真实值），.env 进 gitignore
├── Dockerfile                 # stage-08 的容器化
├── docs/
│   └── adr/                   # 架构决策记录（第 3 节产物）
├── src/
│   ├── config.py              # 集中配置：Pydantic Settings 读环境变量
│   ├── ingest/                # 数据上游：解析、清洗、分块、写入向量库
│   ├── retrieval/             # 检索层：embedding + 向量检索 + rerank
│   ├── generation/            # 生成/编排层：Prompt、RAG、Agent 循环
│   ├── guardrails/            # 护栏（stage-08）
│   ├── observability/         # Langfuse 埋点（stage-08）
│   └── api/                   # FastAPI 服务（stage-08）
├── eval/
│   ├── dataset.jsonl          # 评估集（stage-07）
│   └── run_eval.py            # 评估脚本，被 CI 当门禁调用
└── tests/                     # 确定性逻辑的单元测试
```

```python
# src/config.py —— 用 Pydantic Settings 集中管理配置，类型安全、从环境变量读取
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # 从 .env 读取；密钥永远不写进代码，.env 不进 Git
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    chat_model: str = "gpt-4o-mini"           # 可调项给默认值，便于切换
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5                            # 检索返回数，调参入口
    cache_sim_threshold: float = 0.92         # 语义缓存阈值（stage-08）
    eval_min_score: float = 0.8               # CI 评估门禁阈值（stage-07）
    langfuse_host: str = "https://cloud.langfuse.com"

# 全局唯一配置实例，到处 import 它，杜绝散落各处的硬编码
settings = Settings()  # 启动即校验：缺 key 会立刻报错，而不是运行到一半才崩
```

### 📚 推荐资料

- [Pydantic Settings 文档](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) — 类型安全的配置与密钥管理事实标准。
- [Make a README](https://www.makeareadme.com/) — 优秀 README 的结构模板。
- [The Good Research Code Handbook](https://goodresearch.dev/) — 如何把研究/原型代码组织成可维护项目，干货极多。
- [pytest 官方文档](https://docs.pytest.org/) — Python 测试事实标准。

### 🛠 动手实践

按上面的骨架重构你的 Capstone 代码：分层目录、`config.py` 集中配置、`.env.example`、至少为护栏和数据分块写单元测试、写一份完整 README。然后做"干净机器测试"：新建虚拟环境，严格照自己 README 跑一遍。验收标准：陌生人能在 5 分钟内按 README 把项目跑起来；repo 里搜不到任何明文密钥；`pytest` 能跑过。

## 第 5 节：讲好你的项目——写进简历、作品集与面试

### 📖 知识点

做完项目只是完成了一半，**能把它讲清楚、让别人意识到它的价值，才是 Capstone 的最终交付**。技术再扎实，讲不出来，在求职市场上约等于没做。这一节是把工程成果转化为职业资本的关键。

**写进简历**：一条好的 AI 项目经历遵循 **STAR + 量化** 原则。不要写"做了一个 RAG 问答系统"（人人都能写），要写："为 X 场景构建端到端 RAG 助手，通过混合检索 + rerank 将答案忠实度从 72% 提升至 91%（基于 50 题评估集），接入 Langfuse 监控并用语义缓存降低 40% 推理成本，FastAPI + Docker 部署"。**每一句都带技术决策和量化结果**——评估数据、成本数据、延迟数据，这些数字是你和"只会套壳"的人之间的护城河。

**作品集呈现**：一个可访问的在线 demo（或一段 2 分钟演示视频）+ 规范的 GitHub repo + 一篇技术博客（讲清你踩了什么坑、做了什么权衡）。博客尤其加分，它证明你不仅会做，还能反思和表达。

**面试讲项目**：准备一个**分层叙事**——30 秒电梯版（解决什么问题、效果如何）、3 分钟架构版（数据流 + 关键技术决策）、深挖版（针对每一环都能回答"为什么这么做、遇到什么坑、怎么解决、还能怎么优化"）。面试官最爱问的是**权衡和失败**："你遇到的最大挑战是什么？""如果重做你会改什么？""为什么不用 Agent？" 这些问题考的不是你做对了什么，而是你**理解多深**。

> **专家心法**：面试讲项目，要主动暴露你思考过的"取舍"和"做得还不够好的地方"，而不是把项目吹得完美无缺。说"我用了朴素 RAG，因为评估显示加 Agent 收益不明显且会翻倍成本"，比说"我用了最先进的多 Agent 架构"更能打动资深面试官——前者证明你有判断力，后者可能只是在堆砌名词。**真诚的取舍 > 虚假的完美**。

### 💻 代码示例

把你的评估结果固化成一份"项目效果报告"，它同时服务于简历量化、README 展示和面试举证——一份产出，三处复用。

```python
# report.py —— 生成项目效果报告，把工程成果翻译成"可写进简历的数字"
import json
from dataclasses import dataclass, asdict

@dataclass
class ProjectMetrics:
    eval_set_size: int            # 评估集规模，证明数字不是拍脑袋
    faithfulness_before: float    # 优化前忠实度（朴素 RAG 基线）
    faithfulness_after: float     # 优化后（加了 rerank / 改了 Prompt）
    cost_per_query_before: float  # 优化前单次成本（美元）
    cost_per_query_after: float   # 接入语义缓存/模型路由后
    p95_latency_ms: int

    def resume_bullets(self) -> list[str]:
        # 把指标直接转成简历可用的量化句子
        faith_gain = (self.faithfulness_after - self.faithfulness_before) * 100
        cost_cut = (1 - self.cost_per_query_after / self.cost_per_query_before) * 100
        return [
            f"基于 {self.eval_set_size} 题评估集，将答案忠实度提升 {faith_gain:.0f} 个百分点"
            f"（{self.faithfulness_before:.0%}→{self.faithfulness_after:.0%}）",
            f"通过语义缓存与模型路由降低单次推理成本 {cost_cut:.0f}%",
            f"端到端 P95 延迟控制在 {self.p95_latency_ms}ms，FastAPI + Docker 部署并接入 Langfuse 监控",
        ]

if __name__ == "__main__":
    m = ProjectMetrics(50, 0.72, 0.91, 0.012, 0.0072, 1800)
    print(json.dumps(asdict(m), ensure_ascii=False, indent=2))
    print("\n--- 简历可用句子 ---")
    for b in m.resume_bullets():
        print("•", b)
```

### 📚 推荐资料

- [Coursera: How to Write a Software Engineer Resume](https://www.coursera.org/articles/software-engineer-resume) — 工程简历的量化写法。
- [System Design Interview (ByteByteGo)](https://bytebytego.com/) — 讲清架构、应对深挖式追问的系统设计表达框架。
- [How to Build a Portfolio Project (freeCodeCamp)](https://www.freecodecamp.org/news/how-to-build-a-developer-portfolio/) — 作品集呈现的实操指南。
- [Writing technical blog posts (Julia Evans)](https://jvns.ca/blog/2017/03/20/blogging-principles/) — 如何把项目写成有说服力的技术博客。

### 🛠 动手实践

为你的 Capstone 产出三样东西：(1) 简历上的 2~3 条量化项目经历（每条都含技术决策 + 数字）；(2) 一份面试用的"分层叙事"提纲（30 秒 / 3 分钟 / 深挖三档）；(3) 一篇至少讲清"一个关键权衡 + 一个踩过的坑"的技术博客草稿。验收标准：找一位朋友扮演面试官，用你的提纲讲一遍，并接住"为什么这么选""最大挑战是什么"两个追问。

## 🎯 阶段实战项目：Capstone 端到端 AI 产品（毕业作品）

这就是本课程的最终交付物。它不是又一个练习，而是你求职作品集的核心、你 AI Engineer 能力的总检验。

**目标**：独立完成一个**打通 stage-01~08 全链路**的真实 AI 产品——从数据上游，经 RAG/Agent 推理，到评估、部署、监控，形成可演进的闭环。

**候选方向（三选一，或自拟同等深度的题目）**：

1. **个人知识库 / arXiv 论文助手**
   - 价值：让用户对自己的文献/笔记/收藏做语义问答与摘要追问。
   - 链路重点：PDF/网页解析与分块（数据上游）→ 混合检索 + rerank（RAG）→ 带溯源的问答（生成）→ 忠实度评估 → 部署监控。
   - 难点与亮点：分块策略对长文档的影响、答案溯源、"检索不到时诚实说不知道"。

2. **企业/团队制度文档智能客服**
   - 价值：员工自然语言查询 HR/IT/报销等制度，替代翻文档与问人。
   - 链路重点：多文档管理与元数据过滤（按部门/时效）→ 检索 → 受控生成（绝不编造政策）→ 护栏（拒答越权问题）→ 监控高频问题。
   - 难点与亮点：幻觉控制（答错政策代价高）、权限与边界、用监控数据发现文档盲区。

3. **多步任务 Agent（如"调研助手"）**
   - 价值：给定一个调研主题，自动检索、筛选、综合、产出结构化报告。
   - 链路重点：工具调用（搜索/抓取）→ 多步规划与反思（Agent 编排）→ 结果综合 → 评估报告质量 → 护栏（成本/循环熔断）→ 监控每步轨迹。
   - 难点与亮点：Agent 循环的可控性与成本熔断、中间步骤的可观测、何时该停。

**硬性要求（缺一不可，对应六个环节）**：

1. **数据上游**：有真实数据源，完成解析、清洗、分块、入向量库，并说明分块策略的选择理由。
2. **RAG/Agent**：实现核心检索/推理链路，答案可溯源；复杂度与问题匹配（用 ADR 说明为何用/不用 Agent）。
3. **评估**：建立 ≥30 题的离线评估集，定义并测量至少一个质量指标（如忠实度/正确率），给出基线 vs 优化后的对比数据。
4. **部署**：FastAPI 服务化（含流式）+ Docker 容器化，一键可跑。
5. **监控与护栏**：接入 Langfuse 全链路 trace + 成本统计；双向 Guardrails；至少一项降本措施（语义缓存或模型路由）并量化效果。
6. **工程化与交付**：分层代码结构、Pydantic 配置管理、`.env.example`、确定性逻辑单测、≥3 篇 ADR、一份杀手级 README、CI 含评估门禁。

**验收标准**：

- **可复现**：陌生人按 README 在干净环境下 5~10 分钟内跑起来，repo 无明文密钥。
- **闭环完整**：六个环节齐备，能现场演示"一个请求从入口到出口"的完整流转，并在 Langfuse 里回放其 trace。
- **数据说话**：提供项目效果报告，含评估指标的基线/优化对比、降本百分比、P95 延迟。
- **讲得清楚**：能用 30 秒 / 3 分钟 / 深挖三档叙事讲明项目，并接住"为什么这么选""最大挑战"的追问。
- **简历就绪**：产出 2~3 条量化的项目经历句子。

**可选进阶**：

- 部署一个公开可访问的在线 demo（Hugging Face Spaces / Railway / Fly.io 等），附二维码进简历。
- 写一篇技术博客复盘关键权衡与踩坑。
- 做一轮"上线就绪评审"：模拟上游故障、提示注入、预算超限三种场景，给出应对预案（呼应 stage-08）。
- 用真实用户跑一周，收集 Langfuse 中的"踩"反馈，构建第二版评估集并迭代——完成"监控反馈回流改进"的真正闭环。
