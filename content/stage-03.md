---
stage: 3
title: "RAG · 检索增强生成"
level: "进阶"
duration: "2 周"
goal: "掌握从加载到生成的完整 RAG 链路，能搭出可溯源、可调优、抗幻觉的企业级文档问答系统"
prereq: "完成 stage-01、02，了解向量检索基础（会用 LCEL 编排管道）"
stack: ["Python", "LangChain", "Chroma", "pgvector", "bge / OpenAI Embeddings", "Rerank"]
---

## 第 1 节：RAG 全景——为什么需要它，整条链路长什么样

### 📖 知识点

先厘清一个根本问题：**RAG（Retrieval-Augmented Generation，检索增强生成）解决的是 LLM 的"知识"问题，而不是"能力"问题。** LLM 的参数里固化的是训练截止时间前的通用知识，它**不知道**你公司的内部文档、最新的产品手册、昨天的数据。强行问，它要么说不知道，要么——更危险——**自信地编造（幻觉）**。

有两条路给模型补充私有/新知识：

- **微调（Fine-tuning）**：用你的数据继续训练模型，把知识"焊进"参数。成本高、更新慢、容易遗忘，且**改的是"风格/能力"比改"事实"更有效**。
- **RAG**：不动模型，而是在**回答前先去外部知识库检索相关片段，把它们塞进 Prompt 上下文**，让模型"开卷考试"。便宜、实时可更新、可溯源。

**划重点（这是新手最常犯的概念错误）：RAG 不是训练，不改一个模型参数。它是"检索 + 把结果拼进 Prompt"的工程方案。** 当需求是"让模型基于特定、可变、可溯源的事实回答"时，首选 RAG 而非微调。

一条完整的 RAG 链路分**离线建库**和**在线问答**两段：

```
【离线 / 建库】 加载 Load → 切分 Chunking → 向量化 Embedding → 存入向量库 Vector Store
【在线 / 问答】 用户问题 → (查询改写) → 检索 Retrieve → 重排 Rerank → 拼上下文 → 生成 Generate（带引用）
```

每个环节都会影响最终效果，且**误差会累积**：切分切坏了，检索再好也召回垃圾；检索召回不准，模型再强也"巧妇难为无米之炊"。所以 RAG 调优是一个**全链路系统工程**，不是调一个参数的事。

> **专家心法**：RAG 系统的天花板由**检索质量**决定，而不是由生成模型决定。新手总想换更强的 LLM 来救效果，老手先去看"召回的片段对不对"。记住一句话：**Garbage in, garbage out**——喂进上下文的若是无关片段，再强的 GPT 也只能基于垃圾编答案。

### 💻 代码示例

先用 LangChain 跑通一个**最小但完整**的 RAG，建立整体直觉（后续各节再逐环节拆解优化）：

```python
# pip install langchain langchain_openai langchain_community langchain_chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1) 加载文档
docs = TextLoader("./knowledge.txt", encoding="utf-8").load()

# 2) 切分成块
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3) Embedding + 存入向量库（建库，离线）
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # 检索 top-4

# 4) 在线问答管道：检索结果拼进 Prompt
prompt = ChatPromptTemplate.from_template(
    "仅根据以下上下文回答问题，若上下文没有答案就说不知道，不要编造。\n\n"
    "上下文：\n{context}\n\n问题：{question}"
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# LCEL 组合整条链：问题 → 检索→格式化 + 原问题 → Prompt → 模型 → 文本
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

print(rag_chain.invoke("文档里提到的核心功能是什么？"))
```

### 📚 推荐资料

- [RAG 原始论文（Lewis et al., 2020）](https://arxiv.org/abs/2005.11401) — "Retrieval-Augmented Generation" 概念出处。
- [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/) — 官方端到端 RAG 构建指南。
- [What is RAG（AWS 科普）](https://aws.amazon.com/what-is/retrieval-augmented-generation/) — 通俗讲清 RAG vs 微调的取舍。
- [A Survey on RAG（综述论文）](https://arxiv.org/abs/2312.10997) — 系统梳理 RAG 各环节与前沿方法。

### 🛠 动手实践

准备一篇你熟悉的长文档（如某产品 README 或一篇技术博客），用上面的最小链路搭一个问答。然后故意问一个"文档里没有"的问题，观察模型是否遵守"没有就说不知道"。验收标准：文档内问题答得准；文档外问题不胡编（若胡编，记下来——这正是后续章节要治的"失败模式"）。

## 第 2 节：加载与切分（Chunking）——RAG 效果的隐形地基

### 📖 知识点

Chunking（切分）是 RAG 里**最被低估、却最影响效果**的环节。为什么必须切？两个原因：(1) 文档常常很长，超过 Embedding 模型和上下文窗口的限制；(2) 检索的粒度若太粗（整篇文档一个向量），命中的相关性会被无关内容稀释；太细（一句话一块），又会丢失上下文。**切分本质是在"语义完整性"和"检索精度"之间找平衡。**

主流切分策略：

- **固定长度切分（Fixed-size）**：按字符/token 数硬切，最简单。问题是会把一句话、一个表格从中间劈开。
- **重叠切分（Overlap）**：相邻块之间留一段重叠（如 `chunk_overlap=50`），缓解"边界处语义被切断"。几乎是标配。
- **递归字符切分（RecursiveCharacterTextSplitter）**：LangChain 默认推荐。按优先级用一组分隔符（先 `\n\n` 段落，再 `\n` 行，再句号，再空格）递归地切，**尽量在自然语义边界断开**，是通用首选。
- **结构感知切分**：针对 Markdown/HTML/代码，按标题层级、函数边界切（如 `MarkdownHeaderTextSplitter`），保留结构语义。
- **语义切分（Semantic Chunking）**：用 Embedding 相似度判断"话题是否变了"来决定切点，效果好但成本高。

**关键参数**：`chunk_size`（块大小）和 `chunk_overlap`（重叠量）。没有万能值——技术文档/法律条款宜大块（保上下文），FAQ/对话宜小块（保精度）。中文还要注意：很多 splitter 默认分隔符偏英文，中文要确保按标点（。！？）正确断句。

> **专家心法**：调 RAG 时，**先怀疑 chunk，再怀疑别的**。一个经典翻车是 `chunk_size` 太大，导致一个块里混了好几个主题，Embedding 向量变成"四不像"，检索相似度全都不高不低，召回质量崩坏。建议：拿真实问题做实验，把召回的 chunk 打印出来肉眼看——它们是不是真的包含答案？这个"看 chunk"的习惯能解决一半以上的 RAG 问题。

### 💻 代码示例

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
)

text = open("./doc.md", encoding="utf-8").read()

# 1) 通用首选：递归字符切分。中文补充标点分隔符，避免在句中硬切
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)
chunks = splitter.split_text(text)
print(f"共切出 {len(chunks)} 块")
for i, c in enumerate(chunks[:2]):
    print(f"\n--- chunk {i} (len={len(c)}) ---\n{c}")

# 2) 结构感知：Markdown 按标题层级切，保留 h1/h2 作为元数据
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
)
md_chunks = md_splitter.split_text(text)
# 每个块带 metadata={'H1': ..., 'H2': ...}，检索时可用于过滤/溯源
print(md_chunks[0].metadata)
```

实验对比不同 `chunk_size` 对召回的影响（这是该做的调参动作）：

```python
for size in (200, 500, 1000):
    s = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=int(size*0.1))
    n = len(s.split_text(text))
    print(f"chunk_size={size:>4} -> {n} 块，平均每块约 {len(text)//max(n,1)} 字")
```

### 📚 推荐资料

- [LangChain Text Splitters 概念](https://python.langchain.com/docs/concepts/text_splitters/) — 各类切分器对比与选型。
- [5 Levels of Text Splitting（Greg Kamradt）](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb) — 从固定切到语义切的经典分级讲解。
- [Chunking Strategies for RAG（Pinecone）](https://www.pinecone.io/learn/chunking-strategies/) — 切分策略的系统总结。
- [How to split by HTML/Markdown](https://python.langchain.com/docs/how_to/#text-splitters) — 结构化切分官方 how-to 合集。

### 🛠 动手实践

拿一篇结构清晰的 Markdown 文档，分别用固定切分、递归切分、Markdown 标题切分三种方式处理，把每种切出的前几个 chunk 打印出来对比"语义是否被切断"。再对同一个真实问题，看哪种切法召回的 chunk 最贴合答案。验收标准：能用具体例子说明"为什么递归/结构切优于盲目固定切"。

## 第 3 节：Embedding 与向量库——把语义变成可检索的向量

### 📖 知识点

**Embedding（嵌入）** 是 RAG 的"翻译官"：它把一段文本映射成一个高维向量（如 1024 维），使得**语义相近的文本，向量在空间里也相近**（用余弦相似度衡量）。检索的本质就是：把用户问题也 embedding 成向量，在向量库里找**最近邻**的若干 chunk。

再次强调概念边界（呼应第 1 节）：**Embedding 不是微调、不是生成。** 它是一个独立的、把文本转成向量的模型，产出的是"语义坐标"，本身不回答问题。

**Embedding 模型选型**（这是工程决策点）：

- **OpenAI `text-embedding-3-small/large`**：英文强、接入简单、按量付费，适合快速起步和英文为主场景。
- **中文场景首选开源 bge 系列**（BAAI，如 `bge-large-zh-v1.5`、`bge-m3`）：中文语义理解强、可本地部署（数据不出内网）、免费。`bge-m3` 还支持多语言和混合检索。中文 RAG 用英文 embedding 是常见性能杀手。
- **选型看 [MTEB 榜单](https://huggingface.co/spaces/mteb/leaderboard)**：按你的语言、任务、维度、是否需本地部署综合权衡。

**向量库（Vector Store）** 负责存向量并做高效近邻检索（靠 HNSW 等 ANN 索引，而非暴力遍历）：

- **Chroma**：轻量、易上手、可纯本地，适合开发和中小项目。
- **pgvector**：PostgreSQL 的向量扩展。**强烈推荐生产用**——你的业务数据本就在 PG 里，向量和业务数据同库，可联合查询、复用运维/备份体系，省一个独立组件。
- 其他：Milvus / Qdrant / Weaviate（大规模专用），Pinecone（托管）。

> **专家心法**：(1) **建库和查询必须用同一个 embedding 模型**——换了模型，旧向量全部作废得重建库，这是血泪教训。(2) 别一上来就上 Milvus 这种重型库；多数项目 Chroma 起步、pgvector 上生产足矣，**过早引入分布式向量库是典型的过度工程**。(3) 维度越高不一定越好，要权衡存储/检索成本与精度。

### 💻 代码示例

```python
# 中文场景：用开源 bge 做本地 Embedding（数据不出内网）
# pip install langchain-huggingface sentence-transformers
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True},  # bge 推荐归一化，配合余弦相似度
)
v = emb.embed_query("如何申请年假？")
print(len(v))  # 向量维度，如 1024

# 存入 Chroma（开发期）并持久化到本地目录
from langchain_chroma import Chroma
vs = Chroma.from_documents(chunks, emb, persist_directory="./chroma_db")
results = vs.similarity_search_with_score("请假流程", k=3)
for doc, score in results:
    print(f"[score={score:.3f}] {doc.page_content[:50]}...")
```

生产场景换成 pgvector，业务代码几乎不变（统一抽象的好处）：

```python
# pip install langchain-postgres psycopg
from langchain_postgres import PGVector

vs = PGVector(
    embeddings=emb,
    collection_name="company_docs",
    connection="postgresql+psycopg://user:pwd@localhost:5432/ragdb",
)
vs.add_documents(chunks)            # 写入
hits = vs.similarity_search("报销标准", k=4)  # 检索
```

### 📚 推荐资料

- [MTEB 嵌入模型榜单](https://huggingface.co/spaces/mteb/leaderboard) — 选 Embedding 模型的权威参考（按语言/任务筛）。
- [BGE 模型（BAAI 官方）](https://huggingface.co/BAAI/bge-large-zh-v1.5) — 中文 Embedding 首选，含用法说明。
- [pgvector 项目](https://github.com/pgvector/pgvector) — PostgreSQL 向量扩展，生产首选之一。
- [OpenAI Embeddings 指南](https://platform.openai.com/docs/guides/embeddings) — text-embedding-3 系列官方说明。

### 🛠 动手实践

用 `bge-large-zh-v1.5` 把第 2 节切好的中文 chunk 建成 Chroma 库，写 3~5 个真实问题做 `similarity_search_with_score`，观察 score 和召回内容。再换成 OpenAI 的 embedding 重建一次，对比同样问题的中文召回质量。验收标准：能用数据说明"中文场景下 bge 召回为何更靠谱"，并理解换模型必须重建库。

## 第 4 节：检索进阶——混合检索、Rerank 与查询改写

### 📖 知识点

朴素的"向量相似度 top-k"召回，在真实场景常常不够。三大进阶手段是把 RAG 从"能用"做到"好用"的关键：

**1) 混合检索（Hybrid Search）= 向量检索 + 关键词检索**
- **向量检索**懂语义（"年假"≈"带薪休假"），但对**精确术语、专有名词、编号**（如 "ISO-9001"、产品型号 "X100"）不敏感。
- **关键词检索（BM25 等稀疏检索）**擅长精确匹配，但不懂同义。
- 两者**互补**，把各自结果融合（常用 RRF, Reciprocal Rank Fusion）能显著提升召回。LangChain 用 `EnsembleRetriever` 实现。

**2) 重排（Rerank）——RAG 性价比最高的提升点之一**
- 检索（召回）阶段为了快，用的是"双塔"式 Embedding 相似度，**不够精**。Rerank 用一个 **Cross-Encoder**（如 `bge-reranker`、Cohere Rerank）把"问题 + 每个候选 chunk"成对送进模型精算相关性。
- 标准两段式套路：**先用向量检索粗召回 top-50（重召回率），再用 reranker 精排取 top-5（重精确率）** 送给 LLM。
- 这一步往往能立竿见影地提升答案质量，且实现成本低。

**3) 查询改写 / HyDE**
- 用户的提问常常口语、模糊、含指代，和文档措辞对不上。**查询改写**让 LLM 先把问题改写/扩展成更适合检索的形式（如多角度生成几个变体查询）。
- **HyDE（Hypothetical Document Embeddings）**：让 LLM 先"假想"一个答案文档，用这个假想答案去做向量检索。因为"答案"和"目标文档"在语义空间更接近，常比直接用问题检索更准。

> **专家心法**：如果你的 RAG 只能做一项优化，**做 Rerank**——它对最终答案质量的投入产出比通常最高。优化顺序建议：先把 chunk 和 embedding 搞对（地基）→ 加 Rerank（精排）→ 不够再上混合检索/查询改写。不要一上来堆满所有花活，**每加一层都要用评估集量化它是否真的带来提升**，否则只是增加延迟和成本的"自我感动"。

### 💻 代码示例

```python
# 混合检索：BM25（关键词）+ 向量，二者用 EnsembleRetriever 融合
# pip install rank_bm25
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25 = BM25Retriever.from_documents(chunks)   # 稀疏/关键词
bm25.k = 5
vector_ret = vs.as_retriever(search_kwargs={"k": 5})  # 稠密/语义

hybrid = EnsembleRetriever(
    retrievers=[bm25, vector_ret],
    weights=[0.4, 0.6],   # 关键词与语义的融合权重，可调
)
docs = hybrid.invoke("ISO-9001 认证的年假政策")
```

加一层 Rerank（用 bge-reranker 精排），构成"粗召回→精排"两段式：

```python
# pip install langchain-community sentence-transformers
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# 1) Cross-Encoder reranker
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
reranker = CrossEncoderReranker(model=reranker_model, top_n=4)  # 精排后留 4 条

# 2) 粗召回（向量取 top-20）→ 精排（reranker 取 top-4）
base = vs.as_retriever(search_kwargs={"k": 20})
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=base
)
final_docs = compression_retriever.invoke("报销的额度上限是多少？")
for d in final_docs:
    print(d.page_content[:60], "...")
```

HyDE 思路（用 LLM 生成假想答案再检索）：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
hyde_prompt = ChatPromptTemplate.from_template("针对问题写一段可能的答案（即使不确定）：{q}")
hyde_chain = hyde_prompt | llm | StrOutputParser()

q = "新员工的设备申领流程"
hypo_answer = hyde_chain.invoke({"q": q})       # 假想答案
docs = vs.similarity_search(hypo_answer, k=4)   # 用假想答案去检索，常比用原问题更准
```

### 📚 推荐资料

- [How to do hybrid search（EnsembleRetriever）](https://python.langchain.com/docs/how_to/ensemble_retriever/) — 官方混合检索用法。
- [bge-reranker（BAAI 官方）](https://huggingface.co/BAAI/bge-reranker-v2-m3) — 中文/多语 Rerank 首选模型。
- [HyDE 论文（Precise Zero-Shot Dense Retrieval）](https://arxiv.org/abs/2212.10496) — 查询改写思路的理论来源。
- [Cohere Rerank 文档](https://docs.cohere.com/docs/rerank-overview) — 托管 Rerank 服务，理解 Cross-Encoder 精排。

### 🛠 动手实践

在第 3 节的向量库基础上，加一层 `bge-reranker-v2-m3` 精排，构成"向量粗召回 top-20 → rerank top-4"。挑 5 个真实问题，对比"加 Rerank 前后"召回 top-4 的相关性差异（肉眼判断或人工打分）。再额外做一组混合检索对照。验收标准：能用具体例子证明 Rerank 把更相关的 chunk 排到了前面，并说出它为什么比纯向量检索更准。

## 第 5 节：生成、引用溯源与 RAG 失败模式

### 📖 知识点

检索到好的上下文后，最后一步是**生成**。这一步看似简单（拼 Prompt 调模型），但有两个工程重点：**抗幻觉的 Prompt 设计**和**引用溯源**。

**抗幻觉的生成 Prompt** 要明确три条约束：(1) **只能依据给定上下文回答**；(2) **上下文没有就明说"不知道"，禁止编造**；(3) 最好要求**标注信息来源**。这能大幅降低模型"脑补"。配合 `temperature=0` 提升稳定性。

**引用溯源（Citation）** 是企业级 RAG 的**刚需而非加分项**——用户（尤其法务、医疗、金融场景）必须能核实答案出处。实现要点：在 chunk 的 `metadata` 里保留来源（文件名、页码、URL、标题层级），生成时把来源 ID 一起给模型，要求它在答案里标注 `[来源 1]`，并把对应原文回显给用户。这样答案**可验证、可追责**，也是建立用户信任的关键。

**RAG 常见失败模式**（务必背下来，排错时按图索骥）：

- **检索召回了无关内容**：根因多在 chunk 切分差或 embedding 不适配 → 回去看 chunk、换 embedding、加 Rerank。
- **召回对了但答案错**：上下文里有答案模型却没用好 → 优化生成 Prompt、降温、缩短上下文（"Lost in the Middle"：关键信息在长上下文中间易被忽略，应把最相关的放首尾）。
- **答非所问 / 幻觉**：Prompt 没约束住 → 加"无依据就说不知道"。
- **多跳问题答不全**：需要综合多处信息的问题，单次检索不够 → 用查询分解 / 多轮检索（进阶 Agentic RAG）。
- **"明明文档里有却查不到"**：往往是术语/同义词错位 → 上混合检索或查询改写。

**评估**：别凭感觉调。建一个小评估集（问题 + 标准答案 + 应命中的来源），用 **RAGAS** 等框架量化 faithfulness（答案是否忠于上下文）、answer relevancy、context precision/recall。**没有评估的 RAG 调优都是玄学。**

> **专家心法**：RAG 排错有个黄金流程——**永远先把"实际喂给 LLM 的上下文"打印出来看**。90% 的"模型答错"其实是"根本没召回对内容"，问题在检索不在生成。先定位是"检索错了"还是"生成错了"，再对症下药，能省下大量瞎调模型的时间。引用溯源不仅是给用户看的，也是给你自己 debug 用的最好工具。

### 💻 代码示例

带引用溯源的完整生成链（用 Pydantic 强制结构化输出引用）：

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1) 结构化输出：答案 + 引用的来源编号，强制模型溯源
class AnswerWithCitations(BaseModel):
    answer: str = Field(description="基于上下文的回答；上下文无答案则说明无法回答")
    citations: list[int] = Field(description="支撑答案的来源编号列表")

prompt = ChatPromptTemplate.from_template(
    "你是严谨的知识助手。只依据【上下文】回答，无依据就说无法回答，禁止编造。\n"
    "在 citations 中给出你引用的来源编号。\n\n【上下文】\n{context}\n\n问题：{question}"
)

def format_with_id(docs):
    # 给每个 chunk 编号并附带来源元数据，供模型引用
    lines = []
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "未知")
        lines.append(f"[来源 {i}] (出处: {src})\n{d.page_content}")
    return "\n\n".join(lines)

structured_llm = llm.with_structured_output(AnswerWithCitations)

def answer(question, retriever):
    docs = retriever.invoke(question)
    ctx = format_with_id(docs)
    result: AnswerWithCitations = (prompt | structured_llm).invoke(
        {"context": ctx, "question": question}
    )
    # 把引用编号映射回真实出处，回显给用户核实
    print("答案：", result.answer)
    print("引用来源：")
    for cid in result.citations:
        if 0 <= cid < len(docs):
            print(f"  [来源 {cid}] {docs[cid].metadata.get('source')} -> {docs[cid].page_content[:50]}...")
    return result

# answer("年假有几天？", compression_retriever)
```

用 RAGAS 做量化评估的骨架：

```python
# pip install "ragas>=0.2" datasets
# 注意：faithfulness/answer_relevancy 等指标默认需要评测用 LLM 与 embeddings
# （默认走 OpenAI，需配好 OPENAI_API_KEY）——评估本身也要调模型，别困惑。
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# ragas>=0.2 已改列名：user_input / response / retrieved_contexts / reference
eval_data = EvaluationDataset.from_list([
    {
        "user_input": "年假有几天？",
        "response": "根据制度，年假为 10 天。",          # 你的 RAG 输出
        "retrieved_contexts": ["员工每年享有带薪年假 10 天..."],  # 检索到的上下文
        "reference": "10 天",                            # 标准答案
    },
])
report = evaluate(eval_data, metrics=[faithfulness, answer_relevancy, context_precision])
print(report)  # 用分数驱动优化，而非凭感觉
```

### 📚 推荐资料

- [How to get a RAG application to add citations](https://python.langchain.com/docs/how_to/qa_citations/) — 官方引用溯源实现指南。
- [Lost in the Middle 论文](https://arxiv.org/abs/2307.03172) — 长上下文中间信息易被忽略，影响上下文排布策略。
- [RAGAS 文档](https://docs.ragas.io/) — RAG 量化评估事实标准框架。
- [Advanced RAG（LlamaIndex 视角）](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/) — 生产级 RAG 优化清单。

### 🛠 动手实践

给第 4 节的链路加上"带引用的结构化生成"：每个答案必须附带它引用的来源 chunk 及出处。然后手动构造 3 个会触发不同失败模式的问题（一个文档外的、一个需要综合多段的、一个含精确术语的），观察系统表现并记录属于哪种失败模式、该如何治。验收标准：答案能正确回显引用出处；能对照"失败模式清单"诊断至少 2 种问题的根因。

## 🎯 阶段实战项目：企业文档问答系统（带引用溯源）

**背景**：某公司有一批内部制度文档（员工手册、报销规定、请假流程等，PDF/Markdown 混合）。你要做一个问答系统，员工用自然语言提问，系统基于这些文档准确回答，**并给出可核实的出处**，绝不编造。

**要求**：

1. **完整链路**：实现 加载（支持 PDF + Markdown）→ 切分（用 `RecursiveCharacterTextSplitter`，中文分隔符配好）→ Embedding（中文用 `bge-large-zh-v1.5`）→ 存向量库（开发用 Chroma 持久化，进阶可换 pgvector）。
2. **进阶检索**：实现"向量粗召回 top-20 → `bge-reranker` 精排 top-4"的两段式检索；可选叠加 BM25 混合检索。
3. **抗幻觉生成**：Prompt 明确"只依据上下文、无依据说不知道"，`temperature=0`。
4. **引用溯源（硬性）**：每条答案必须返回引用的来源（文件名 + 原文片段），用 `with_structured_output` 强制结构化，前端/CLI 把出处回显给用户。
5. **评估**：自建一个 ≥10 条的评估集（问题 + 标准答案 + 应命中来源），用 RAGAS 或自定义指标，量化 faithfulness 与召回准确率。
6. **失败模式应对**：至少处理"问文档外问题"（答不知道）和"精确术语查询"（混合检索或查询改写）两种情况。

**技术要点**：多格式加载、中文 Chunking、bge Embedding、Chroma/pgvector、Rerank、引用溯源、抗幻觉 Prompt、RAGAS 评估、失败模式诊断。

**验收标准**：
- 文档内问题答得准且**每条答案都带可核实出处**；
- 文档外问题明确回答"无法从文档中找到"，不编造；
- 加 Rerank 后，用评估集能看到检索/答案指标相对纯向量检索有可量化的提升；
- 含精确术语（如某编号/型号）的问题，混合检索能正确命中。

**可选进阶**：
- 接入 **查询改写 / HyDE**，对比开关前后在"口语化提问"上的召回变化；
- 实现**增量更新**：文档变更时只重建受影响的 chunk，而非全库重建；
- 把系统包成一个带流式输出的 Web 界面（如 Streamlit/Gradio），答案区实时显示、引用区可点击展开原文；
- 处理**多跳问题**：用查询分解（把复杂问题拆成子问题分别检索再汇总）做一版 Agentic RAG，对比单次检索的差距。
