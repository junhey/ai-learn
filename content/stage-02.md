---
stage: 2
title: "编排框架 · LangChain / LlamaIndex"
level: "基础"
duration: "1.5 周"
goal: "掌握用 LCEL 把 Prompt、模型、解析器、记忆与工具串成可维护的管道，并理解何时该用框架、何时不该用"
prereq: "完成 stage-01（会调 LLM API、懂采样参数与结构化输出）"
stack: ["Python", "LangChain (LCEL)", "langchain_openai", "LlamaIndex", "Pydantic"]
---

## 第 1 节：编排框架到底解决什么问题——从裸 SDK 到 LangChain

### 📖 知识点

在 stage-01 里，你已经能用 `openai` 裸 SDK 调通模型了。那为什么还需要 LangChain 这类"编排框架"？答案是：**当应用从"调一次模型"变成"一条由多个步骤组成的流水线"时，裸 SDK 的胶水代码会迅速失控。**

想象一个稍微真实的需求：拿用户问题 → 套一个带变量的 Prompt 模板 → 调模型 → 把输出解析成 JSON → 失败了要重试 → 还要带上历史对话 → 有时要先查知识库再回答。如果全用裸 SDK 手写，你会写出一堆 `messages.append(...)`、字符串拼接、`try/except`、手搓的解析函数，而且每换一个模型供应商（OpenAI → 通义 → 本地 vLLM）就要重写一遍。

LangChain 的核心价值有三层：

- **统一抽象**：把"模型""提示词""解析器""检索器""工具"都抽象成标准组件，换供应商只改一行。
- **可组合（这是灵魂）**：通过 **LCEL（LangChain Expression Language）**，用 `|` 操作符像搭乐高一样把组件串成"管道"，每个组件都遵守统一的 `Runnable` 接口，天然支持流式、异步、批处理、重试、并行。
- **生态**：几百个现成的文档加载器、向量库、工具集成，不用重复造轮子。

但要先泼一盆冷水：**框架不是免费的**。它引入了抽象层、版本碎片化（LangChain 0.1→0.3 经历过剧烈重构）、调试时多了一层"黑盒"。所以本阶段最后一节会专门讲"何时不该用框架"。

> **专家心法**：把 LangChain 理解成"LLM 时代的管道编排语言"，而不是"一个能帮你写好 Prompt 的魔法盒"。它帮你管理的是**组合与数据流**，不是帮你思考。Prompt 质量、检索质量这些"内功"，框架一点都不会替你提升。

### 💻 代码示例

先装包、跑通最小的 `ChatOpenAI` 调用。注意 LangChain 0.1 之后已经**模块化拆包**：聊天模型在独立的 `langchain_openai` 里，不要再 `from langchain.chat_models import ...`（旧写法已废弃）。

```python
# pip install langchain langchain_openai
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ChatOpenAI 是对 OpenAI Chat 接口的标准化封装
# temperature=0 用于需要稳定输出的场景（见 stage-01）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 用标准 Message 对象组织对话，比裸 SDK 的 dict 更类型安全
messages = [
    SystemMessage(content="你是一个简洁的技术助手。"),
    HumanMessage(content="用一句话解释什么是 LCEL。"),
]

resp = llm.invoke(messages)   # invoke 是所有 Runnable 的统一同步入口
print(resp.content)
print(resp.response_metadata.get("token_usage"))  # 用量信息
```

换成兼容 OpenAI 接口的国产/本地模型，只需改 `base_url` 和 `model`，业务代码一行不动——这就是"统一抽象"的收益：

```python
# 例如指向本地 vLLM / Ollama / 通义等 OpenAI 兼容端点
llm = ChatOpenAI(
    model="qwen2.5-7b-instruct",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    temperature=0,
)
```

### 📚 推荐资料

- [LangChain 官方文档 - Introduction](https://python.langchain.com/docs/introduction/) — 框架全景与最新（v0.3）架构总览。
- [Why LangChain（架构与设计哲学）](https://python.langchain.com/docs/concepts/) — 官方概念页，讲清各抽象的职责边界。
- [langchain_openai 集成文档](https://python.langchain.com/docs/integrations/chat/openai/) — ChatOpenAI 的所有参数与用法。
- [LangChain 包结构说明](https://python.langchain.com/docs/concepts/architecture/) — 理解 langchain-core / langchain / 集成包的分层。

### 🛠 动手实践

装好 `langchain` 与 `langchain_openai`，用 `ChatOpenAI` 实现一个最小问答，并打印 token 用量。然后把 `base_url` 指向任意一个 OpenAI 兼容端点（如 Ollama 跑的本地模型），验证"业务代码不变、只换配置"即可切换供应商。验收标准：同一段调用代码能跑通两个不同的模型后端。

## 第 2 节：LCEL——用 `|` 把 Prompt、Model、Parser 串成管道

### 📖 知识点

LCEL 是 LangChain 现代写法的**绝对核心**。它的思想借鉴了 Unix 管道：每个组件都实现统一的 `Runnable` 接口（有 `invoke` / `batch` / `stream` / `ainvoke` 等方法），于是可以用 `|` 操作符把它们左到右连起来，前一个的输出自动成为后一个的输入。

一条最经典的"三件套"管道是：

```
PromptTemplate | ChatModel | OutputParser
```

含义是：用户变量填进**提示词模板** → 生成的消息送进**模型** → 模型返回的 `AIMessage` 经**输出解析器**变成你想要的类型（纯字符串 / JSON / Pydantic 对象）。

为什么这套设计如此重要？因为 `|` 拼出来的 `chain` 本身**又是一个 Runnable**，自动获得这些能力，全部免费：

- `chain.invoke(x)`：跑一次；
- `chain.batch([x1, x2, ...])`：自动批处理；
- `chain.stream(x)`：流式逐 token 输出（做打字机效果）；
- `await chain.ainvoke(x)`：异步，高并发场景必备；
- `chain.with_retry()` / `chain.with_fallbacks([...])`：重试与降级。

这就是"组合优于配置"的威力——你不需要为"流式""并发"分别写代码，组合好的管道天然支持。

几个关键组件：

- **`ChatPromptTemplate`**：带占位符的提示词模板，`from_messages` 可定义 system/human/placeholder 多角色，用 `{var}` 注入变量。
- **`StrOutputParser`**：最常用，把 `AIMessage` 抽成纯字符串。
- **结构化输出**：现代推荐直接用 `llm.with_structured_output(PydanticModel)`，比老的 `PydanticOutputParser`（靠 Prompt 里塞格式说明）更可靠，原理是走模型原生的 function calling / structured outputs（呼应 stage-01）。

> **专家心法**：能用 LCEL 表达的逻辑，就不要退回去手写 `for` 循环和 `if`。LCEL 管道是声明式的——你描述"数据怎么流"，框架负责"怎么高效执行"。一旦你开始在管道外面写大量胶水代码，往往说明抽象用错了地方，该停下来重新设计数据流。

### 💻 代码示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1) Prompt 模板：{topic} 是运行时注入的变量
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位资深技术作者，输出精炼、有洞察。"),
    ("human", "用三个要点解释 {topic}，每点不超过 20 字。"),
])

# 2) 用 | 把三件套串成管道；StrOutputParser 把 AIMessage 转成 str
chain = prompt | llm | StrOutputParser()

# 3) invoke 传入的 dict 会去填模板里的变量
print(chain.invoke({"topic": "向量数据库"}))

# 4) 管道天然支持批处理与流式，无需改写逻辑
results = chain.batch([{"topic": "RAG"}, {"topic": "Agent"}])
for r in results:
    print("---", r)

print("\n--- 流式输出 ---")
for chunk in chain.stream({"topic": "LCEL"}):
    print(chunk, end="", flush=True)   # 打字机效果
```

结构化输出推荐用 `with_structured_output`（替代手搓 Parser）：

```python
from pydantic import BaseModel, Field

class Topic(BaseModel):
    name: str = Field(description="主题名")
    difficulty: int = Field(description="难度 1-5")
    points: list[str] = Field(description="关键要点")

# 直接绑定 Pydantic 模型，底层走模型原生结构化输出，稳定可靠
structured_llm = llm.with_structured_output(Topic)
chain = prompt | structured_llm
result: Topic = chain.invoke({"topic": "LangChain"})
print(type(result), result.difficulty, result.points)
```

### 📚 推荐资料

- [LCEL 官方概念文档](https://python.langchain.com/docs/concepts/lcel/) — Runnable 接口与组合机制的权威说明。
- [Runnable interface 详解](https://python.langchain.com/docs/concepts/runnables/) — invoke/batch/stream/ainvoke 全接口。
- [How to use structured outputs](https://python.langchain.com/docs/how_to/structured_output/) — with_structured_output 的完整指南。
- [ChatPromptTemplate How-to](https://python.langchain.com/docs/how_to/#prompt-templates) — 各类提示词模板用法合集。

### 🛠 动手实践

用 LCEL 搭一条 `ChatPromptTemplate | ChatOpenAI | StrOutputParser` 的翻译管道（输入中文，输出英文），然后：(1) 用 `batch` 一次翻译 5 个句子；(2) 用 `stream` 实现打字机效果；(3) 再加一个版本用 `with_structured_output` 返回 `{original, translated, word_count}` 结构。验收标准：三种调用方式共用同一条核心管道，业务逻辑不重复。

## 第 3 节：记忆与多轮对话——RunnableWithMessageHistory

### 📖 知识点

LLM API 本质是**无状态**的：你每次调用，模型都"失忆"，它能记住上文的唯一原因是你**把历史消息一起塞进了请求**。所谓"对话记忆"，工程上就是"如何存储、注入、裁剪历史消息"。

很多旧教程用 `ConversationBufferMemory` 这类 `Memory` 类——**注意：这些在 LangChain 0.3 里已被标记为 legacy/deprecated，不要再用**。现代标准做法是 **`RunnableWithMessageHistory`**：它包裹你的 LCEL 管道，在 Prompt 里用 `MessagesPlaceholder` 预留一个历史插槽，每次调用时按 `session_id` 自动把对应会话的历史读出来注入、并把新一轮对话存回去。

它的关键设计：

- 用一个工厂函数 `get_session_history(session_id) -> BaseChatMessageHistory` 提供存储，你可以接内存（`ChatMessageHistory`）、Redis、数据库等。
- 调用时通过 `config={"configurable": {"session_id": "..."}}` 区分不同用户/会话，天然支持多用户隔离。

更进一步要懂**记忆的成本与裁剪**：历史越长，token 越多，越贵越慢，还可能超出上下文窗口。生产里常用策略：

- **窗口截断**：只保留最近 N 轮（`trim_messages`）。
- **摘要记忆**：把久远历史用模型压缩成摘要，省 token。

> **专家心法**：记忆不是"越多越好"。无脑全量拼接历史是新手最常踩的坑——既烧钱又稀释模型注意力（关键信息淹没在闲聊里）。把"记什么、记多久、怎么压缩"当成一个产品设计问题来权衡。另外提一句方向：LangChain 官方正把有状态/记忆类应用推向 **LangGraph**，它用图 + 持久化 checkpoint 管理状态，比 `RunnableWithMessageHistory` 更适合复杂多步 Agent；本阶段先用前者打牢基础，stage 后期会接触 LangGraph。

### 💻 代码示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 1) Prompt 里用 MessagesPlaceholder 预留历史插槽
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好的助手，会结合上下文连续对话。"),
    MessagesPlaceholder(variable_name="history"),   # 历史消息注入点
    ("human", "{input}"),
])
chain = prompt | llm | StrOutputParser()

# 2) 用一个 dict 当简易存储：session_id -> 历史对象
store: dict[str, BaseChatMessageHistory] = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 3) 用 RunnableWithMessageHistory 包裹管道
chat = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",     # 用户输入对应的 key
    history_messages_key="history", # 注入历史的 placeholder 名
)

# 4) 通过 session_id 区分会话；同一 session 自动带历史
cfg = {"configurable": {"session_id": "user-001"}}
print(chat.invoke({"input": "我叫小明，喜欢爬山。"}, config=cfg))
print(chat.invoke({"input": "我叫什么？我的爱好是什么？"}, config=cfg))
# 第二轮能答出"小明、爬山"——证明历史被正确注入

# 换一个 session_id，则是全新的、无记忆的对话
cfg2 = {"configurable": {"session_id": "user-002"}}
print(chat.invoke({"input": "我叫什么？"}, config=cfg2))  # 答不出来
```

历史过长时用 `trim_messages` 裁剪（接在管道前）：

```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=2000, strategy="last",   # 只保留最近的、总量不超 2000 token
    token_counter=llm, include_system=True,
)
# 可在构建 chain 时把 trimmer 接进 history 处理，控制成本
```

### 📚 推荐资料

- [How to add message history（官方）](https://python.langchain.com/docs/how_to/message_history/) — RunnableWithMessageHistory 权威用法。
- [How to trim messages](https://python.langchain.com/docs/how_to/trim_messages/) — 历史裁剪与 token 控制。
- [Chat history 概念](https://python.langchain.com/docs/concepts/chat_history/) — 记忆模型的设计思路。
- [LangGraph 持久化与记忆](https://langchain-ai.github.io/langgraph/concepts/persistence/) — 复杂有状态应用的进阶方向。

### 🛠 动手实践

用 `RunnableWithMessageHistory` 做一个能记住用户名字和偏好的多轮聊天 CLI，支持用命令行参数切换 `session_id`。再加上 `trim_messages`，把历史限制在固定 token 内，并打印每轮请求实际携带的历史条数。验收标准：同一 session 能跨多轮记住信息；不同 session 互相隔离；历史超限时旧消息被正确丢弃。

## 第 4 节：工具调用与 Agent——让模型能"动手"

### 📖 知识点

到目前为止模型只能"说"，不能"做"。但真实应用常需要它查天气、算数、查数据库、调 API——这就是 **Tool Calling（工具调用）** 与 **Agent** 要解决的。

理解机制最关键的一点：**模型本身不会执行任何代码**。所谓"工具调用"，是模型在看到你声明的工具列表后，决定"该调哪个工具、用什么参数"，然后**以结构化形式（一个 tool_call 请求）返回给你**；真正执行函数的是你的程序。执行完把结果再喂回模型，模型据此组织最终回答。这是一个"模型出意图 → 程序执行 → 结果回灌"的循环。

现代 LangChain 写法：

- 用 `@tool` 装饰器把普通 Python 函数变成工具，**函数的 docstring 极其重要**——模型靠它判断何时该用这个工具，写不清就调不对。
- 用 `llm.bind_tools([...])` 把工具"挂"到模型上，模型就会在需要时返回 `tool_calls`。
- **Agent** 则把"调工具 → 看结果 → 再决定下一步"这个循环自动化。现代推荐用 `create_tool_calling_agent` + `AgentExecutor`（取代了老的、基于文本解析的 ReAct/`initialize_agent` 写法，那些容易解析失败）。

**Agent 的本质是一个带循环的决策器**：它反复问模型"基于现在掌握的信息，下一步做什么（调工具还是给最终答案）"，直到得出答案或达到步数上限。

> **专家心法**：工具描述（docstring + 参数 schema）就是给模型的"API 文档"，质量直接决定调用成功率——这比调模型参数重要得多。另外，Agent 很"强大"但也很"贵且慢且不可控"（多轮往返、可能死循环）。生产中能用固定 LCEL 管道（确定性流程）解决的，就别上 Agent；只有当"下一步该做什么"真的需要模型动态决策时，Agent 才是对的工具。

### 💻 代码示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1) 用 @tool 定义工具；docstring 是模型判断"何时用"的依据，务必写清
@tool
def add(a: int, b: int) -> int:
    """计算两个整数 a 和 b 的和。"""
    return a + b

@tool
def get_word_length(word: str) -> int:
    """返回一个英文单词的字母个数。"""
    return len(word)

tools = [add, get_word_length]

# 2) Agent 需要一个含 agent_scratchpad 占位符的 Prompt（存放中间步骤）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手，必要时调用工具，不要凭空计算。"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # Agent 记录工具调用过程
])

# 3) 组装 Agent 与执行器
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4) 模型会自己决定先后调用哪些工具
result = executor.invoke({"input": "单词 'strawberry' 有几个字母？再把这个数和 100 相加。"})
print(result["output"])   # 应得到 110，全程靠工具而非模型心算
```

如果只想看"模型决定调什么工具"（不自动执行），用 `bind_tools`：

```python
llm_with_tools = llm.bind_tools(tools)
ai_msg = llm_with_tools.invoke("3 加 5 等于几？")
print(ai_msg.tool_calls)  # [{'name': 'add', 'args': {'a': 3, 'b': 5}, ...}]
# 注意：此时模型只是"返回了调用意图"，执行 add 是你程序的事
```

### 📚 推荐资料

- [How to use tools（官方）](https://python.langchain.com/docs/how_to/tool_calling/) — bind_tools 与 @tool 的权威指南。
- [Tool calling agent 教程](https://python.langchain.com/docs/how_to/agent_executor/) — create_tool_calling_agent + AgentExecutor。
- [LangChain Tools 概念](https://python.langchain.com/docs/concepts/tools/) — 工具/工具调用的设计模型。
- [ReAct 论文](https://arxiv.org/abs/2210.03629) — 理解 Agent "推理+行动"循环的思想源头。

### 🛠 动手实践

定义两个工具：一个查询"当前时间"，一个做"简单四则运算"（用 Python 安全地算，别用 `eval` 全量执行）。组装成 `AgentExecutor`，提问类似"现在几点？再算一下 23×17 等于多少"，开 `verbose=True` 观察 Agent 的决策轨迹。验收标准：能看到模型自主选择并按顺序调用两个工具，最终给出正确合并答案。

## 第 5 节：LlamaIndex 的定位，与"何时不该用框架"

### 📖 知识点

学了 LangChain，很多人会困惑：**LlamaIndex 又是什么？和 LangChain 是竞品吗？** 准确的理解是：两者有重叠，但**侧重点不同**。

- **LangChain** 是**通用编排框架**，强项是把任意组件（模型、工具、Agent、流程）灵活组合成管道，适合做 Agent、复杂工作流、多步骤应用。
- **LlamaIndex** 起家于、且至今最擅长 **RAG / 数据索引**。它围绕"如何把你的私有数据高效地灌给 LLM"做了大量开箱即用的封装：丰富的数据连接器（LlamaHub）、节点（Node）抽象、索引结构、查询引擎。在"快速搭一个高质量文档问答"这件事上，LlamaIndex 往往更省心、默认效果更好。

一句话选型直觉：**做复杂 Agent / 工作流编排，偏 LangChain；做以检索为核心的知识库问答，LlamaIndex 上手更快。** 而且二者可混用（LlamaIndex 可作为 LangChain 的检索器）。stage-03 的 RAG 我们两边的思想都会用到。

更重要的是要建立"**框架成本意识**"——**何时不该用框架**：

- **需求极简单**：只是单次调模型、套个固定 Prompt，裸 SDK 几行就够，引入框架反而增加依赖和认知负担。
- **要极致控制/极致性能**：框架的抽象层会隐藏细节、带来开销，对延迟和 token 极敏感的场景，手写更可控。
- **团队要长期维护**：LangChain 版本演进快（0.1→0.3 有过破坏性变更），要评估升级成本。
- **调试困难时**：当你发现大量时间花在"和框架抽象搏斗"而非业务上，就是信号。

> **专家心法**：成熟工程师的判断不是"用不用框架"，而是"**用框架的哪一部分**"。常见的高性价比姿势：用 LangChain 的 LCEL + 文档加载器 + 向量库集成（省事的部分），但 Agent 的核心循环、关键 Prompt 自己掌控（要控制的部分）。框架是工具箱，不是信仰。先能徒手用裸 SDK 实现一遍，再用框架，你才知道框架替你省了什么、藏了什么。

### 💻 代码示例

用 LlamaIndex 搭一个"五行代码级"的文档问答，感受它在 RAG 上的开箱即用（细节原理在 stage-03 展开）：

```python
# pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1) 读取 ./data 目录下所有文档（自动识别 pdf/txt/md 等）
documents = SimpleDirectoryReader("./data").load_data()

# 2) 一行建立向量索引（内部自动完成切分→Embedding→建库）
index = VectorStoreIndex.from_documents(documents)

# 3) 得到一个查询引擎并提问
query_engine = index.as_query_engine()
response = query_engine.query("这些文档讲了什么核心观点？")
print(response)
print(response.source_nodes)   # 还能拿到引用来源（溯源，stage-03 重点）
```

对比一下"何时裸 SDK 更合适"——固定单次任务，没必要上任何框架：

```python
from openai import OpenAI
client = OpenAI()
# 就一个固定翻译任务，三行搞定，引入框架纯属负担
def translate(text):
    r = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role": "user", "content": f"翻译成英文：{text}"}])
    return r.choices[0].message.content
```

### 📚 推荐资料

- [LlamaIndex 官方文档](https://docs.llamaindex.ai/en/stable/) — RAG/索引框架全貌。
- [LlamaIndex Starter Tutorial](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/) — 五分钟跑通文档问答。
- [LangChain vs LlamaIndex（官方 LlamaIndex 视角对比）](https://docs.llamaindex.ai/en/stable/getting_started/concepts/) — 理解定位差异。
- [LlamaHub 数据连接器](https://llamahub.ai/) — 几百个开箱即用的数据源接入。

### 🛠 动手实践

挑同一个小需求（如"给定一段产品介绍，生成 3 条卖点"），分别用 (1) 裸 OpenAI SDK、(2) LangChain LCEL 两种方式各实现一遍，记录代码行数、依赖数量、可读性。再写一段 200 字的判断笔记：这个需求该不该用框架，理由是什么。验收标准：能基于亲手对比，说清框架带来的收益与成本，而非人云亦云。

## 🎯 阶段实战项目：带记忆与工具的多轮对话机器人

**背景**：你要做一个命令行版的智能助手 "DevBot"，它能连续多轮对话、记住上下文，并且在需要时自动调用工具（查时间、算数、查询一个本地"知识小词典"），而不是靠模型瞎编。

**要求**：

1. **核心管道用 LCEL 搭建**：`ChatPromptTemplate | ChatOpenAI`，禁止用已废弃的 `ConversationBufferMemory` 等 legacy Memory 类。
2. **多轮记忆**：用 `RunnableWithMessageHistory` 按 `session_id` 管理历史；用 `trim_messages` 把历史控制在固定 token 内（防止越聊越贵）。
3. **工具调用**：至少实现 3 个 `@tool`（如 `get_current_time`、`calculator`、`lookup_term`——后者查一个你预置的术语字典 dict）。用 `create_tool_calling_agent` + `AgentExecutor` 组装，让模型自主决定何时调工具。
4. **流式输出**：最终回答用 `stream` 实现打字机效果。
5. **CLI 交互**：支持 `/new` 开新会话（换 session_id）、`/exit` 退出；每轮打印本次 token 用量。

**技术要点**：LCEL 组合、`RunnableWithMessageHistory`、`MessagesPlaceholder`、`@tool` + docstring 设计、`AgentExecutor` 循环、`trim_messages` 成本控制、流式。

**验收标准**：
- 能多轮记住用户信息（如名字、之前问过的内容）；
- 问"现在几点 + 算个数"时，能看到 Agent 自主、按序调用工具并给出正确合并结果；
- `/new` 后历史清空、互不污染；
- 历史超过 token 上限时自动裁剪旧消息，程序不崩。

**可选进阶**：
- 把会话历史持久化到 SQLite（替换内存 dict 的 `get_session_history`），重启程序后还能续上之前的会话；
- 加一个"摘要记忆"：当历史超过 N 轮时，用模型把早期对话压缩成一句摘要存入 system，对比纯截断的 token 节省效果；
- 用 `with_fallbacks` 给主模型配一个备用模型，主模型超时/报错时自动降级。
