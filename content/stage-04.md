---
stage: 4
title: "Agent & MCP · 智能体"
level: "进阶"
duration: "2 周"
goal: "理解 Agent 的本质是「LLM + 工具 + 循环」，掌握 Function Calling、ReAct、MCP 与 LangGraph，能构建可控、可观测的有状态智能体"
prereq: "完成 stage-01~03（Prompt 工程、结构化输出、RAG），熟悉 Python 异步与类型标注"
stack: ["Python", "OpenAI SDK", "LangGraph", "MCP (Model Context Protocol)", "Pydantic", "SQLite"]
---

## 第 1 节：Agent 的本质——从「一次问答」到「推理-行动循环」

### 📖 知识点

很多人把 Agent 想得很玄，其实它的内核异常简单：**Agent = LLM + 一组工具 + 一个循环**。普通的 Chat 调用是「问一句、答一句」的单次映射；而 Agent 是把 LLM 放进一个 `while` 循环里，让它**反复地「观察现状 → 决定下一步 → 调用工具 → 看结果 → 再决定」**，直到任务完成。LLM 在这里扮演的不是「答案生成器」，而是「决策大脑（policy）」——它每一轮的输出是一个**动作（action）**，而不是最终答案。

最经典的形式化叫 **ReAct（Reasoning + Acting）**，由 Yao 等人在 2022 年提出。它的精髓是让模型把「思考」和「行动」**交替显式输出**：

- **Thought（推理）**：我现在知道什么？还缺什么？下一步该干嘛？
- **Action（行动）**：调用某个工具，传入参数（比如 `search("2024 GDP")`）。
- **Observation（观察）**：工具返回的结果被拼回上下文。
- 然后回到 Thought，循环往复，直到模型输出 `Final Answer`。

**Why——为什么需要这个循环？** 因为 LLM 的参数记忆是「冻结在训练时刻的、不可靠的」。它不知道今天的天气、查不了你的数据库、算不准复杂乘法。但只要给它「能查、能算、能执行」的工具，再给它「多走几步、看反馈再修正」的循环，它就能把一个模糊的大任务拆解、执行、纠错，逼近一个真正的「能动手做事的系统」。**单次问答是静态的知识检索，Agent 是动态的问题求解。**

**How——现代实现的演进。** 早期 ReAct 是靠纯文本 Prompt 约定格式（让模型输出 `Thought:/Action:` 字符串，再用正则解析），脆弱且易跑偏。2023 年后，主流厂商把「调用工具」做成了模型原生能力——**Function Calling / Tool Calling**（下一节详讲），模型直接吐出结构化的工具调用请求，不再靠正则抠字符串。所以今天写 ReAct，**底层用 Tool Calling，循环逻辑自己控制或交给框架（如 LangGraph）**。

> **专家心法**：判断一个系统是不是「真 Agent」，就看它有没有那个**反馈循环**。把 LLM 包一层「调一次工具就返回」的代码，那只是「带工具的问答」；只有当模型能根据上一步的 Observation **自主决定要不要、以及如何走下一步**时，它才真正具备了 agency（能动性）。循环的「终止条件」和「最大步数」是工程上最先要守住的两条命。

### 💻 代码示例

不依赖任何框架，用 OpenAI 原生 Tool Calling 手写一个最小 ReAct 循环，把「循环」这件事彻底看清楚。

```python
# pip install openai
import json
from openai import OpenAI

client = OpenAI()

# 1) 定义两个真实可用的本地工具
def calculator(expression: str) -> str:
    """安全地计算一个数学表达式。"""
    try:
        # 仅允许数字与运算符，避免 eval 注入
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "error: illegal characters"
        return str(eval(expression))  # 教学用；生产请用 ast 解析
    except Exception as e:
        return f"error: {e}"

def get_word_length(word: str) -> str:
    """返回一个单词的字母数。"""
    return str(len(word))

TOOLS_IMPL = {"calculator": calculator, "get_word_length": get_word_length}

# 2) 用 JSON Schema 把工具「描述」给模型（这就是 Tool Calling 的接口）
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，如 '23*47'",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_word_length",
            "description": "返回单词的字母个数",
            "parameters": {
                "type": "object",
                "properties": {"word": {"type": "string"}},
                "required": ["word"],
            },
        },
    },
]

def run_agent(question: str, max_steps: int = 6):
    messages = [
        {"role": "system", "content": "你是一个会用工具的助手。需要计算或数字母时必须调用工具，不要心算。"},
        {"role": "user", "content": question},
    ]
    for step in range(max_steps):          # ← 守住最大步数，防死循环
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",            # 让模型自己决定调不调、调哪个
            temperature=0,
        )
        msg = resp.choices[0].message
        messages.append(msg)               # 把模型这一轮的输出放回上下文

        if not msg.tool_calls:             # ← 终止条件：模型不再要工具，给出最终答案
            return msg.content

        # 模型要求调用工具：逐个执行，把 Observation 拼回去
        for call in msg.tool_calls:
            fn = TOOLS_IMPL[call.function.name]
            args = json.loads(call.function.arguments)
            result = fn(**args)
            print(f"[step {step}] {call.function.name}({args}) -> {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,    # 必须回填对应的 id
                "content": result,
            })
    return "达到最大步数仍未完成"

print(run_agent("单词 'strawberry' 有几个字母？再把这个数字乘以 17 是多少？"))
# 模型会先调 get_word_length('strawberry')->10，再调 calculator('10*17')->170
```

跑一遍你会清楚地看到 `[step 0] get_word_length(...)` → `[step 1] calculator(...)` → 最终答案，这就是 ReAct 循环在真实运转。

### 📚 推荐资料

- [ReAct: Synergizing Reasoning and Acting in Language Models (论文)](https://arxiv.org/abs/2210.03629) — ReAct 范式的原始论文，必读。
- [OpenAI Function Calling 指南](https://platform.openai.com/docs/guides/function-calling) — Tool Calling 的权威接口文档。
- [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — OpenAI 研究员对 Agent 架构（规划/记忆/工具）的系统综述，业内必读。
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 用工程视角讲清 workflow 与 agent 的区别，反对过度设计。

### 🛠 动手实践

基于上面的最小循环，给 Agent 再加一个 `get_current_time()` 工具（返回当前时间），然后问它一个需要「先查时间、再算时间差」的复合问题。验收标准：能在日志里看到模型**分两步**调用了不同工具，且最终答案正确；同时故意把 `max_steps` 设成 1，观察并解释为什么会得到「未完成」。

## 第 2 节：Function Calling / Tool Calling——给 LLM 装上手

### 📖 知识点

Function Calling（OpenAI 现已统称 Tool Calling）是 Agent 时代最重要的一块基础设施。它解决的问题是：**如何让一个「只会生成文本」的模型，可靠地表达「我想调用某个函数、参数是这些」的意图？**

关键要厘清一个常见误解：**模型本身不会执行任何代码**。整个流程是「模型提议、你来执行」：

1. 你把工具的「名字、用途、参数 schema」用 JSON Schema 描述好，随请求一起发给模型。
2. 模型分析用户意图，如果觉得需要某个工具，**不会**返回普通文本，而是返回一个结构化的 `tool_calls`（含函数名 + JSON 参数）。
3. **你的代码**负责真正执行这个函数，拿到结果。
4. 你把结果作为 `role: "tool"` 消息塞回对话，模型据此生成最终回答或决定下一步。

**Why——为什么这是「原生能力」而非「Prompt 技巧」？** 因为模型在后训练阶段专门用大量「工具调用」数据微调过，被对齐成「该调工具时就吐结构化 JSON、参数符合 schema」。这比你用 Prompt 求它「请返回 JSON」可靠得多——参数类型、必填项、枚举值都能被模型较好地遵守。这也是为什么要尽量用平台原生的 `tools` 参数，而不是自己在 Prompt 里发明一套格式。

**How——工程要点：**

- **并行调用（parallel tool calls）**：现代模型一轮可以同时请求多个工具调用（比如同时查三个城市的天气），`tool_calls` 是个数组，要遍历执行。
- **`tool_choice`**：`"auto"`（模型自己决定）、`"required"`（必须调某个工具）、或指定某个函数名（强制）。
- **描述即接口**：`description` 字段是模型理解「何时该用这个工具」的唯一依据。写得含糊，模型就乱调或漏调。**好的工具描述 = 好的 API 文档**。
- **错误要回传**：工具执行失败时，别抛异常中断循环，而要把错误信息作为 observation 回传给模型，让它有机会换参数重试。

> **专家心法**：工具设计要遵循「**少而正交**」原则。给模型 20 个功能重叠的工具，它会选择困难、频繁调错；给 5 个职责清晰、互不重叠的工具，它表现反而更好。另外，**工具的返回值也是 Prompt**——返回一坨杂乱 JSON，不如返回模型易读的精炼文本。把工具的输入输出都当成「和模型对话的接口」来设计。

### 💻 代码示例

用 Pydantic 自动生成工具的 JSON Schema（避免手写易错），并演示「并行调用」。

```python
# pip install openai pydantic
import json
from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI()

# 1) 用 Pydantic 定义参数，model_json_schema() 自动产出 schema
class WeatherArgs(BaseModel):
    city: str = Field(description="城市名，如 '北京'")
    unit: str = Field(default="celsius", description="celsius 或 fahrenheit")

def get_weather(city: str, unit: str = "celsius") -> str:
    # 假数据，演示用
    data = {"北京": 5, "上海": 12, "广州": 22}
    temp = data.get(city, 20)
    return json.dumps({"city": city, "temp": temp, "unit": unit}, ensure_ascii=False)

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询某城市当前气温",
        "parameters": WeatherArgs.model_json_schema(),  # ← 自动生成，不手写
    },
}]

messages = [{"role": "user", "content": "北京、上海、广州现在分别多少度？"}]
resp = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=tools, temperature=0,
)
msg = resp.choices[0].message
messages.append(msg)

# 2) 模型一次性返回 3 个并行 tool_calls，遍历执行
print(f"模型请求了 {len(msg.tool_calls)} 个工具调用")
for call in msg.tool_calls:
    args = WeatherArgs(**json.loads(call.function.arguments))  # Pydantic 顺带校验
    result = get_weather(args.city, args.unit)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

# 3) 把结果喂回去，拿自然语言总结
final = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=tools, temperature=0,
)
print(final.choices[0].message.content)
```

### 📚 推荐资料

- [OpenAI Function Calling 指南](https://platform.openai.com/docs/guides/function-calling) — 含并行调用、`tool_choice`、严格模式（strict）的完整说明。
- [Anthropic Tool Use 文档](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — Claude 的工具调用机制，对比理解更深刻。
- [JSON Schema 官方规范](https://json-schema.org/learn/getting-started-step-by-step) — 工具参数描述的底层标准，写复杂参数时必查。
- [Pydantic: model_json_schema](https://docs.pydantic.dev/latest/concepts/json_schema/) — 用 Pydantic 自动生成工具 schema 的官方说明。

### 🛠 动手实践

设计一组「故意有歧义」的工具（如同时有 `search_web` 和 `search_db`，描述写得含糊），观察模型选错工具的情况；然后只优化 `description` 字段（不改代码逻辑），让模型选对。验收标准：用前后对比证明「工具描述质量直接决定调用准确率」，并写下你总结的 3 条工具描述写作准则。

## 第 3 节：MCP（Model Context Protocol）——工具生态的「USB-C」

### 📖 知识点

上一节我们手写工具定义，问题来了：**如果每个团队、每个工具都自己发明一套接入方式，会怎样？** 你为 GitHub 写一套工具适配、为 Slack 写一套、为数据库再写一套，换个 Agent 框架又得全部重写——这就是 2023~2024 年 Agent 工具生态的混乱现状：**N 个模型 × M 个工具 = N×M 套定制集成**，没有标准。

**MCP（Model Context Protocol）** 是 Anthropic 在 2024 年 11 月开源的协议，目标就是终结这种混乱。一个广为流传的比喻：**MCP 是 AI 应用的「USB-C 接口」**。在 USB-C 之前，每种设备一种充电口；之后，一个标准口通吃。MCP 想做的就是：**定义一套标准协议，让任何「工具/数据源」只要实现一次 MCP Server，就能被任何支持 MCP 的「AI 应用」即插即用地接入**，把 N×M 变成 N+M。

**核心架构（三个角色）：**

- **MCP Host**：用户直接使用的 AI 应用（如 Claude Desktop、Cursor、你自己的 Agent）。
- **MCP Client**：Host 内部为「每一个」Server 维护的连接器，一对一通信。
- **MCP Server**：轻量程序，对外暴露三类能力——**Tools（可调用的函数）**、**Resources（可读取的数据，如文件/数据库记录）**、**Prompts（预设的提示词模板）**。

通信走 **JSON-RPC 2.0**，传输层支持 `stdio`（本地子进程，最常用）和基于 HTTP 的 `Streamable HTTP`（远程）。

**Why——为什么它重要到值得单开一节？** 三个层面：

1. **生态层面**：MCP 已被 OpenAI、Google DeepMind、微软等广泛采纳，正在成为事实标准。社区已有数千个现成 MCP Server（GitHub、Postgres、文件系统、Slack、浏览器……），意味着你的 Agent **可以「站在生态肩膀上」**，而不是事事重造轮子。
2. **解耦层面**：工具的实现方（Server）和使用方（Agent）彻底解耦。后端换数据库、加新工具，Agent 侧零改动。
3. **安全与边界**：Server 是独立进程/服务，权限、密钥都收敛在 Server 内，Agent 不直接接触底层凭证。

> **专家心法**：不要把 MCP 理解成「又一个框架」——它是**协议**，类比 HTTP 之于 Web。判断要不要为某个能力写 MCP Server 的标准是：**这个能力会被多个 Agent / 多个应用复用吗？** 一次性的内部工具，直接 Function Calling 就够了；需要跨应用复用、或想接入海量现成生态时，MCP 才是正解。别为了用 MCP 而 MCP。

### 💻 代码示例

用官方 `mcp` Python SDK 写一个最小 MCP Server（暴露一个工具 + 一个资源），并用客户端连接调用。

```python
# pip install "mcp[cli]"
# ---------- server.py：一个标准 MCP Server ----------
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")   # 给 Server 起个名

# 用装饰器暴露一个「工具」——签名和 docstring 会自动转成 schema 给模型
@mcp.tool()
def add(a: int, b: int) -> int:
    """把两个整数相加并返回结果。"""
    return a + b

# 暴露一个「资源」——可被 Host 读取的数据，用 URI 模板寻址
@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """返回对某人的问候语。"""
    return f"Hello, {name}! 欢迎使用 MCP。"

if __name__ == "__main__":
    mcp.run(transport="stdio")   # 以 stdio 方式运行，由 Host 作为子进程拉起
```

```python
# ---------- client.py：连接上面的 Server 并调用 ----------
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 告诉客户端如何启动 server（这里就是跑 python server.py）
    params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1) 发现：这个 Server 提供了哪些工具？
            tools = await session.list_tools()
            print("可用工具:", [t.name for t in tools.tools])

            # 2) 调用工具
            result = await session.call_tool("add", {"a": 23, "b": 19})
            print("add(23,19) =", result.content[0].text)

            # 3) 读取资源
            res = await session.read_resource("greeting://Alice")
            print("资源内容:", res.contents[0].text)

asyncio.run(main())
```

把 `list_tools()` 拿到的工具 schema 转交给 LLM 的 `tools` 参数，就把 MCP Server 接进了上一节的 Agent 循环——**这就是「即插即用」的含义**。

### 📚 推荐资料

- [MCP 官方文档](https://modelcontextprotocol.io/) — 协议规范、架构、快速上手，第一手资料。
- [MCP Python SDK (GitHub)](https://github.com/modelcontextprotocol/python-sdk) — `FastMCP` 的源码与完整示例。
- [Anthropic: Introducing MCP](https://www.anthropic.com/news/model-context-protocol) — 官方发布博客，讲清设计动机。
- [Awesome MCP Servers](https://github.com/modelcontextprotocol/servers) — 官方维护的现成 Server 列表（数据库、文件、Git 等），直接拿来用。

### 🛠 动手实践

写一个 MCP Server，暴露一个 `query_notes(keyword)` 工具，从本地一个 JSON 文件里检索笔记。然后在 Claude Desktop 或 Cursor 的 MCP 配置中注册它，在真实 AI 应用里调用你的 Server。验收标准：在 Host 应用里输入自然语言，能触发你的工具并返回检索结果——亲手体验「一次实现、任意 Host 接入」。

## 第 4 节：LangGraph——构建有状态、可控的 Agent

### 📖 知识点

手写 `while` 循环跑小 Demo 没问题，但真实 Agent 会面临：多分支决策、需要持久化记忆、要支持人工介入（human-in-the-loop）、出错要能从断点恢复、还要可观测可调试。这时就需要专业框架。**LangGraph** 是目前生产级 Agent 的主流选择，它的核心抽象不是「链」，而是**图（Graph）**——把 Agent 的工作流建模成一张**状态机**。

三个核心概念：

- **State（状态）**：一个贯穿全程的共享数据结构（通常是 `TypedDict`），每个节点读它、改它。这是 Agent 的「记忆」与「黑板」。LangGraph 用 **reducer**（如 `add_messages`）定义状态字段如何更新（追加还是覆盖）。
- **Node（节点）**：一个普通 Python 函数，输入当前 State，返回对 State 的更新。一个节点可以是「调用 LLM」「执行工具」「做检索」等任意逻辑。
- **Edge（边）**：连接节点，定义流转。普通边是固定跳转；**Conditional Edge（条件边）** 是关键——它根据当前 State 动态决定下一个去哪个节点，**这就是 Agent「自主决策」在图上的落地**（比如「模型要调工具就去 tools 节点，否则去 END」）。

**Why——为什么是「图」而不是「链」？** 因为 Agent 的本质是**带环、带分支的控制流**：要循环（调工具→看结果→再决策）、要分叉（不同情况走不同路）、有时要回头。线性的 Chain 表达不了「环」，而图天生就能。把 Agent 画成图后，**整个决策流程变得显式、可视、可断点**——你能看到它卡在哪、为什么这么走，调试体验天差地别。

**How——关键能力：**

- **Checkpointer（检查点）**：给图配上 `checkpointer`，State 会在每步自动持久化。这带来三大能力：**多轮对话记忆**（用 `thread_id` 区分会话）、**断点恢复**、**human-in-the-loop**（在某节点暂停，等人类输入/审批再继续）。
- **`StateGraph`**：定义图的入口；`add_node` / `add_edge` / `add_conditional_edges` 搭结构；`compile()` 编译成可执行 app。
- 现代 LangGraph 也提供 `create_react_agent` 这类预制件，但理解底层的图模型，你才能定制复杂流程。

> **专家心法**：用 LangGraph 时，**先画图再写码**。在纸上把节点和边画清楚——哪些是确定流转、哪些是条件分支、循环的出口在哪、终止条件是什么。图画对了，代码是水到渠成的翻译；图没想清楚就堆代码，必然写出一团乱麻的死循环。**State 的设计是第一要务**：想清楚「整个流程需要共享哪些信息」，State 就是 Agent 的架构。

### 💻 代码示例

用 LangGraph 标准模式搭一个带工具循环的 Agent，并加上 checkpointer 实现多轮记忆。

```python
# pip install langgraph langchain-openai
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1) 定义工具
@tool
def multiply(a: int, b: int) -> int:
    """两个整数相乘。"""
    return a * b

@tool
def get_population(city: str) -> str:
    """查询城市人口（万人）。"""
    return str({"北京": 2184, "上海": 2487}.get(city, 1000))

tools = [multiply, get_population]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# 2) 定义 State：messages 字段用 add_messages reducer，表示「追加」而非「覆盖」
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 3) 定义节点：调用 LLM
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 4) 组装图
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))   # 预制的工具执行节点

builder.add_edge(START, "chatbot")
# 条件边：tools_condition 检查 LLM 是否请求了工具
#   是 -> 去 "tools" 节点执行；否 -> 去 END
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")          # 工具执行完回到 chatbot，形成循环

# 5) 编译，挂上 checkpointer 实现记忆
app = builder.compile(checkpointer=MemorySaver())

# 6) 运行：用 thread_id 标识一次会话，状态会被持久化
cfg = {"configurable": {"thread_id": "user-1"}}
out = app.invoke(
    {"messages": [("user", "北京人口是多少万？再乘以 3 是多少？")]}, cfg
)
print(out["messages"][-1].content)

# 第二轮：因为有 checkpointer，它「记得」上一轮的北京人口
out2 = app.invoke({"messages": [("user", "那上海呢？同样乘以 3")]}, cfg)
print(out2["messages"][-1].content)
```

注意第二轮无需重复上下文——`thread_id` + checkpointer 让 Agent 自动拥有了记忆。这就是「有状态」的价值。

### 📚 推荐资料

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) — 概念、教程、API，主入口。
- [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/) — 从零搭建带工具与记忆的 Agent，对应本节代码。
- [LangGraph: Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — 生产级 Agent 必备的人工审批/介入机制。
- [LangSmith](https://docs.smith.langchain.com/) — 配套的 Agent 可观测性平台，看清每一步的输入输出与 token 消耗。

### 🛠 动手实践

把第 1 节手写的 ReAct 循环用 LangGraph 重写，并新增一个需求：**当工具连续调用超过 3 次还没结束时，强制走向一个「兜底节点」返回「任务过于复杂」**（用条件边 + State 里的计数器实现）。验收标准：能画出你的状态图，并演示死循环被兜底机制成功拦截。

## 第 5 节：多 Agent 协作、Planning/Reflection 与避坑指南

### 📖 知识点

单个 Agent 有上限：工具一多就选择困难、上下文一长就「失忆」、职责一杂就表现下降。解法是**多 Agent 协作**——把一个大任务拆给多个「各有专长、上下文隔离」的 Agent。常见模式：

- **Supervisor（主管模式）**：一个 supervisor Agent 负责「调度」，把子任务路由给专长 worker（如 researcher、coder、writer），收集结果再汇总。最常用、最好控制。
- **Network（网络/对等模式）**：Agent 之间可互相移交（hand-off），更灵活但更难控、易失控。
- **Hierarchical（分层）**：supervisor 之上再套 supervisor，应对超大型任务。

两个让单 Agent 更聪明的关键技巧：

- **Planning（规划）**：让 Agent 先「制定计划」再执行（Plan-and-Execute）。先产出一个步骤清单，再逐步执行并按需重规划。好处是把「长程任务」结构化，减少走一步看一步的短视。
- **Reflection（反思）**：让 Agent 执行后**自我批判**——「我这个答案/代码对吗？有没有更好的？」典型如 Reflexion、Self-Refine：生成→自评→根据批评修订。对写代码、写长文这类「能验证、可迭代」的任务提升明显。

**Agent 的常见坑（专家最关心的部分）：**

1. **死循环 / 无限震荡**：模型在两个工具间反复横跳，或一直调同一个工具。**必须**设最大步数（max iterations）作为硬性熔断。
2. **幻觉调用（hallucinated tool call）**：模型调用不存在的工具、或编造参数（如瞎填一个数据库表名）。对策：工具描述精确、参数用 schema 严格约束、对非法调用返回明确报错让其纠正。
3. **上下文爆炸**：循环多轮后，历史 observation 堆满上下文，既贵又让模型分心。对策：对工具结果做摘要、滑动窗口裁剪、或把长结果存外部再用引用。
4. **错误雪崩**：某一步工具出错，模型没意识到，基于错误结果继续推理。对策：工具返回结构化的成功/失败标志，并在 System Prompt 里要求「遇到 error 先停下分析」。
5. **过度自主**：能用确定性代码（if-else、固定流程）解决的，别交给 LLM 决策。**LLM 决策有成本、有不确定性**。

> **专家心法**：Anthropic 在《Building Effective Agents》里的核心忠告值得刻在脑子里——**能用简单 workflow 解决的，绝不上自主 Agent**。多数生产需求其实是「固定流程 + 几个 LLM 节点」的 workflow，而非需要模型自由决策的 agent。自主性越高，越不可控、越贵、越难调试。**先问「这步真的需要模型自己决定吗」，把不确定性压到最小**——这是区分玩具和生产系统的分水岭。

### 💻 代码示例

用 LangGraph 的 supervisor 思路，实现一个「主管 + 两个专家」的极简多 Agent 路由。

```python
# pip install langgraph langchain-openai
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

# 1) Supervisor：用结构化输出决定把任务交给谁
class Route(BaseModel):
    next: Literal["math_expert", "writer", "FINISH"]

def supervisor(state: State):
    sys = ("你是主管。根据对话决定下一步交给谁：math_expert 处理计算，"
           "writer 处理写作。任务完成则返回 FINISH。")
    decision = llm.with_structured_output(Route).invoke(
        [("system", sys)] + state["messages"]
    )
    return {"next": decision.next}

# 2) 两个专家节点
def math_expert(state: State):
    r = llm.invoke([("system", "你是数学专家，给出计算结果。")] + state["messages"])
    return {"messages": [r], "next": "supervisor"}

def writer(state: State):
    r = llm.invoke([("system", "你是文案专家，润色成一句话。")] + state["messages"])
    return {"messages": [r], "next": "supervisor"}

# 3) 路由函数：根据 supervisor 的决定跳转
def route(state: State) -> Literal["math_expert", "writer", "__end__"]:
    return END if state["next"] == "FINISH" else state["next"]

builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("math_expert", math_expert)
builder.add_node("writer", writer)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route)
builder.add_edge("math_expert", "supervisor")   # 专家干完回主管
builder.add_edge("writer", "supervisor")

app = builder.compile()
out = app.invoke({"messages": [HumanMessage("帮我算 25*8，然后用一句话夸夸这个结果")], "next": ""})
for m in out["messages"]:
    print(f"{m.type}: {m.content}")
```

### 📚 推荐资料

- [LangGraph Multi-Agent 概念](https://langchain-ai.github.io/langgraph/concepts/multi_agent/) — supervisor/network/hierarchical 各模式详解。
- [Reflexion 论文](https://arxiv.org/abs/2303.11366) — 让 Agent 通过语言反馈自我反思、迭代改进的经典工作。
- [Plan-and-Solve Prompting 论文](https://arxiv.org/abs/2305.04091) — Planning 范式的代表性研究。
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 关于「何时该用 agent、何时该用 workflow」的最佳实践，避坑必读。

### 🛠 动手实践

给上面的多 Agent 系统加一个 Reflection 节点：writer 产出后，由一个 critic 节点打分（1~5），低于 4 分则打回 writer 重写（最多重写 2 次，用 State 计数防死循环）。验收标准：日志能展示「生成→批评→重写→通过」的完整反思闭环，且重写次数被正确限制。

## 🎯 阶段实战项目：能连数据库的数据分析 Agent

**背景**：业务方不会写 SQL，但想用自然语言问数据——「上个月哪个品类销量最高？」「华东区客单价同比涨了多少？」。你要构建一个 Agent，把自然语言问题翻译成 SQL、查库、并给出带解读的答案。

**要求**：

1. **数据准备**：用 SQLite 建一个 `sales` 库（含 orders、products、regions 几张表，灌入示例数据）。
2. **工具设计**（至少三个，职责正交）：
   - `list_tables()`：返回所有表名。
   - `get_schema(table)`：返回某表的字段与类型。
   - `run_sql(query)`：执行**只读** SQL（必须拦截 `DROP/DELETE/UPDATE/INSERT`，只允许 `SELECT`）并返回结果。
3. **用 LangGraph 搭建 ReAct 循环**：Agent 要能「先看有哪些表 → 看相关表结构 → 写 SQL → 执行 → 若报错则根据错误信息修正重试」。
4. **守住安全与稳定**：SQL 注入/危险操作拦截；最大步数熔断（防死循环）；SQL 出错时把数据库报错回传给模型纠正。
5. **输出**：不仅给数据，还要给一句话业务解读。

**技术要点**：Tool Calling、LangGraph 状态图与条件边（循环）、错误回传与自我修正、只读 SQL 安全沙箱、最大步数熔断、（可选）用 checkpointer 支持追问。

**验收标准**：

- 对 5 个不同复杂度的自然语言问题（含需要 JOIN 的），能稳定给出正确 SQL 与答案。
- 故意问一个「表里没有的字段」，Agent 能根据报错自我修正或如实说明查不到，**而不是幻觉编造数据**。
- 尝试注入 `'; DROP TABLE orders; --`，被安全层成功拦截。

**可选进阶**：

1. 把三个工具改造成一个 **MCP Server**（`sqlite-mcp`），让 Agent 通过 MCP 协议接入——体会「工具与 Agent 解耦」。
2. 加 human-in-the-loop：执行 SQL 前，把生成的 SQL 展示给用户确认再跑（用 LangGraph 的 interrupt 机制）。
3. 接入真实的开源 [SQLite MCP Server](https://github.com/modelcontextprotocol/servers)，对比自己实现与生态方案的差异。
