---
stage: 1
title: "地基 · LLM 原理与 Prompt 工程"
level: "基础"
duration: "1 周"
goal: "理解 LLM 到底在做什么，掌握工程化的提示词设计与稳定的 API 调用"
prereq: "Python 基础、会用 pip 与虚拟环境"
stack: ["Python", "OpenAI SDK", "Pydantic", "tiktoken"]
---

## 第 1 节：LLM 到底在做什么——下一个 Token 预测

### 📖 知识点

大语言模型（LLM）的本质，比大多数人想象的要朴素：**它是一个在海量文本上训练出来的「下一个 token 概率预测器」**。给定一段文本（上下文），模型输出的不是"答案"，而是词表里每一个 token 作为下一个词的概率分布，然后按某种策略采样一个出来，再把它拼回上下文，循环往复。所谓"生成一段话"，其实是这个过程重复了几百上千次。

理解这一点能解释很多现象：

- **为什么会"幻觉"**：模型优化的目标是"下一个词读起来通顺、符合训练分布"，而不是"事实正确"。流畅 ≠ 真实，这是机制层面的，不是 bug。
- **为什么 Prompt 如此重要**：你的输入就是模型用来计算概率分布的"上下文"。上下文写得好，高质量回答的概率就高。Prompt 工程本质是在**操纵条件概率**。
- **什么是 Token**：模型不是按"字"或"词"处理，而是按 token（子词单元）。英文里一个 token 约 0.75 个单词，中文里一个汉字通常是 1~2 个 token。这直接关系到**成本（按 token 计费）**和**上下文窗口限制**。

> **专家心法**：永远记住模型是"概率机器"而非"知识库"。当你需要它给出确定、可溯源的事实时，不要指望它的参数记忆，而要通过 RAG（后续阶段）把事实喂进上下文。

### 💻 代码示例

先把环境和"最小可用调用"跑通。这里用 OpenAI 新版 SDK（`>=1.0`），注意旧教程里的 `openai.ChatCompletion.create` 已废弃。

```python
# pip install openai tiktoken
import os
from openai import OpenAI

# 推荐用环境变量管理密钥，不要硬编码到代码里
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "你是一个简洁的技术助手。"},
        {"role": "user", "content": "用一句话解释什么是 token。"},
    ],
)
print(resp.choices[0].message.content)
print("用量:", resp.usage)  # prompt_tokens / completion_tokens / total_tokens
```

观察 `resp.usage`——这就是你的账单。再用 `tiktoken` 看看 token 是怎么切的：

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o-mini")
text = "Hello, 世界! LLM"
tokens = enc.encode(text)
print(len(tokens), tokens)
print([enc.decode([t]) for t in tokens])  # 看每个 token 对应的片段
```

### 📚 推荐资料

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — 图解 Transformer，理解模型内部最经典的入门读物。
- [OpenAI Tokenizer 可视化](https://platform.openai.com/tokenizer) — 在线看任意文本如何被切成 token。
- [Andrej Karpathy: Let's build GPT (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 从零手写一个 GPT，彻底搞懂"下一个 token 预测"。
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) — 系统的免费课程。

### 🛠 动手实践

写一个脚本，输入任意一段中英混合文本，输出：(1) token 总数；(2) 每个 token 解码后的片段；(3) 估算用 gpt-4o-mini 调用一次的输入成本（自行查当前价格）。验收标准：能正确显示"中文一个字常占 1~2 token"这一现象。

## 第 2 节：采样参数——Temperature、Top-p 与确定性

### 📖 知识点

模型给出概率分布后，"挑哪个 token"由采样策略决定。掌握这几个参数是写出稳定应用的前提：

- **temperature（温度，0~2）**：缩放概率分布的"陡峭程度"。`temperature=0` 近似贪心解码，每次都选概率最高的 token，输出最确定、最可复现；温度越高，分布越平，输出越随机、越有创意，也越容易跑偏。**做抽取/分类/代码用低温（0~0.3），做创意写作用高温（0.7~1.0）**。
- **top_p（核采样，0~1）**：只在累积概率达到 p 的最小 token 集合里采样。`top_p=0.1` 表示只从最可能的那一小撮里选。它和 temperature 是两种控制随机性的旋钮，**实践中通常只调其中一个**，另一个保持默认。
- **max_tokens**：限制输出长度，直接影响成本和延迟。
- **seed**：部分模型支持，配合 `temperature=0` 提升可复现性（但不保证 100%）。

> **专家心法**：生产环境里，凡是需要"结构稳定、可解析"的输出（如返回 JSON），一律用 `temperature=0`。不确定性是 Demo 的浪漫，是生产的灾难。

### 💻 代码示例

```python
from openai import OpenAI
client = OpenAI()

def ask(prompt, temperature):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=60,
    )
    return r.choices[0].message.content

# 同一个 prompt，对比不同温度的"发散度"
for t in (0.0, 1.2):
    print(f"\n=== temperature={t} ===")
    for i in range(2):
        print(f"[{i}] {ask('给一家咖啡店起个名字', t)}")
# 你会看到 0.0 两次几乎一致，1.2 两次差异很大
```

### 📚 推荐资料

- [OpenAI API Reference - Chat](https://platform.openai.com/docs/api-reference/chat/create) — 所有参数的权威说明。
- [How to generate text (Hugging Face)](https://huggingface.co/blog/how-to-generate) — 贪心/beam/top-k/top-p 采样策略详解。
- [What is temperature in LLMs?](https://www.hopsworks.ai/dictionary/llm-temperature) — 温度参数直觉化解释。

### 🛠 动手实践

写一个函数，对同一个 prompt 分别用 temperature ∈ {0, 0.5, 1.0, 1.5} 各调用 3 次，把输出整理成一张表格，直观感受"温度 vs 一致性"的关系。验收标准：能用数据说明为什么抽取任务该用低温。

## 第 3 节：结构化输出——让 LLM 返回可靠的 JSON

### 📖 知识点

应用开发中，你几乎从不需要"一段自然语言"，而是需要**程序能直接解析的结构化数据**。让 LLM 稳定吐出合法 JSON 是工程化第一道坎。三种从弱到强的手段：

1. **Prompt 里要求**："请只返回 JSON，不要任何解释"——简单但不可靠，模型可能加 ```json 包裹或多余文字。
2. **JSON Mode**：`response_format={"type": "json_object"}`，模型保证输出是合法 JSON（但不保证字段符合你的 schema）。
3. **Structured Outputs / 工具约束**：用 `response_format` 传入 JSON Schema（或配合 Pydantic），模型**保证字段结构完全符合**。这是目前最可靠的方式。

配合 **Pydantic** 做校验，能把"模型输出"无缝接入你的类型系统，出错时立刻暴露。

> **专家心法**：不要用正则去"抠"模型返回的 JSON——那是脆弱的补丁。用平台原生的 Structured Outputs + Pydantic 校验，让非法输出在边界处就被拦下。

### 💻 代码示例

```python
from pydantic import BaseModel
from openai import OpenAI
client = OpenAI()

# 1) 定义你想要的结构
class Article(BaseModel):
    title: str
    tags: list[str]
    sentiment: str  # positive / neutral / negative

# 2) 用 SDK 的 parse 接口直接拿到结构化对象（Structured Outputs）
resp = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "抽取文章信息。"},
        {"role": "user", "content": "标题：AI 改变世界。内容积极向上，关于人工智能与未来。"},
    ],
    response_format=Article,   # 直接传 Pydantic 模型
    temperature=0,
)
article: Article = resp.choices[0].message.parsed
print(article.title, article.tags, article.sentiment)
print(type(article))  # <class '__main__.Article'> —— 已是强类型对象
```

### 📚 推荐资料

- [OpenAI Structured Outputs 指南](https://platform.openai.com/docs/guides/structured-outputs) — 官方权威，含 schema 限制说明。
- [Pydantic 官方文档](https://docs.pydantic.dev/latest/) — 数据校验事实标准，AI 工程必备。
- [Instructor 库](https://python.useinstructor.com/) — 跨模型的结构化输出封装，多模型场景神器。

### 🛠 动手实践

定义一个 `Resume`（简历）Pydantic 模型，含 name、years_of_experience（int）、skills（list）、是否符合某岗位（bool）。给一段自由格式的简历文本，让模型抽取并校验。验收标准：当模型返回字段类型不对时，Pydantic 能抛出清晰错误。

## 第 4 节：提示词工程实战——从 Zero-shot 到 CoT

### 📖 知识点

Prompt 工程不是"咒语玄学"，而是有章法的工程实践。核心技巧：

- **角色与指令（System Prompt）**：设定模型身份、语气、约束。把"它是谁、做什么、不做什么"讲清楚。
- **Few-shot（少样本示例）**：在 prompt 里给 2~5 个"输入→输出"范例，模型会模仿格式和风格。对格式要求严、任务偏门时极有效。
- **Chain-of-Thought（CoT，思维链）**：对需要推理的任务，加一句"让我们一步步思考"或要求模型先写推理过程再给答案，能显著提升正确率。原理是给模型更多"中间计算 token"的空间。
- **分隔符与结构化**：用 `###`、XML 标签、Markdown 把指令、上下文、数据分区，减少模型混淆。
- **明确输出格式**：与其说"简短点"，不如说"用不超过 3 个 bullet"。约束越具体越好。

> **专家心法**：调 Prompt 时遵循"控制变量"原则——一次只改一个地方，记录效果。把好用的 Prompt 当代码一样**版本管理**，这是后续 LLMOps 阶段的伏笔。

### 💻 代码示例

```python
from openai import OpenAI
client = OpenAI()

# Few-shot + CoT 组合：做一个稳定的"情感分类 + 理由"任务
SYSTEM = "你是情感分析专家，只输出指定格式。"
FEWSHOT = """示例：
输入：这家店服务太差了，再也不来。
输出：{"label": "negative", "reason": "表达不满与拒绝再次光顾"}

输入：还行吧，没什么特别的。
输出：{"label": "neutral", "reason": "评价中性、无明显倾向"}
"""

def classify(text):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{FEWSHOT}\n输入：{text}\n输出："},
        ],
        temperature=0,
    )
    return r.choices[0].message.content

print(classify("物流很快，包装也用心，超出预期！"))
# -> {"label": "positive", "reason": "..."}
```

### 📚 推荐资料

- [OpenAI Prompt Engineering 指南](https://platform.openai.com/docs/guides/prompt-engineering) — 官方六大策略。
- [Prompt Engineering Guide (DAIR.AI)](https://www.promptingguide.ai/zh) — 最全的中文提示工程指南。
- [Chain-of-Thought Prompting 论文](https://arxiv.org/abs/2201.11903) — CoT 的原始论文，理解其有效性来源。
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) — 另一视角的高质量实践。

### 🛠 动手实践

挑一个"模型容易答错"的推理题（如简单数学应用题），分别用 (1) 直接问、(2) 加"一步步思考"、(3) Few-shot 示范推理过程，三种 Prompt 各测 5 次，统计正确率。验收标准：用数据证明 CoT 对推理任务的提升。

## 🎯 阶段实战项目：智能信息抽取 CLI

**背景**：你拿到一批杂乱的客户反馈文本，需要批量结构化，导入数据库分析。

**要求**：

1. 用 Pydantic 定义反馈结构：`category`（枚举：bug/feature/praise/other）、`urgency`（1~5 int）、`summary`（一句话）、`needs_reply`（bool）。
2. 写一个命令行工具，读入一个 `.txt`（每行一条反馈），用 `temperature=0` + Structured Outputs 逐条抽取，输出为 `.jsonl`。
3. 加入错误处理：单条失败不影响整体，记录失败行。
4. 打印统计：总 token 消耗、估算成本、各 category 数量。

**技术要点**：System Prompt 设计、Structured Outputs、Pydantic 校验、usage 统计、批处理容错。

**验收标准**：给 20 条混合反馈，能稳定输出合法 jsonl，且统计准确。

**可选进阶**：加上 `--concurrency` 参数用 `asyncio` 并发调用，对比串行/并发的总耗时（为后续高并发打基础）。
