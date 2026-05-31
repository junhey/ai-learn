---
stage: 7
title: "推理优化 · 部署"
level: "部署"
duration: "1.5 周"
goal: "理解 LLM 推理为何慢且贵，掌握 KV Cache、批处理、量化、投机解码等核心优化原理，并能用 vLLM/Ollama/llama.cpp 把开源模型部署为生产可用的 OpenAI 兼容服务"
prereq: "完成 stage-01~06，了解 Transformer 基本结构、有 Linux 命令行与 Docker 基础，最好有一张可用的 NVIDIA GPU"
stack: ["vLLM", "Ollama", "llama.cpp", "OpenAI SDK", "CUDA"]
---

## 第 1 节：推理为什么慢、为什么贵——KV Cache 与显存墙

### 📖 知识点

要优化推理，先得知道时间和钱花在哪。LLM 的自回归生成分两个阶段，性能特征截然不同：

- **Prefill（预填充）阶段**：把整个 prompt 一次性喂进模型，并行计算所有输入 token 的表示。这一步是**计算密集型（compute-bound）**，GPU 算力跑满，处理几百个 token 也很快。
- **Decode（解码）阶段**：逐个生成输出 token，每生成一个都要把整个模型权重从显存搬到计算单元跑一遍。这一步是**访存密集型（memory-bound）**——瓶颈不在算力，而在**把几十 GB 的权重反复从显存读出来的带宽**。这就是为什么「输出 100 个 token」远比「输入 1000 个 token」慢：decode 是逐 token 串行的，且每步都被显存带宽卡住。

理解这个「memory-bound」的本质，是所有推理优化的钥匙：**几乎所有优化手段，要么是减少要搬运的数据量（量化），要么是让每次搬运服务更多请求（批处理），要么是减少搬运次数（投机解码）。**

而连接这两个阶段的关键机制是 **KV Cache（键值缓存）**。Transformer 的 attention 在生成每个新 token 时，都需要用到前面所有 token 的 Key 和 Value 向量。如果不缓存，每生成一个 token 就要把前面所有 token 重算一遍 K/V，复杂度爆炸。KV Cache 把已经算过的 K/V 存下来，新 token 只算自己的、复用历史的——**用显存换计算**，这是 LLM 能实用化的根基。

但 KV Cache 本身就是显存吃货。它的大小约等于：`2（K和V）× 层数 × 序列长度 × (KV头数 × head_dim) × 精度字节数 × batch`。注意这里用的是 **KV 头数 × head_dim** 而非完整的隐藏维度——现代模型几乎都用 **GQA（分组查询注意力）**，KV 头数远少于 Q 头数（如 Llama-3-8B 是 32 个 Q 头但只有 8 个 KV 头），KV Cache 因此比按 hidden_dim 估算的小好几倍。一个 7B/8B 模型、上下文几千 token、并发几十路，KV Cache 仍能吃掉好几 GB——而且它随**序列长度和并发数线性增长**。这就引出推理部署最现实的约束：**显存墙**。你的显存要同时装下「模型权重 + KV Cache + 激活值」，三者任一超标都会 OOM。

> **专家心法**：估算 KV Cache 时，务必去查模型 config 里的 `num_key_value_heads`，而不是想当然用 hidden_dim。GQA 模型的 KV Cache 比按 hidden_dim（即 MHA 假设）估算的小好几倍，搞错会把显存需求高估 4~7 倍、做出错误的选卡决策。还有一个反直觉但极关键的点：**KV Cache 恒为 fp16，不随权重量化而缩小**——你把权重量化成 INT4，省的全是权重那部分，KV Cache 一字节不少。这正是「INT4 都量化了显存怎么还吃紧」的常见困惑根源。

> **专家心法**：调试推理性能时，第一个要问的问题永远是「我是 compute-bound 还是 memory-bound？」。批量小、追求低延迟时，你卡在访存；批量大、追求高吞吐时，你才可能卡在算力。搞错了方向，优化全是无用功——给 memory-bound 的场景堆算力，一分钱效果都没有。

### 💻 代码示例

用一段显存估算脚本把「显存墙」量化出来——这是部署前必做的容量规划，决定你需要几张什么卡。

```python
# GPU 显存估算：部署一个模型到底需要多少显存？
def estimate_vram(params_billion: float, bytes_per_param: float,
                  context_len: int, batch: int,
                  num_layers: int, hidden_dim: int,
                  num_kv_heads: int, head_dim: int) -> dict:
    """
    params_billion : 参数量（B，十亿）
    bytes_per_param: 每参数字节数  fp16=2, int8=1, int4=0.5
    num_kv_heads   : KV 头数（GQA 模型远少于 Q 头数，查 config 的 num_key_value_heads）
    head_dim       : 每个注意力头的维度（= hidden_dim / Q头数）
    """
    # 1) 权重显存 = 参数量 × 每参数字节数
    weights_gb = params_billion * 1e9 * bytes_per_param / 1e9

    # 2) KV Cache = 2(K,V) × 层数 × 序列长 × (KV头数 × head_dim) × 精度 × batch
    #    关键：GQA 下 KV 维度是 num_kv_heads*head_dim，而非完整 hidden_dim；
    #    用 hidden_dim 估算会把 KV Cache 高估好几倍（如 Llama-3-8B 高估 4 倍）。
    #    另注：KV Cache 恒为 fp16（2 字节/元素），不随权重量化而缩小。
    kv_bytes = 2 * num_layers * context_len * (num_kv_heads * head_dim) * 2 * batch
    kv_gb = kv_bytes / 1e9

    # 3) 经验：激活值 + 框架开销，按权重的 ~20% 粗估
    overhead_gb = weights_gb * 0.2

    total = weights_gb + kv_gb + overhead_gb
    return {
        "weights_GB": round(weights_gb, 2),
        "kv_cache_GB": round(kv_gb, 2),
        "overhead_GB": round(overhead_gb, 2),
        "total_GB": round(total, 2),
    }

# 示例：Llama-3-8B（32层，hidden=4096，32个Q头，8个KV头(GQA)，head_dim=128）
#       上下文 4096，并发 16 路
print("FP16:", estimate_vram(8, 2, 4096, 16, 32, 4096, num_kv_heads=8, head_dim=128))
# -> 权重 16GB + KV Cache 约 8.6GB + 开销约 3.2GB ≈ total 27.8GB
#    超过 24GB（如 4090）一截，单张 24GB 卡装不下，需 A100-40G 或更大

print("INT4:", estimate_vram(8, 0.5, 4096, 16, 32, 4096, num_kv_heads=8, head_dim=128))
# -> 权重降到 4GB，但 KV Cache 仍是 8.6GB（fp16，不随权重量化而缩小）！
#    total ≈ 13.4GB，这下能进 24GB 卡 —— 注意省下的全是权重，KV 一点没省

# 经验法则速记：
# fp16 权重显存(GB) ≈ 参数量(B) × 2
# int8 权重显存(GB) ≈ 参数量(B) × 1
# int4 权重显存(GB) ≈ 参数量(B) × 0.5
# 关键洞察：量化只压缩权重，KV Cache 恒为 fp16、不随量化缩小——
# 这正是 INT4 量化后显存仍可能吃紧的根因（高并发/长上下文时 KV 可能比权重还大）。
```

跑一遍你就明白：为什么同样一个 70B 模型，有人单卡跑得动、有人要 4 张卡——差别全在精度和并发带来的显存账。

### 📚 推荐资料

- [Transformer Inference Arithmetic (kipply)](https://kipp.ly/transformer-inference-arithmetic/) — 把推理的算力/访存/显存算到底的经典长文。
- [Hugging Face: LLM Inference Optimization](https://huggingface.co/docs/transformers/main/en/llm_optims) — 官方推理优化总览，含 KV Cache、量化、批处理。
- [Efficient Memory Management for LLM Serving (PagedAttention/vLLM 论文)](https://arxiv.org/abs/2309.06180) — KV Cache 显存管理的开创性工作。
- [The KV Cache: Memory Usage in Transformers](https://medium.com/@plienhar/llm-inference-series-3-kv-caching-unveiled-048152e461c8) — 图解 KV Cache 与显存占用。

### 🛠 动手实践

用上面的脚本，估算「Llama-3-8B」「Qwen2.5-14B」「Llama-3-70B」三个模型在 fp16/int8/int4 三种精度、上下文 8192、并发 32 下的总显存。验收标准：做一张表，标注每种组合「单张 24GB / 单张 80GB / 需多卡」分别能否装下，并指出哪些场景 KV Cache 已超过权重显存。

## 第 2 节：吞吐的引擎——批处理与连续批处理（Continuous Batching）

### 📖 知识点

第 1 节说过 decode 阶段是 memory-bound：把整个模型权重搬一次，只为生成 1 个 token，太浪费了。**批处理（Batching）**的核心洞察是：既然权重已经搬到计算单元了，何不让它同时为多个请求各算一个 token？把 N 个请求打包，权重只搬一次就服务 N 个请求，**显存带宽的利用率提升 N 倍**——这就是为什么提升吞吐（throughput）几乎总是靠增大 batch。

但传统的**静态批处理（Static Batching）**有致命缺陷：它必须等齐一个 batch 的所有请求，且要等 batch 里**最慢（生成最长）的那个**请求结束，整批才能返回、才能接新请求。结果是：短请求被长请求拖着干等，GPU 大量空转。在请求长度参差不齐的真实流量下，静态批处理的 GPU 利用率惨不忍睹。

**连续批处理（Continuous Batching，又叫 in-flight batching）**是 vLLM 等现代推理引擎的杀手锏。它把调度粒度从「整个请求」细化到「每个生成步（iteration）」：在每一步解码后，**已经生成完的请求立即离开 batch 并返回结果，腾出的位置马上塞进排队中的新请求**。batch 像一条永不停歇的流水线，槽位一空就补，GPU 几乎不空转。在真实负载下，连续批处理相比静态批处理能带来**数倍到十几倍的吞吐提升**，这是 vLLM 性能碾压的主因之一。

与之配套的是 **PagedAttention**：传统实现给每个请求预留一整块连续显存放 KV Cache，但你事先不知道请求会生成多长，只能按最大长度预留，导致大量显存碎片和浪费。PagedAttention 借鉴操作系统虚拟内存分页的思想，把 KV Cache 切成固定大小的「块（block）」，按需分配、非连续存储，显存浪费从 60%~80% 降到 4% 以下。**省下的显存可以装更大的 batch，于是吞吐又进一步提升**——PagedAttention 和连续批处理相辅相成。

> **专家心法**：吞吐（throughput, tokens/s 总量）和延迟（latency, 单请求响应时间）是一对需要权衡的指标。增大 batch 提升吞吐，但单个请求可能等更久。面向 C 端聊天（重延迟）和面向离线批量处理（重吞吐）的部署配置完全不同。**先想清楚你优化的是哪个，再去调 batch、调并发**——这是部署决策的第一性问题。

### 💻 代码示例

用 vLLM 的离线批量接口，直观感受「一次性喂一批 prompt」相比「逐条调用」的吞吐差异。vLLM 内部自动做连续批处理和 PagedAttention，你无需手写调度。

```python
# pip install vllm
# 注意：需要 NVIDIA GPU + CUDA 环境
from vllm import LLM, SamplingParams
import time

# 加载模型（vLLM 会自动管理 KV Cache、PagedAttention）
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="float16",
    gpu_memory_utilization=0.9,   # 允许 vLLM 用到 90% 显存放权重+KV Cache
    max_model_len=4096,
)

sampling = SamplingParams(temperature=0, max_tokens=128)

# 一次性提交一批 prompt —— vLLM 自动做连续批处理
prompts = [f"用一句话解释什么是 {topic}。"
           for topic in ["RAG", "KV Cache", "量化", "投机解码", "连续批处理"] * 20]  # 100 条

t0 = time.time()
outputs = llm.generate(prompts, sampling)   # 批量推理，引擎自动调度
dt = time.time() - t0

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f"处理 {len(prompts)} 条，生成 {total_tokens} tokens，耗时 {dt:.1f}s")
print(f"吞吐: {total_tokens/dt:.1f} tokens/s")   # 远高于逐条 for 循环调用
print("示例输出:", outputs[0].outputs[0].text)
```

把 100 条 prompt 一次性交给 vLLM，和你写 `for` 循环逐条调用相比，吞吐能差一个数量级——这就是连续批处理 + PagedAttention 在背后默默替你榨干 GPU。

### 📚 推荐资料

- [How Continuous Batching Enables 23x Throughput (Anyscale)](https://www.anyscale.com/blog/continuous-batching-llm-inference) — 用图和数据讲透连续批处理，必读。
- [vLLM 官方文档](https://docs.vllm.ai/en/latest/) — PagedAttention、连续批处理的权威实现与配置。
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180) — vLLM 的理论基石，理解 KV Cache 分页管理。
- [NVIDIA: Mastering LLM Inference (In-flight Batching)](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) — TensorRT-LLM 视角的批处理优化。

### 🛠 动手实践

用 vLLM 跑两组对比：(1) 把 100 条 prompt 一次性 `llm.generate(prompts)`；(2) 写 `for` 循环逐条 `llm.generate([p])`。记录两者总耗时和 tokens/s。验收标准：用数据说明连续批处理带来的吞吐提升倍数，并解释为什么逐条调用浪费了 GPU。

## 第 3 节：用更少的位换更大的模型——量化（GGUF / AWQ / GPTQ）

### 📖 知识点

模型权重默认用 fp16/bf16（每参数 2 字节）存储。**量化（Quantization）**就是用更少的比特（int8、int4 甚至更低）来表示权重，**直接成倍降低显存占用，并因为「要搬运的数据变少」而加速 memory-bound 的 decode**。一个 fp16 下要 16GB 的 7B 模型，int4 量化后只需约 4GB——让消费级显卡甚至笔记本跑大模型成为可能。代价是精度损失，但现代量化方法已能把损失控制得很小。

关键是搞清楚三种主流量化格式的定位差异，这是面试和选型高频考点：

- **GPTQ**：一种**训练后量化（PTQ）**算法，用少量校准数据，逐层最小化量化误差。主打 **GPU 推理**，常见 int4/int3，在 vLLM、AutoGPTQ 等 GPU 推理栈里广泛支持。特点是「为 GPU 优化、需要校准数据」。
- **AWQ（Activation-aware Weight Quantization）**：同样是 GPU 向的 PTQ，但核心洞察是「**不是所有权重同等重要**」——它根据激活值分布识别出少数「显著权重」并加以保护，从而在同样 int4 下通常比 GPTQ **精度更好、推理更快**。是目前 GPU 部署 int4 的优选之一，vLLM 原生支持。
- **GGUF**：注意它**不是量化算法，而是 llama.cpp 生态的文件格式**（GPT-Generated Unified Format，前身是 GGML）。它把模型权重和元数据打包，支持多种量化级别（如 `Q4_K_M`、`Q5_K_M`、`Q8_0`），**专为 CPU 和混合 CPU/GPU 推理设计**。Ollama、llama.cpp、LM Studio 都用 GGUF。特点是「跨平台、CPU 友好、能 CPU/GPU 分层卸载」。

一句话记忆选型：**要在 NVIDIA GPU 上高吞吐部署，用 AWQ/GPTQ（配 vLLM）；要在 Mac/CPU/本地开发或显存不够想 CPU 分担，用 GGUF（配 Ollama/llama.cpp）**。

> **专家心法**：量化不是「越激进越好」。int4 对多数任务够用，但在数学推理、代码、长链路 agent 这类对精度敏感的任务上，量化误差会被逐步放大、暴露得很明显。上线任何量化版本前，**务必用 stage-06 的评估流水线在你的真实任务上对比 fp16 vs 量化版的指标**——别让省下的显存以悄悄掉点的精度为代价。先评估，再省钱。

### 💻 代码示例

vLLM 直接加载 AWQ 量化模型，部署成本立刻减半。Hugging Face 上有大量社区量化好的现成权重。

```python
# pip install vllm
from vllm import LLM, SamplingParams

# 直接加载社区量化好的 AWQ int4 权重（无需自己量化）
# Hugging Face 上搜 "AWQ" 即可找到大量现成模型
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",   # AWQ int4 量化版
    quantization="awq",                       # 显式指定量化方式
    dtype="float16",
    gpu_memory_utilization=0.9,
)
# 同一张卡，int4 版相比 fp16 版：权重显存约降到 1/4，
# 省出的显存可装更大 batch / 更长上下文 / 更多并发

out = llm.generate(["解释什么是 AWQ 量化。"],
                   SamplingParams(temperature=0, max_tokens=100))
print(out[0].outputs[0].text)
```

如果想在本地（含 Mac）用 GGUF，则走 llama.cpp 路线，可指定有多少层卸载到 GPU：

```bash
# llama.cpp：下载 GGUF 格式量化模型并本地推理
# 安装（macOS 示例）：brew install llama.cpp

# 跑一个 Q4_K_M（int4，K-quant，质量/体积平衡最常用的档位）
llama-cli \
  -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M \
  -p "用一句话解释 GGUF 是什么。" \
  -n 100 \
  -ngl 99          # n-gpu-layers：把 99 层卸载到 GPU（Mac 上即 Metal），CPU 跑不动的交给 GPU

# 起一个 OpenAI 兼容的本地 server（llama.cpp 自带）
llama-server \
  -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M \
  -ngl 99 \
  --port 8080      # 之后可用 OpenAI SDK 指向 http://localhost:8080/v1
```

### 📚 推荐资料

- [AWQ 论文](https://arxiv.org/abs/2306.00978) — Activation-aware 量化的核心思想，理解为什么它精度更好。
- [GPTQ 论文](https://arxiv.org/abs/2210.17323) — 训练后量化的开创性方法。
- [llama.cpp GGUF 量化类型说明](https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/README.md) — Q4_K_M 等档位的含义与取舍。
- [Hugging Face Quantization 概览](https://huggingface.co/docs/transformers/main/en/quantization/overview) — 各量化方案的对比与使用。

### 🛠 动手实践

选一个 7B 模型，分别拿到它的 fp16 版和 AWQ（或 GGUF Q4_K_M）量化版。用 stage-06 的评估流水线，在你的真实任务数据集上跑两版，对比：(1) 显存占用；(2) 吞吐；(3) faithfulness/answer relevance 等质量指标。验收标准：用一张表呈现「省了多少显存、快了多少、掉了多少分」，并给出「这个任务该不该上量化版」的明确结论。

## 第 4 节：再榨一层延迟——投机解码（Speculative Decoding）

### 📖 知识点

回到 decode 阶段 memory-bound 的本质：每生成 1 个 token，都要把整个大模型权重搬一遍。**投机解码（Speculative Decoding）**的精妙构想是：能不能让一次「搬运大模型」生成不止 1 个 token？

它的做法是引入一个**小而快的「草稿模型（draft model）」**（参数量远小于主模型，或用主模型自身的轻量头）：

1. **草稿（Draft）**：让小模型快速、连续地「猜」出接下来的 k 个 token（比如一口气猜 4 个）。小模型快，这一步成本很低。
2. **验证（Verify）**：把这 k 个候选 token 一次性交给大模型**并行验证**——关键在于，**并行验证 k 个 token 和生成 1 个 token 的大模型访存成本几乎一样**（权重就搬一次），却能一次确认多个 token 是否符合大模型的真实分布。
3. **接受/回退**：从前往后看大模型认可了几个。被认可的 token 直接采纳（白赚的加速）；遇到第一个不认可的就停下，用大模型自己的输出纠正，再开始下一轮。

它的「免费午餐」在于：**输出分布在数学上严格等价于大模型单独解码**——投机解码不改变生成质量，只改变速度。当草稿模型猜得准（在可预测的、模板化的文本上命中率高），多个 token 被一次性确认，端到端延迟可降低 2~3 倍；猜不准时退化为正常解码，不会更差（只是白跑了草稿）。

vLLM、TensorRT-LLM 等都内置了投机解码支持，还有不需要单独草稿模型的变体（如 n-gram、Medusa、EAGLE），降低了部署复杂度。

> **专家心法**：投机解码主要优化的是**单请求延迟**，而非总吞吐——在高并发、batch 已经很大、GPU 算力快被打满的场景，它的收益会缩水甚至变负（验证 k 个 token 的额外算力开销吃掉了好处）。所以它最适合**低并发、追求快速响应**的场景（如交互式 coding 助手）。又是那句话：先明确你优化吞吐还是延迟，再决定要不要上投机解码。

### 💻 代码示例

vLLM 开启投机解码只需一个配置——用一个小草稿模型加速大主模型。

```python
# pip install vllm
from vllm import LLM, SamplingParams

# 主模型大、草稿模型小，二者最好同系列（词表/分布更接近，命中率更高）
llm = LLM(
    model="Qwen/Qwen2.5-14B-Instruct",       # 主模型（大、准、慢）
    speculative_config={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",  # 草稿模型（小、快）
        "num_speculative_tokens": 4,             # 每轮让草稿模型一口气猜 4 个 token
    },
    dtype="float16",
    gpu_memory_utilization=0.9,
)

# 用法和普通推理完全一样 —— 输出质量不变，只是更快
out = llm.generate(
    ["写一个 Python 函数，判断字符串是否为回文。"],
    SamplingParams(temperature=0, max_tokens=200),
)
print(out[0].outputs[0].text)
# 在低并发、可预测文本（如代码）上，端到端延迟通常能降 2~3 倍；
# 注意：高并发打满 GPU 时收益会下降，需实测权衡
```

是否真的更快，**必须在你的真实负载下实测**——不同任务、不同并发，投机解码的收益差异很大。

### 📚 推荐资料

- [Speculative Decoding 原始论文 (Leviathan et al.)](https://arxiv.org/abs/2211.17192) — 投机解码的理论基础与等价性证明。
- [vLLM Speculative Decoding 文档](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 配置草稿模型、n-gram、EAGLE 等多种方案。
- [Medusa: Simple Framework for Inference Acceleration](https://arxiv.org/abs/2401.10774) — 无需独立草稿模型的多头投机解码。
- [Hugging Face: Assisted Generation](https://huggingface.co/blog/assisted-generation) — transformers 里投机解码（辅助生成）的入门讲解。

### 🛠 动手实践

用 vLLM 对同一个大模型跑两组：(1) 不开投机解码；(2) 配一个小草稿模型开启投机解码。分别在「低并发（1~2 路）」和「高并发（≥16 路）」下，测单请求平均延迟。验收标准：用数据展示投机解码在低并发下加速明显、高并发下收益缩水，并解释这与「算力是否被打满」的关系。

## 第 5 节：把模型变成服务——vLLM / Ollama / llama.cpp 部署实战

### 📖 知识点

优化原理学完，最后一步是**把模型变成一个稳定、可被应用调用的服务**。这一节落到工程落地，核心是理解三个工具的定位，按场景选型，而不是迷信某一个：

- **vLLM —— 生产级高吞吐推理服务器**。这是把开源模型部署到生产的首选。它内置 PagedAttention、连续批处理、支持张量并行（多卡）、量化、投机解码，且能一键起一个 **OpenAI 兼容 API**。「OpenAI 兼容」是它的关键价值：你之前所有用 OpenAI SDK 写的应用代码，**只要把 `base_url` 指向你的 vLLM 服务、`api_key` 随便填，一行不改就能切到自部署的开源模型**——这让「从调 API 到自托管」的迁移成本几乎为零，也是避免供应商锁定的关键。
- **Ollama —— 本地开发与原型的最佳体验**。一行 `ollama run qwen2.5` 就能在你的 Mac/PC 上把模型跑起来，自动处理 GGUF 下载、量化选择、显存/CPU 调度。它也提供 OpenAI 兼容端点。适合本地开发、离线 demo、隐私敏感场景，**不适合高并发生产**（吞吐远不如 vLLM）。
- **llama.cpp —— 极致轻量与跨平台**。纯 C/C++ 实现，无重依赖，能在 CPU、Mac Metal、树莓派、甚至手机上跑 GGUF 模型。Ollama 的底层就是它。当你要在没有 NVIDIA GPU 的环境、或资源极度受限的边缘设备上部署时，它是终极方案。

选型心法：**生产高并发 → vLLM；本地开发/隐私/快速试 → Ollama；无 GPU/边缘/嵌入 → llama.cpp**。三者都能暴露 OpenAI 兼容接口，所以你的上层应用代码可以保持统一。

> **专家心法**：「OpenAI 兼容 API」是这个时代最被低估的工程杠杆。把它当成一种**抽象层**：上层应用永远只认 OpenAI 接口协议，下层后端（OpenAI 官方 / vLLM 自托管 / Ollama 本地）可以随意替换。开发时用 Ollama 省钱省事，压测和生产切 vLLM，必要时回退云端 API——**同一份应用代码，三套后端自由切换**。架构上预留这一层，你会在成本和灵活性上受益无穷。

### 💻 代码示例

第一步：用 vLLM 一行命令起一个 OpenAI 兼容服务器。

```bash
# pip install vllm
# 启动 OpenAI 兼容 API 服务器（默认监听 8000 端口）
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --api-key my-secret-key \
  --port 8000

# 多卡张量并行（如 2 张卡跑一个大模型）：加 --tensor-parallel-size 2
# 部署量化版省显存：--quantization awq 并把 model 换成 AWQ 权重
```

第二步：用你早已熟悉的 OpenAI SDK 调用它——**代码和调 OpenAI 官方几乎一模一样，只改 `base_url`**。

```python
# pip install openai
from openai import OpenAI

# 关键：base_url 指向你的 vLLM 服务，api_key 用你启动时设的值
client = OpenAI(
    base_url="http://localhost:8000/v1",   # 指向自部署服务
    api_key="my-secret-key",
)

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",       # 用启动时的模型名
    messages=[
        {"role": "system", "content": "你是简洁的技术助手。"},
        {"role": "user", "content": "vLLM 的连续批处理解决了什么问题？"},
    ],
    temperature=0,
)
print(resp.choices[0].message.content)
# 流式输出也完全兼容（stream=True），上层应用代码无需任何改动
```

第三步：本地开发用 Ollama，同样是 OpenAI 兼容，丝滑切换。

```bash
# 安装后（https://ollama.com），拉起模型一行搞定
ollama run qwen2.5:7b        # 自动下载 GGUF 量化权重并进入交互
# Ollama 后台默认在 11434 端口提供 OpenAI 兼容 API
```

```python
from openai import OpenAI

# 同样的 OpenAI SDK，只把 base_url 换成 Ollama 的端点
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",          # Ollama 不校验 key，填任意非空值即可
)
resp = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "用一句话介绍 Ollama。"}],
)
print(resp.choices[0].message.content)
# 看到没——上层代码几乎没变，只切了 base_url 和 model 名。
# 这就是「OpenAI 兼容」作为抽象层的威力：后端随便换，应用不动。
```

### 📚 推荐资料

- [vLLM OpenAI Compatible Server 文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) — `vllm serve` 全部参数与用法的权威文档。
- [Ollama 官方文档](https://github.com/ollama/ollama/blob/main/docs/README.md) — 本地模型管理、OpenAI 兼容端点、Modelfile 自定义。
- [llama.cpp 官方仓库](https://github.com/ggml-org/llama.cpp) — 构建、server、量化、各平台部署说明。
- [vLLM Distributed Inference (多卡部署)](https://docs.vllm.ai/en/latest/serving/distributed_serving.html) — 张量并行与多卡/多机部署。

### 🛠 动手实践

本地装 Ollama，跑起一个 7B 模型，用 OpenAI SDK 通过 `base_url` 调通它。然后把 stage-01~05 里你写过的**任意一个**应用（如信息抽取 CLI、RAG 问答），**只改 `base_url` 和 `model` 两处**，让它跑在本地开源模型上。验收标准：同一份应用代码，能在「OpenAI 官方」和「本地 Ollama」之间通过改两行配置自由切换并都能正常工作。

## 🎯 阶段实战项目：用 vLLM 部署开源模型提供 OpenAI 兼容 API 服务

**背景**：公司出于成本与数据合规考虑，要把一部分线上 LLM 流量从云端 API 迁移到自托管的开源模型。你负责搭建这个自托管推理服务，要求**对上层应用零侵入**（继续用 OpenAI 协议），并做好容量规划与性能验证。

**要求**：

1. **容量规划**：选定一个开源模型（如 Qwen2.5-7B/14B-Instruct），用第 1 节的显存估算脚本，规划在你的硬件（注明 GPU 型号/显存）上，fp16 与量化版分别支持的最大上下文与并发，给出选型结论。
2. **部署服务**：用 `vllm serve` 起一个 OpenAI 兼容服务，配置 `--api-key`、`--max-model-len`、`--gpu-memory-utilization`；若显存紧张，部署 AWQ 量化版。
3. **应用零改造接入**：拿 stage-02/03 的 RAG 应用或 stage-01 的抽取 CLI，**仅改 `base_url`/`model`**，证明它能无缝切到你的自托管服务。
4. **性能压测**：写一个并发压测脚本（`asyncio` 或 `locust`），测不同并发（1/8/32/64）下的吞吐（tokens/s）和 P50/P95 延迟，画出「并发 vs 吞吐/延迟」曲线。
5. **优化对照实验**：至少做一组对照——量化（fp16 vs AWQ）或投机解码（开/关），用数据量化收益与代价。
6. **质量守门**：用 stage-06 的评估流水线，确认自托管/量化版在真实任务上的质量没有不可接受的回归。

**技术要点**：显存估算、`vllm serve`、OpenAI 兼容 API、连续批处理、量化/投机解码配置、并发压测、吞吐/延迟权衡、跨后端无缝切换。

**验收标准**：
- 服务稳定运行，OpenAI SDK 能正常（含流式）调用。
- 上层应用**一行业务逻辑都不改**，仅改连接配置即可切到自托管模型。
- 有一份压测报告：含「并发-吞吐-延迟」曲线、量化/投机解码对照数据、质量回归结论，并给出「这套部署在什么 QPS 下需要扩容/加卡」的明确建议。

**可选进阶**：(1) 用 Docker Compose 把 vLLM 服务容器化，加一层 Nginx/LiteLLM 做网关与多模型路由；(2) 接入 Prometheus + Grafana 监控 GPU 利用率、KV Cache 占用、吞吐与延迟，做出一块生产监控大盘。
