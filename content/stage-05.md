---
stage: 5
title: "模型微调 · Fine-tuning"
level: "高级"
duration: "2 周"
goal: "搞清楚「何时该微调、何时不该」，吃透 LoRA/QLoRA 的省显存原理，能用 PEFT + TRL 在消费级显卡上完成一次 SFT 并合并部署"
prereq: "完成 stage-01~04（Prompt、RAG、Agent），理解 Transformer 基本结构，有一张 ≥16GB 显存的 GPU（或用 Colab/云）"
stack: ["Python", "PyTorch", "HuggingFace Transformers", "PEFT", "TRL", "bitsandbytes", "Unsloth"]
---

## 第 1 节：何时该微调——微调不是第一选择，而是最后手段

### 📖 知识点

新手最常犯的错，是把微调当成「让模型变好」的万能钥匙，一上来就想训模型。**资深工程师的第一反应恰恰相反：先穷尽 Prompt 和 RAG，确认它们不够用了，才考虑微调。** 因为微调成本高（算力、数据、时间、维护），且会引入新问题（灾难性遗忘、过拟合、部署复杂度）。

先厘清三者各自解决什么问题，这是本阶段最重要的认知：

- **Prompt 工程**：改变模型的**行为/输出格式**。零成本、秒级迭代。能用 Prompt 解决的，永远优先。
- **RAG（检索增强）**：给模型补充**它不知道的、动态变化的知识**（私有文档、实时数据）。知识是「外挂」进上下文的，模型参数不变。**「让模型知道新事实」是 RAG 的活，不是微调的活**——这是最高频的概念混淆。
- **Fine-tuning（微调）**：改变模型的**内在能力、风格、领域适应性**。它教的是「怎么做」（一种技能、一种语气、一种格式遵循），而不是「记住什么事实」。

**什么时候微调才是对的选择？**

1. **稳定的风格/格式/语气**：要模型始终用某种特定口吻、严格遵循某种私有输出格式，Prompt 里塞一堆示例又长又不稳——微调能把它「内化」。
2. **特定领域的能力提升**：法律、医疗、代码等专业领域，需要模型掌握领域术语和推理范式。
3. **用小模型替代大模型降本**：把 GPT-4o 的能力「蒸馏」到一个微调过的小模型上，推理成本可能降一两个数量级。这是工业界微调最常见的动机。
4. **Prompt 太长导致成本/延迟不可接受**：当你需要 10 个 few-shot 示例才能稳定，不如微调把这些「编译」进权重，换来短 Prompt。
5. **降低延迟、可私有化部署**：微调小模型可本地部署，数据不出域。

**什么时候不该微调：** 需要补充事实知识（用 RAG）、需求频繁变化（微调迭代慢）、数据量太少（几十条不如写好 Prompt）、只是输出格式问题（用 Structured Outputs）。

> **专家心法**：记住这个决策顺序——**Prompt → Few-shot → RAG → Fine-tuning**，从左到右成本递增、迭代变慢。把微调放在最后。还有一句反直觉的真相：**微调改变的是「形」（风格、格式、技能），RAG 改变的是「实」（知识、事实）**。想让模型「说话像客服」用微调，想让模型「知道今天的退货政策」用 RAG。两者常常组合使用：微调出风格 + RAG 注入知识。

### 💻 代码示例

这一节是决策框架，给一段「自检脚本」帮你在动手训练前理清思路（而非具体训练代码）。

```python
# 一个微调决策自检清单（伪决策树，跑起来逼自己回答关键问题）
def should_finetune() -> str:
    def ask(q): return input(f"{q} (y/n): ").strip().lower() == "y"

    if ask("你的核心需求是『让模型知道新的事实/私有知识』吗"):
        return "❌ 用 RAG，不是微调。微调不擅长记事实，且知识一变就得重训。"

    if ask("你已经认真试过 Prompt 工程 + Few-shot 了吗"):
        if not ask("即便如此，输出风格/格式/质量仍不稳定吗"):
            return "❌ 先把 Prompt 调好。微调成本远高于改 Prompt。"
    else:
        return "❌ 先做 Prompt 工程。这是免费且最快的手段。"

    if not ask("你有 ≥ 几百条高质量、风格一致的标注样本吗"):
        return "⚠️ 数据不足。先攒数据，或用更大模型生成蒸馏数据。"

    if ask("需求是否稳定（未来几周不会大改）"):
        return "✅ 适合微调：用 LoRA/QLoRA 做 SFT，成本可控。"
    return "⚠️ 需求多变时微调迭代慢，谨慎评估。"

if __name__ == "__main__":
    print(should_finetune())
```

### 📚 推荐资料

- [OpenAI: When to fine-tune](https://platform.openai.com/docs/guides/fine-tuning#when-to-use-fine-tuning) — 官方对「微调 vs Prompt」的判断标准。
- [Anyscale: Fine-tuning vs RAG](https://www.anyscale.com/blog/fine-tuning-and-retrieval-augmented-generation-rag-the-right-tool-for-the-job) — 把两者的适用场景讲得很透。
- [Hugging Face: Fine-tuning 概述](https://huggingface.co/docs/transformers/training) — 微调的整体认知与生态入口。
- [a16z: Emerging Architectures for LLM Applications](https://a16z.com/emerging-architectures-for-llm-applications/) — 从架构视角看 RAG 与微调在系统中的位置。

### 🛠 动手实践

针对你自己的一个真实需求（或设想），写一份不超过一页的「微调必要性论证」：明确说明为什么 Prompt/RAG 不够、微调要解决什么、需要什么数据、预期收益与风险。验收标准：能用上面的决策顺序自圆其说，**并且至少列出一个「换成 RAG 可能更好」的反方观点**。

## 第 2 节：SFT 监督微调——微调的主力范式

### 📖 知识点

**SFT（Supervised Fine-Tuning，监督微调）** 是当今微调的主力。它的思路朴素得和预训练一脉相承：**给模型看大量「输入→期望输出」的配对样本，让它继续做「预测下一个 token」的训练，只不过现在的训练数据是你精心准备的、符合你需求的高质量样本。** 模型通过梯度下降微调权重，逐渐学会在你的场景下「该怎么回答」。

**和预训练的区别：** 预训练是在无标注的海量文本上学「语言和世界的统计规律」；SFT 是在有标注的「指令-回答」对上学「如何遵循指令、按期望方式回应」。所以 SFT 也常叫 **Instruction Tuning（指令微调）**。

**关键概念：**

- **数据格式即一切**：SFT 的数据是「对话/指令」格式。现代做法是用模型的 **chat template**（聊天模板）把 `messages`（system/user/assistant）渲染成模型训练时见过的特殊标记格式。**用错模板 = 训练信号错位**，是新手最容易翻车的地方。
- **Loss masking（损失掩码）**：训练时通常**只对 assistant 的回答部分计算 loss**，user 的提问部分不算（你要模型学「怎么答」，不是学「怎么问」）。TRL 的 `SFTTrainer` 可自动处理。
- **过拟合与 epoch**：SFT 数据量通常不大，epoch 设太多（如 >3）极易过拟合——模型死记硬背训练样本，泛化变差。一般 1~3 个 epoch。
- **学习率**：LoRA 学习率通常高于全参 SFT（约 1e-4 ~ 3e-4），而全参 SFT 约 1e-5 ~ 2e-5。太大易训崩。

**全参数微调 vs 参数高效微调（PEFT）：** 全参数微调（更新所有权重）效果好但极吃显存——微调一个 7B 模型，光优化器状态就要几十 GB，消费级显卡根本扛不住。所以引出了下一节的主角：**LoRA**，只训练极少量参数就能逼近全量微调效果。**本阶段及工业界主流都是 PEFT，全参数微调只在资源充足、追求极致时才用。**

> **专家心法**：**数据质量 >> 数据数量 >> 模型大小 >> 训练技巧**。这个优先级是无数次踩坑换来的。500 条干净、风格统一、覆盖典型场景的高质量样本，胜过 5 万条嘈杂数据。微调失败十有八九是数据问题，不是超参问题。先把精力花在「造好数据」上，而不是反复调学习率。另外，**务必留一个不参与训练的验证集**，否则你根本不知道自己是在学习还是在背书。

### 💻 代码示例

用 HuggingFace `datasets` 构造一个标准 SFT 数据集，并用 chat template 渲染——把「数据格式」这件事看清楚。

```python
# pip install datasets transformers
from datasets import Dataset
from transformers import AutoTokenizer

# 1) SFT 数据的标准形态：每条是一个 messages 列表（多轮对话）
raw = [
    {"messages": [
        {"role": "system", "content": "你是简洁的客服助手。"},
        {"role": "user", "content": "怎么申请退款？"},
        {"role": "assistant", "content": "进入「我的订单」→选择订单→点「申请退款」，3 个工作日内到账。"},
    ]},
    {"messages": [
        {"role": "system", "content": "你是简洁的客服助手。"},
        {"role": "user", "content": "可以改收货地址吗？"},
        {"role": "assistant", "content": "未发货可在订单页直接修改；已发货请联系快递改派。"},
    ]},
]
dataset = Dataset.from_list(raw)

# 2) 用模型自带的 chat_template 渲染——这一步决定训练信号是否对齐
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
example = tok.apply_chat_template(
    dataset[0]["messages"],
    tokenize=False,                 # 先看渲染成什么样的字符串
    add_generation_prompt=False,
)
print("=== 渲染后的训练文本 ===")
print(example)   # 你会看到 <|im_start|>system ... <|im_end|> 这类特殊标记

# 3) 划分训练/验证集（务必留验证集！）
splits = dataset.train_test_split(test_size=0.5, seed=42)
print("train:", len(splits["train"]), " eval:", len(splits["test"]))
```

运行后观察那串带 `<|im_start|>`/`<|im_end|>` 的文本——**这就是模型实际「吃」进去的格式**。换个模型（如 Llama）模板完全不同，用错就训废。

### 📚 推荐资料

- [TRL: Supervised Fine-tuning Trainer](https://huggingface.co/docs/trl/sft_trainer) — `SFTTrainer` 官方文档，SFT 的事实标准工具。
- [HuggingFace: Chat Templates](https://huggingface.co/docs/transformers/chat_templating) — 讲清为什么模板如此关键，必读。
- [Sebastian Raschka: Finetuning LLMs](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) — 对 SFT 流程和取舍的深度长文。
- [Alpaca 数据集](https://github.com/tatsu-lab/stanford_alpaca#data-release) — 经典指令微调数据集，学习指令格式的范本。

### 🛠 动手实践

收集（或自造）20 条「同一风格」的指令-回答样本（如「全部用文言文回答」），分别用 Qwen 和 Llama 的 tokenizer 渲染同一条数据，对比两者 chat template 的差异。验收标准：能指出两个模型特殊标记的不同，并解释「为什么训练时必须用模型自己的模板」。

## 第 3 节：LoRA / QLoRA——为什么能在消费级显卡上微调大模型

### 📖 知识点

这是本阶段技术含量最高、也最该吃透的一节。**LoRA（Low-Rank Adaptation，低秩适配）** 是让微调从「土豪专属」变成「人人可玩」的关键突破。

**先理解痛点：** 全参数微调一个 7B 模型，你要在显存里同时放：fp32 主权重（~28GB）+ 梯度 + Adam 优化器状态（动量 m + 方差 v，fp32 ≈ 56GB），合计通常 100GB+，单张消费级卡（24GB）连零头都放不下。

**LoRA 的核心洞察：** 研究发现，**微调时权重的「变化量」ΔW 其实是低秩的**——也就是说，模型适应新任务所需的有效改变，可以用两个很小的矩阵相乘来近似。于是 LoRA 做了一件聪明事：

- **冻结原始权重 W**（完全不动，不需要存它的梯度和优化器状态）。
- 在旁边并联一个「低秩旁路」：`ΔW = B × A`，其中 `A` 是 `r×d`、`B` 是 `d×r`，`r`（秩）极小（如 8、16、64）。
- 前向时输出变成 `h = Wx + BAx`。**只训练 A 和 B 这两个小矩阵**。

**为什么省显存（Why）：** 假设 `d=4096`，全量要训 `4096×4096≈1678 万`个参数，而 LoRA（r=8）只训 `2×4096×8≈6.5 万`个——**可训练参数减少约 99%+**。梯度和优化器状态只需为这极少量参数维护，显存需求断崖式下降。而且原始权重冻结，天然缓解灾难性遗忘。

**QLoRA = Quantization + LoRA**，更狠的省显存术，三个关键技术：

1. **4-bit NF4 量化**：把冻结的基座模型权重压成 4-bit（NormalFloat4，一种为正态分布权重优化的数据类型）存储，显存再降约 4 倍。计算时临时反量化成 bf16。
2. **Double Quantization（双重量化）**：连量化用的常数也量化，再省一点。
3. **Paged Optimizers**：用 NVIDIA 统一内存处理显存峰值，防 OOM。

**效果：** QLoRA 大幅拉低了微调的显存门槛——**24GB 级显卡可微调约 30B 级模型；65B/70B 需 48GB 级单卡**（QLoRA 原论文的标志性结论是单张 48GB 微调 65B），而性能几乎不输全参数微调。这是 2023 年最有影响力的工程突破之一。

**关键超参：**

- **`r`（秩）**：旁路的「容量」。越大能学的越多，也越占资源、越易过拟合。常用 8~64，多数任务 16 够用。
- **`lora_alpha`**：缩放因子，实际缩放 = `alpha/r`。经验上常设 `alpha = 2r`。
- **`target_modules`**：给哪些层加 LoRA。通常加在注意力的 `q_proj/k_proj/v_proj/o_proj`，要更强可加上 MLP 层。

> **专家心法**：别被 `r` 越大越好的直觉骗了——**对多数任务，r=16 和 r=64 效果差别很小，但后者更易过拟合、更慢**。先用小 `r` 跑通，不够再加。还有个高频疑问：**LoRA 会损失精度吗？** 在绝大多数下游任务上，LoRA/QLoRA 与全参微调的差距小到可忽略；只有在需要大幅改变模型底层能力（如教全新语言）时，PEFT 才可能不如全量。99% 的业务场景，QLoRA 就是性价比之王。

### 💻 代码示例

用 `peft` 给模型加 LoRA、用 `bitsandbytes` 做 4-bit 量化（QLoRA），并打印「可训练参数占比」直观感受省了多少。

```python
# pip install transformers peft bitsandbytes accelerate
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "Qwen/Qwen2.5-0.5B-Instruct"   # 小模型便于本地跑通；逻辑对大模型完全一致

# 1) QLoRA：4-bit NF4 量化配置，把基座权重压到 4-bit 加载
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NormalFloat4，专为正态权重设计
    bnb_4bit_compute_dtype=torch.bfloat16,# 计算时反量化成 bf16
    bnb_4bit_use_double_quant=True,       # 双重量化，再省一点显存
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto",
)
model = prepare_model_for_kbit_training(model)  # 量化训练的必要预处理

# 2) LoRA 配置：只在注意力投影层加低秩旁路
lora_config = LoraConfig(
    r=16,                                  # 秩：旁路容量
    lora_alpha=32,                         # 缩放，约定俗成 = 2*r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 加在注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# 3) 直观看「省了多少」——可训练参数占比
model.print_trainable_parameters()
# 典型输出: trainable params: 0.5M || all params: 494M || trainable%: ~0.1%
# 意味着 99.9% 的参数被冻结，只训那 0.1%
```

`trainable%` 那一行就是 LoRA 的魔法所在——**用 0.1% 的可训练参数，撬动整个模型的领域适应**。

### 📚 推荐资料

- [LoRA 论文](https://arxiv.org/abs/2106.09685) — 低秩适配的原始论文，理解「ΔW 低秩」假设的来源。
- [QLoRA 论文](https://arxiv.org/abs/2305.14314) — NF4、双重量化、Paged Optimizer 的出处，工程细节丰富。
- [HuggingFace PEFT 文档](https://huggingface.co/docs/peft/index) — LoRA/QLoRA 等参数高效方法的官方库与指南。
- [bitsandbytes 文档](https://huggingface.co/docs/bitsandbytes/main/en/index) — 4-bit 量化的底层实现说明。

### 🛠 动手实践

用同一个模型，分别配置 `r=4`、`r=16`、`r=64`，调用 `print_trainable_parameters()` 记录三者的可训练参数量；再分别用 4-bit 量化 vs 不量化加载，用 `nvidia-smi` 记录显存占用。验收标准：用一张表说明「r 对参数量的影响」和「量化对显存的影响」，并解释 QLoRA 为什么能在小显存上跑大模型。

## 第 4 节：完整 SFT + LoRA 训练流水线（PEFT + TRL）

### 📖 知识点

前几节的零件——数据格式、chat template、LoRA、量化——现在用 **TRL（Transformer Reinforcement Learning）库的 `SFTTrainer`** 串成一条完整流水线。TRL 是 HuggingFace 官方的对齐/微调库，`SFTTrainer` 把 SFT 的繁琐细节（模板渲染、loss masking、packing、与 PEFT 集成）都封装好了，是当下做 SFT 的标准工具。

**一条标准 SFT 流水线的步骤：**

1. **加载模型 + tokenizer**（按需 4-bit 量化）。
2. **准备数据集**：整理成 `messages` 格式，`SFTTrainer` 会自动套 chat template。
3. **配置 LoRA**（`LoraConfig`）。
4. **配置训练超参**（`SFTConfig`：学习率、epoch、batch size、梯度累积等）。
5. **训练**：`trainer.train()`，全程监控 train/eval loss。
6. **保存**：只保存 LoRA adapter（几 MB ~ 几十 MB，而非整个模型）。

**几个生产级要点：**

- **梯度累积（gradient accumulation）**：显存装不下大 batch 时，用小 batch 多步累积梯度再更新，等效放大 batch size。
- **梯度检查点（gradient checkpointing）**：用计算换显存，训练大模型几乎必开。
- **监控 eval loss**：train loss 一直降但 eval loss 开始升 = 过拟合，该早停。
- **`bf16`**：现代 GPU 上用 bf16 混合精度，比 fp16 数值更稳。

**Unsloth——值得知道的加速利器：** Unsloth 通过手写优化的 CUDA kernel，让 LoRA/QLoRA 微调**速度提升约 2 倍、显存再省约 50%**，API 与 HuggingFace 高度兼容。资源紧张时强烈推荐。**LLaMA-Factory** 则提供了「零代码 / WebUI / YAML 配置」的微调方案，覆盖 SFT/DPO/全量/LoRA，适合快速实验与不想写训练代码的场景。

> **专家心法**：第一次跑训练，**务必先用极小数据（如 10 条）、1 个 epoch 跑通整条流水线**，确认能正常 train、save、load、推理，再上全量数据。无数人一上来就挂全量数据训几小时，结果发现模板用错/保存路径不对，白烧算力。另外，**保存的是 adapter 不是全模型**——这是 LoRA 的又一好处：一个基座可以挂多个不同任务的 adapter，几十 MB 一个，随时切换。

### 💻 代码示例

一条端到端可运行的 SFT + QLoRA 训练脚本（小模型 + 小数据，本地可跑通）。

```python
# pip install "trl>=0.12" "transformers>=4.46" peft datasets bitsandbytes accelerate
# 钉版本：SFTConfig / eval_strategy 等为较新 TRL/Transformers 命名，旧环境会踩签名错误
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# 1) 4-bit 量化加载（QLoRA）
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2) 构造一个「文言文风格」的小数据集（演示风格微调）
samples = [
    {"messages": [
        {"role": "user", "content": "今天天气怎么样？"},
        {"role": "assistant", "content": "今日天朗气清，惠风和畅，宜出游。"},
    ]},
    {"messages": [
        {"role": "user", "content": "帮我写句鼓励的话。"},
        {"role": "assistant", "content": "千里之行，始于足下；锲而不舍，金石可镂。"},
    ]},
] * 20   # 教学用，重复扩充；真实场景请用真数据
ds = Dataset.from_list(samples).train_test_split(test_size=0.1, seed=42)

# 3) LoRA 配置
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# 4) 训练超参（SFTConfig 继承自 TrainingArguments）
args = SFTConfig(
    output_dir="./wenyan-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,     # 等效 batch=8
    learning_rate=2e-4,                # LoRA 常用量级
    bf16=True,
    logging_steps=5,
    eval_strategy="epoch",             # 每个 epoch 评估，监控过拟合
    save_strategy="epoch",
    max_length=512,
)

# 5) SFTTrainer 自动处理 chat template、loss masking、与 PEFT 集成
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    peft_config=peft_config,
)
trainer.train()

# 6) 只保存 LoRA adapter（很小）
trainer.save_model("./wenyan-lora")
print("✅ 训练完成，adapter 已保存到 ./wenyan-lora")
```

### 📚 推荐资料

- [TRL SFTTrainer 文档](https://huggingface.co/docs/trl/sft_trainer) — 本节代码的权威依据，含所有参数说明。
- [Unsloth 官方仓库](https://github.com/unslothai/unsloth) — 2x 加速、省显存的微调框架，附大量可直接跑的 Colab notebook。
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — 零代码/WebUI 微调框架，支持 SFT/DPO/LoRA/全量，国内社区活跃。
- [HuggingFace PEFT + TRL 教程](https://huggingface.co/docs/trl/en/peft_integration) — PEFT 与 TRL 集成的官方指南。

### 🛠 动手实践

用上面的脚本跑通一次完整训练（建议先在 Colab 免费 T4 上跑 0.5B 模型）。训练前后各让模型回答同一个问题，对比输出风格的变化。验收标准：能展示训练日志（train/eval loss 曲线），并用「微调前 vs 微调后」的输出对比，证明模型确实学到了目标风格。

## 第 5 节：DPO/RLHF 对齐、灾难性遗忘与合并部署

### 📖 知识点

SFT 让模型「会按指令回答」，但要让回答更**符合人类偏好**（更有用、更安全、语气更得体），还需要**对齐（alignment）**。这一节把对齐、遗忘、部署三个收尾问题讲清。

**RLHF 与 DPO——从「会答」到「答得好」：**

- **RLHF（基于人类反馈的强化学习）**：经典三步——SFT → 训练奖励模型（Reward Model，学人类「更喜欢哪个回答」）→ 用 PPO 强化学习优化策略。效果好但**流程复杂、训练不稳、要同时维护 4 个模型**，工程门槛高。
- **DPO（Direct Preference Optimization，直接偏好优化）**：2023 年的重要简化。它**跳过显式的奖励模型和 RL**，直接用「偏好数据对（chosen / rejected）」通过一个巧妙的损失函数优化模型，数学上等价于 RLHF 的目标，但**像普通监督训练一样稳定、简单**。数据格式是 `{prompt, chosen, rejected}`。TRL 提供 `DPOTrainer`，已成为开源社区对齐的首选。
- 典型 pipeline：**先 SFT（学会基本格式与能力）→ 再 DPO（按偏好精修）**。

**灾难性遗忘（Catastrophic Forgetting）——微调的头号副作用：** 模型学新任务时，**可能「忘掉」原有的通用能力**（如微调成客服后，数学/编程能力退化）。成因是梯度更新覆盖了原有知识。**对策：**

1. **用 LoRA**：冻结原权重，天然大幅缓解遗忘（这是 PEFT 被偏爱的又一原因）。
2. **少量 epoch + 小学习率**：别训太狠。
3. **混入通用数据**：训练集里掺一定比例的通用指令数据，「提醒」模型别忘本。
4. **保留验证集监控通用能力**：不只看任务指标，也测下通用能力有没有退化。

**量化训练 vs 量化部署：** 注意区分——QLoRA 是「训练时量化基座以省显存」；部署时还有**推理量化**（GPTQ、AWQ、GGUF 等），把最终模型压成低比特以加速推理、减小体积。两者目的不同但都属「量化」家族。

**合并（merge）与部署：** LoRA 训练出的是 adapter，部署有两种方式：

1. **不合并，运行时挂载**：基座 + adapter 分开加载，灵活（可热切换多个 adapter），但有轻微推理开销。
2. **合并（merge_and_unload）**：把 LoRA 权重「焊」回基座，得到一个独立的完整模型。**部署最干净**——和普通模型无异，可直接用 vLLM/TGI 高性能推理、或转 GGUF 给 Ollama/llama.cpp 本地跑。

> **专家心法**：**LoRA 是缓解灾难性遗忘的最务实手段**——因为它压根没动原始权重。如果你做了全参微调发现模型「变笨了」，八成是遗忘，先换成 LoRA 试试。部署上给个明确建议：**单一任务、追求极致推理性能 → 合并后用 vLLM；需要多任务热切换 → 不合并、动态挂载 adapter**。最后，合并务必在 fp16/bf16 下做，别在 4-bit 量化态直接合并（会掉精度）。

### 💻 代码示例

演示两件收尾事：把 LoRA adapter 合并进基座导出完整模型，以及 DPO 数据格式与训练骨架。

```python
# === 1) 合并 LoRA adapter 到基座，导出可独立部署的完整模型 ===
# pip install transformers peft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_dir = "./wenyan-lora"

# 注意：合并要在 fp16/bf16 全精度下做，不要用 4-bit 量化态合并
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_dir)   # 挂载 adapter

merged = model.merge_and_unload()                       # ← 把 LoRA 权重焊回基座
merged.save_pretrained("./wenyan-merged")               # 现在是一个普通完整模型
AutoTokenizer.from_pretrained(base_id).save_pretrained("./wenyan-merged")
print("✅ 已合并并导出，可直接用 vLLM/Ollama 部署")
```

```python
# === 2) DPO 偏好对齐的数据格式与训练骨架 ===
# pip install "trl>=0.12" "transformers>=4.46" peft datasets
# 钉版本：DPOTrainer 的 processing_class= 为较新 TRL 命名，旧环境会踩签名错误
from datasets import Dataset
from trl import DPOTrainer, DPOConfig

# DPO 数据：每条含 prompt + 人类偏好的 chosen + 不偏好的 rejected
pref_data = Dataset.from_list([
    {
        "prompt": "用户问：这个产品好用吗？",
        "chosen": "这款产品在易用性上表现优秀，多数用户反馈上手快、体验流畅。",  # 期望的风格
        "rejected": "还行吧，我也不太清楚，你自己试试。",                      # 不期望的风格
    },
    # ... 真实场景需要数百上千条偏好对
])

dpo_args = DPOConfig(
    output_dir="./dpo-out",
    per_device_train_batch_size=2,
    learning_rate=5e-6,        # DPO 学习率通常比 SFT 更小
    num_train_epochs=1,
    beta=0.1,                  # KL 约束强度：控制模型偏离参考模型的程度
    bf16=True,
)

# trainer = DPOTrainer(model=model, args=dpo_args, train_dataset=pref_data,
#                      processing_class=tokenizer, peft_config=peft_config)
# trainer.train()    # 用法与 SFTTrainer 高度一致，先 SFT 再 DPO 是常见流程
print("DPO 数据格式：{prompt, chosen, rejected}；beta 控制偏离参考模型的程度")
```

### 📚 推荐资料

- [DPO 论文](https://arxiv.org/abs/2305.18290) — 直接偏好优化的原始论文，理解它如何绕开奖励模型。
- [TRL DPOTrainer 文档](https://huggingface.co/docs/trl/dpo_trainer) — DPO 训练的官方实现与数据格式。
- [HuggingFace RLHF 博客](https://huggingface.co/blog/rlhf) — 把 RLHF 三步流程讲清楚的经典科普。
- [PEFT: Merge LoRA](https://huggingface.co/docs/peft/developer_guides/lora#merge-lora-weights-into-the-base-model) — adapter 合并的官方说明。
- [vLLM 文档](https://docs.vllm.ai/) — 合并后模型的高性能推理部署方案。

### 🛠 动手实践

把第 4 节训练好的 adapter 合并进基座，导出完整模型；然后用 `transformers` 的 `pipeline` 加载合并后的模型做推理，确认风格保留。验收标准：合并前（基座+adapter）与合并后（独立模型）对同一问题的输出一致，证明合并无损；并写一段话说明你的模型该选「合并部署」还是「挂载部署」及理由。

## 🎯 阶段实战项目：微调一个领域风格小模型并合并部署

**背景**：你要为某个细分场景打造一个「风格鲜明」的小模型——例如「鲁迅文风写作助手」「赛博朋克风文案生成器」或「严谨的法律条款解释器」。要求模型在该风格上明显优于基座的「通用 Prompt」效果，且能本地部署。

**要求**：

1. **选题与数据构建**（项目成败的关键）：
   - 选定一个清晰、可验证的风格/领域。
   - 构建 **≥100 条**高质量、风格高度一致的「指令-回答」样本（可人工写，或用更大模型蒸馏生成后人工筛选）。
   - 严格留出验证集（如 10%）。
2. **QLoRA 微调**：用 `transformers + peft + trl` + 4-bit 量化，在消费级显卡/Colab 上完成 SFT。合理设置 `r`、学习率、epoch，全程监控 train/eval loss。
3. **防过拟合与遗忘**：epoch 不超过 3；混入少量通用数据；训练后测一下模型的通用能力是否明显退化。
4. **效果评估**：设计 ≥5 个测试 prompt，做「基座 + 通用 Prompt」vs「微调后模型」的**对照评测**，从风格契合度、稳定性两方面打分。
5. **合并与部署**：用 `merge_and_unload` 合并 adapter，导出完整模型，并用 `pipeline` 或 vLLM 跑通推理。

**技术要点**：微调必要性判断、SFT 数据格式与 chat template、QLoRA 配置、SFTTrainer 流水线、loss 监控与早停、灾难性遗忘缓解、adapter 合并与部署。

**验收标准**：

- 微调后模型在目标风格上**明显且稳定地**优于「基座+Prompt」，有对照数据支撑。
- 提供 train/eval loss 曲线，能说明没有严重过拟合。
- 能验证微调后模型的通用能力没有灾难性退化（如还能正常回答常识问题）。
- 成功合并并本地推理，输出独立的完整模型目录。

**可选进阶**：

1. 在 SFT 之上再做一轮 **DPO**：为模型构造偏好对（更符合 / 不符合目标风格），用 `DPOTrainer` 精修，对比 SFT-only 与 SFT+DPO 的效果。
2. 用 **Unsloth** 重跑训练，对比训练速度与显存占用，量化它的加速收益。
3. 把合并后的模型转成 **GGUF** 格式，用 Ollama/llama.cpp 在 CPU 上本地运行，体验「完全私有化部署」。
