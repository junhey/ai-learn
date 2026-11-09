# AI-Learn · 全栈 AI Engineer 实战课程

[English](README.md)


一套从 LLM 原理到生产落地的完整课程。内容用 Markdown 编写，一键构建为**课程阅读站**和**Reveal.js 授课 PPT**。

## 目录结构

```
ai-learn/
├── content/            # 课程内容（Markdown，唯一数据源）
│   ├── stage-01.md     # 地基 · LLM 原理与 Prompt
│   ├── stage-02.md     # 编排框架 · LangChain / LlamaIndex
│   ├── stage-03.md     # RAG · 检索增强生成
│   ├── stage-04.md     # Agent & MCP · 智能体
│   ├── stage-05.md     # 模型微调 · Fine-tuning
│   ├── stage-06.md     # 评估体系 · Evals
│   ├── stage-07.md     # 推理优化 · 部署
│   ├── stage-08.md     # 生产工程 · LLMOps
│   └── stage-09.md     # 毕业项目 · 端到端 AI 产品
├── assets/style.css    # 共享设计系统
├── build.py            # 构建脚本（含内容 linter）
├── _SPEC.md            # 内容写作规范（写作契约）
├── run.sh              # 一键构建并本地预览
└── dist/               # 构建产物（自动生成）
    ├── index.html      # 课程阅读站首页
    ├── stage-XX.html   # 各阶段阅读页
    └── slides/         # Reveal.js 授课 PPT
```

## 快速开始

```bash
# 首次：建虚拟环境装依赖
python3 -m venv .venv
./.venv/bin/pip install markdown pygments

# 构建（会先跑 lint，不合规则拒绝构建）
./.venv/bin/python build.py

# 仅校验内容合规
./.venv/bin/python build.py --lint

# 一键构建 + 本地预览
./run.sh
```

打开 `dist/index.html` 阅读课程，或 `dist/slides/index.html` 进入授课模式。

## 授课模式快捷键（Reveal.js）

| 键 | 作用 |
|----|------|
| `→` / `Space` | 下一页 |
| `↓` | 进入当前章节的细节子页 |
| `S` | 打开演讲者备注窗口 |
| `F` | 全屏 |
| `ESC` | 幻灯片总览 |

## 每个阶段的内容结构

每节课固定包含四要素（由 `build.py` 强制校验）：

- 📖 **知识点** — What / Why / How + 专家洞察
- 💻 **代码示例** — 可运行、带注释的现代 API 代码
- 📚 **推荐资料** — 真实的官方文档 / 论文 / 优质博客
- 🛠 **动手实践** — 有验收标准的小练习

阶段末尾有 🎯 **实战项目**。

## 如何修改内容

直接编辑 `content/stage-XX.md`，然后重新 `./.venv/bin/python build.py`。
写作规范见 `_SPEC.md`——这是内容质量的契约，linter 据此校验。
