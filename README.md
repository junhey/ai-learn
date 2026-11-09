# AI-Learn · A hands-on course for full-stack AI engineers

[中文](README.zh-CN.md)

A complete course from how LLMs work through production. The lessons are written in Markdown, and one command builds both a **reading site** and **Reveal.js lecture slides**.

## Layout

```
ai-learn/
├── content/            # Course content (Markdown, the only source of truth)
│   ├── stage-01.md     # Foundations · LLM principles and prompting
│   ├── stage-02.md     # Orchestration · LangChain / LlamaIndex
│   ├── stage-03.md     # RAG · retrieval-augmented generation
│   ├── stage-04.md     # Agents and MCP
│   ├── stage-05.md     # Fine-tuning
│   ├── stage-06.md     # Evals
│   ├── stage-07.md     # Inference optimization and deployment
│   ├── stage-08.md     # Production engineering · LLMOps
│   └── stage-09.md     # Capstone · an end-to-end AI product
├── assets/style.css    # Shared design system
├── build.py            # Build script, including a content linter
├── _SPEC.md            # Writing rules for lesson content
├── run.sh              # Build and preview locally
└── dist/               # Build output (generated)
    ├── index.html      # Reading site home
    ├── stage-XX.html   # One page per stage
    └── slides/         # Reveal.js lecture slides
```

## Quick start

```bash
# First run: create a virtualenv and install dependencies
python3 -m venv .venv
./.venv/bin/pip install markdown pygments

# Build (lint runs first and rejects invalid content)
./.venv/bin/python build.py

# Lint only
./.venv/bin/python build.py --lint

# Build and preview locally
./run.sh
```

Open `dist/index.html` to read the course, or `dist/slides/index.html` for lecture mode.

## Lecture-mode keys (Reveal.js)

| Key | Action |
|----|------|
| `→` / `Space` | Next slide |
| `↓` | Open the detail slides for the current section |
| `S` | Open speaker notes |
| `F` | Full screen |
| `ESC` | Slide overview |

## What each stage contains

Every lesson has four parts, and `build.py` checks that they are present:

- **Concepts** — What / Why / How, plus a practitioner's note
- **Code** — Runnable, commented examples against current APIs
- **Further reading** — Official docs, papers, and substantial posts
- **Practice** — A short exercise with acceptance criteria

Each stage ends with a **project**.

## Editing a lesson

Edit `content/stage-XX.md`, then run `./.venv/bin/python build.py` again. The writing rules live in `_SPEC.md`. The linter checks content against that file.
