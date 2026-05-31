#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Learn 构建脚本
  content/stage-XX.md  ──>  dist/index.html + dist/stage-XX.html (课程阅读站)
                       ──>  dist/slides/stage-XX.html (Reveal.js 授课 PPT)

用法:
  python build.py            # 构建全部
  python build.py --lint     # 仅校验内容是否合规(不出错才能发布)
"""
import os, re, sys, glob, html as _html
import markdown

ROOT = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(ROOT, "content")
DIST = os.path.join(ROOT, "dist")
SLIDES = os.path.join(DIST, "slides")

LEVEL_COLOR = {"基础":"green","进阶":"accent","高级":"accent2","部署":"cyan","综合":"pink"}
REQUIRED_SECTIONS = ["📖 知识点", "💻 代码示例", "📚 推荐资料", "🛠 动手实践"]

# ---------- frontmatter ----------
def parse_frontmatter(text):
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
    if not m:
        return {}, text
    raw, body = m.group(1), m.group(2)
    meta = {}
    for line in raw.splitlines():
        if ":" not in line: continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if v.startswith("[") and v.endswith("]"):
            v = [x.strip().strip('"\'') for x in v[1:-1].split(",") if x.strip()]
        else:
            v = v.strip('"\'')
        meta[k] = v
    return meta, body

# ---------- linter ----------
def lint_file(path):
    errs = []
    with open(path, encoding="utf-8") as f:
        text = f.read()
    meta, body = parse_frontmatter(text)
    name = os.path.basename(path)
    for key in ("stage","title","level","goal"):
        if key not in meta:
            errs.append(f"{name}: 缺少 frontmatter 字段 `{key}`")
    if meta.get("level") not in LEVEL_COLOR:
        errs.append(f"{name}: level 非法 (={meta.get('level')})")
    # 章节: 每个 ## 章节须含四要素 (排除 🎯 实战项目章节)
    chapters = re.split(r"\n##\s+", "\n"+body)
    chap_titles = re.findall(r"\n##\s+(.+)", "\n"+body)
    real_chaps = [c for c,t in zip(chapters[1:], chap_titles) if "🎯" not in t]
    if len(real_chaps) < 3:
        errs.append(f"{name}: 章节数不足3 (实际 {len(real_chaps)})")
    for t, c in zip(chap_titles, chapters[1:]):
        if "🎯" in t: continue
        for sec in REQUIRED_SECTIONS:
            if sec not in c:
                errs.append(f"{name}: 章节「{t.strip()[:20]}」缺少要素「{sec}」")
    if "🎯 阶段实战项目" not in body:
        errs.append(f"{name}: 缺少「🎯 阶段实战项目」")
    # 代码块存在性
    if body.count("```") < 4:
        errs.append(f"{name}: 代码块数量过少 (需≥2个完整代码块)")
    return errs, meta

# ---------- markdown -> html ----------
MD_EXT = ["fenced_code","codehilite","tables","attr_list","toc"]
def md2html(body):
    return markdown.markdown(body, extensions=MD_EXT,
                             extension_configs={"codehilite":{"guess_lang":False}})

# ---------- 阅读站页面 ----------
def page_shell(title, sidebar, content):
    return f"""<!DOCTYPE html><html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html.escape(title)}</title>
<link rel="stylesheet" href="assets/style.css"></head><body>
<div class="layout"><aside class="sidebar">{sidebar}</aside>
<main class="content">{content}</main></div></body></html>"""

def build_sidebar(stages, active=None):
    links = ['<a href="index.html"%s><span class="num">★</span> 课程总览</a>'
             % (' class="active"' if active is None else '')]
    for s in stages:
        cls = ' class="active"' if active == s["stage"] else ''
        links.append(f'<a href="stage-{s["stage"]:02d}.html"{cls}>'
                     f'<span class="num">{s["stage"]:02d}</span> {_html.escape(s["short"])}</a>')
    return ('<div class="brand">AI-Learn</div>'
            '<div class="brand-sub">FULL-STACK AI ENGINEER</div>'
            '<div class="nav-group">学习路径</div>'
            f'<nav class="nav">{"".join(links)}</nav>'
            '<div class="nav-group">授课模式</div>'
            '<nav class="nav"><a href="slides/index.html"><span class="num">▶</span> 打开 PPT 演讲</a></nav>')

def build_stage_page(s, stages, body_html):
    sidebar = build_sidebar(stages, active=s["stage"])
    lv = s["level"]
    stack = "".join(f"<span>{_html.escape(x)}</span>" for x in s.get("stack",[]))
    prev_s = next((x for x in stages if x["stage"]==s["stage"]-1), None)
    next_s = next((x for x in stages if x["stage"]==s["stage"]+1), None)
    pager = '<div class="pager">'
    if prev_s:
        pager += f'<a href="stage-{prev_s["stage"]:02d}.html"><div class="dir">← 上一阶段</div><div class="ttl">{_html.escape(prev_s["short"])}</div></a>'
    else:
        pager += '<a href="index.html"><div class="dir">← 返回</div><div class="ttl">课程总览</div></a>'
    if next_s:
        pager += f'<a class="next" href="stage-{next_s["stage"]:02d}.html"><div class="dir">下一阶段 →</div><div class="ttl">{_html.escape(next_s["short"])}</div></a>'
    else:
        pager += '<a class="next" href="index.html"><div class="dir">完成 →</div><div class="ttl">回到总览</div></a>'
    pager += '</div>'
    head = (f'<div class="crumb">课程总览 / 阶段 {s["stage"]:02d}</div>'
            f'<div class="stage-head"><span class="lv lv-{lv}">{lv}</span>'
            f'<h1>{_html.escape(s["title"])}</h1>'
            f'<div class="goal">{_html.escape(s["goal"])}</div>'
            f'<div class="stage-meta"><div>⏱ 周期 <b>{_html.escape(s.get("duration","-"))}</b></div>'
            f'<div>📋 前置 <b>{_html.escape(s.get("prereq","无"))}</b></div></div>'
            f'<div class="stack-row">{stack}</div></div>')
    return page_shell(s["title"]+" · AI-Learn", sidebar, head + body_html + pager)

def build_index(stages):
    sidebar = build_sidebar(stages, active=None)
    total_proj = len(stages)
    hero = ('<div class="hero"><div class="kicker">Full-Stack AI Engineer · Curriculum</div>'
            '<h1>全栈 AI Engineer<br>实战课程</h1>'
            '<p>从 LLM 原理到生产落地的完整学习体系。9 大阶段 · 每阶段含知识点详解、可运行代码、推荐资料、动手实践与实战项目。</p>'
            f'<div class="stat-row"><div><b>{len(stages)}</b><span>核心阶段</span></div>'
            f'<div><b>{total_proj}</b><span>实战项目</span></div>'
            '<div><b>40+</b><span>关键技术栈</span></div>'
            '<div><b>≈6 月</b><span>建议周期</span></div></div></div>')
    cards = '<div class="card-grid">'
    for s in stages:
        lv = s["level"]
        cards += (f'<a class="scard" href="stage-{s["stage"]:02d}.html">'
                  f'<div class="row"><div class="n">{s["stage"]:02d}</div>'
                  f'<h3>{_html.escape(s["short"])}</h3>'
                  f'<span class="lv2 lv-{lv}">{lv}</span></div>'
                  f'<p>{_html.escape(s["goal"])}</p></a>')
    cards += '</div><div style="height:80px"></div>'
    return page_shell("AI-Learn · 全栈 AI Engineer 实战课程", sidebar, hero + cards)

# ---------- Reveal.js 幻灯片 ----------
def md_to_slides(meta, body):
    """把一个阶段的 markdown 切成 reveal sections。
    规则: ## -> 水平新章节(含本章四要素作为纵向子页)。"""
    slides = []
    # 封面
    lv = meta.get("level","")
    slides.append(f"""<section data-background-gradient="linear-gradient(135deg,#171a21,#1d212b)">
<div class="lv-badge lv-{lv}">{lv} · 阶段 {meta.get('stage')}</div>
<h1 class="cover-h1">{_html.escape(meta.get('title',''))}</h1>
<p class="cover-goal">{_html.escape(meta.get('goal',''))}</p>
<aside class="notes">本阶段目标: {_html.escape(meta.get('goal',''))}。前置要求: {_html.escape(meta.get('prereq','无'))}。</aside>
</section>""")
    # 按 ## 切章
    parts = re.split(r"\n##\s+", "\n"+body)
    titles = re.findall(r"\n##\s+(.+)", "\n"+body)
    for title, chunk in zip(titles, parts[1:]):
        is_proj = "🎯" in title
        # 章节标题页(横向)
        v_subs = []
        # 子要素按 ### 切
        subs = re.split(r"\n###\s+", "\n"+chunk)
        sub_titles = re.findall(r"\n###\s+(.+)", "\n"+chunk)
        intro_html = md2html(subs[0]) if subs and subs[0].strip() else ""
        section_title = f"""<section>
<h2 class="chap-title">{_html.escape(title.strip())}</h2>
{intro_html}
<aside class="notes">讲解「{_html.escape(title.strip())}」。按 ↓ 进入细节。</aside>
</section>"""
        v_subs.append(section_title)
        for st, sc in zip(sub_titles, subs[1:]):
            inner = md2html(sc)
            v_subs.append(f"""<section>
<h3 class="sub-title">{_html.escape(st.strip())}</h3>
<div class="slide-body">{inner}</div>
<aside class="notes">{_html.escape(st.strip())}: 重点讲清原理与示例。</aside>
</section>""")
        # 纵向组
        if len(v_subs) > 1:
            slides.append("<section>" + "".join(v_subs) + "</section>")
        else:
            slides.append(v_subs[0])
    return "\n".join(slides)

def build_slide_deck(meta, body):
    sections = md_to_slides(meta, body)
    return f"""<!DOCTYPE html><html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html.escape(meta.get('title',''))} · 授课</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/theme/black.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/plugin/highlight/monokai.css">
<style>
.reveal{{font-family:-apple-system,"PingFang SC",sans-serif}}
.reveal h1,.reveal h2,.reveal h3{{text-transform:none;letter-spacing:-.5px;color:#e6e9ef}}
.cover-h1{{font-size:2.0em;background:linear-gradient(90deg,#fff,#5b8cff,#8b5cf6);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}}
.cover-goal{{color:#9aa3b2;font-size:.7em}}
.chap-title{{font-size:1.5em;color:#5b8cff}}
.sub-title{{font-size:1.0em;color:#22d3ee;border-bottom:2px solid #2a2f3a;padding-bottom:10px}}
.slide-body{{font-size:.62em;line-height:1.55;text-align:left}}
.reveal pre{{width:100%;font-size:.5em;box-shadow:none;border:1px solid #2a2f3a;border-radius:10px}}
.reveal pre code{{max-height:520px;padding:14px}}
.reveal table{{font-size:.55em}}
.reveal ul,.reveal ol{{font-size:.95em}}
.reveal a{{color:#5b8cff}}
.reveal blockquote{{font-size:.6em;background:rgba(251,191,36,.08);border-left:3px solid #fbbf24;width:100%}}
.lv-badge{{display:inline-block;font-size:.4em;padding:6px 16px;border-radius:30px;color:#0f1115;font-weight:700;margin-bottom:20px}}
.lv-基础{{background:#34d399}}.lv-进阶{{background:#5b8cff}}.lv-高级{{background:#8b5cf6}}.lv-部署{{background:#22d3ee}}.lv-综合{{background:#f472b6}}
.reveal .slide-body :not(pre)>code{{background:#1d212b;color:#ffd9a8;padding:1px 6px;border-radius:5px;font-size:.85em}}
.topbar{{position:fixed;top:12px;left:16px;z-index:30;font-size:13px;color:#6b7280}}
.topbar a{{color:#5b8cff;text-decoration:none}}
</style></head><body>
<div class="topbar"><a href="index.html">← 全部课程</a> · 按 <b>S</b> 看演讲备注 · <b>F</b> 全屏 · <b>ESC</b> 总览</div>
<div class="reveal"><div class="slides">
{sections}
</div></div>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.js"></script>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5/plugin/highlight/highlight.js"></script>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5/plugin/notes/notes.js"></script>
<script>
Reveal.initialize({{hash:true,slideNumber:'c/t',transition:'slide',
  plugins:[RevealHighlight,RevealNotes]}});
</script></body></html>"""

def build_slides_index(stages):
    items = ""
    for s in stages:
        lv = s["level"]
        items += (f'<a class="scard" href="stage-{s["stage"]:02d}.html">'
                  f'<div class="row"><div class="n">{s["stage"]:02d}</div>'
                  f'<h3>{_html.escape(s["short"])}</h3>'
                  f'<span class="lv2 lv-{lv}">{lv}</span></div>'
                  f'<p>{_html.escape(s["goal"])}</p></a>')
    body = ('<div class="hero"><div class="kicker">Teaching Mode · Reveal.js</div>'
            '<h1>授课 PPT 模式</h1>'
            '<p>点击任意阶段进入全屏可演讲幻灯片。支持 S 键演讲者备注、键盘翻页、ESC 总览。</p></div>'
            f'<div class="card-grid">{items}</div>'
            '<div style="text-align:center;margin-top:40px"><a href="../index.html">← 返回课程阅读站</a></div>'
            '<div style="height:60px"></div>')
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>授课 PPT · AI-Learn</title>
<link rel="stylesheet" href="../assets/style.css"></head><body>{body}</body></html>"""

# ---------- main ----------
def load_stages():
    files = sorted(glob.glob(os.path.join(CONTENT, "stage-*.md")))
    stages = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            meta, body = parse_frontmatter(f.read())
        meta["stage"] = int(meta["stage"])
        title = meta.get("title","")
        meta["short"] = title.split("·")[-1].strip() if "·" in title else title
        meta["_body"] = body
        meta["_file"] = fp
        stages.append(meta)
    stages.sort(key=lambda x: x["stage"])
    return stages

def do_lint():
    files = sorted(glob.glob(os.path.join(CONTENT, "stage-*.md")))
    all_errs = []
    for fp in files:
        errs, _ = lint_file(fp)
        all_errs += errs
    if not files:
        print("⚠️  没有找到任何 content/stage-*.md")
        return 1
    if all_errs:
        print(f"❌ Lint 失败，共 {len(all_errs)} 个问题:\n")
        for e in all_errs: print("  -", e)
        return 1
    print(f"✅ Lint 通过：{len(files)} 个阶段全部符合规范")
    return 0

def do_build():
    rc = do_lint()
    if rc != 0:
        print("\n⛔ 内容未通过校验，停止构建（测试失败不发布）")
        return rc
    stages = load_stages()
    os.makedirs(DIST, exist_ok=True)
    os.makedirs(SLIDES, exist_ok=True)
    # 资源拷贝
    import shutil
    src_css = os.path.join(ROOT,"assets","style.css")
    os.makedirs(os.path.join(DIST,"assets"), exist_ok=True)
    shutil.copy(src_css, os.path.join(DIST,"assets","style.css"))
    # 阅读站
    with open(os.path.join(DIST,"index.html"),"w",encoding="utf-8") as f:
        f.write(build_index(stages))
    for s in stages:
        body_html = md2html(s["_body"])
        with open(os.path.join(DIST,f"stage-{s['stage']:02d}.html"),"w",encoding="utf-8") as f:
            f.write(build_stage_page(s, stages, body_html))
        # 幻灯片
        with open(os.path.join(SLIDES,f"stage-{s['stage']:02d}.html"),"w",encoding="utf-8") as f:
            f.write(build_slide_deck(s, s["_body"]))
    with open(os.path.join(SLIDES,"index.html"),"w",encoding="utf-8") as f:
        f.write(build_slides_index(stages))
    print(f"\n🎉 构建完成 → {DIST}")
    print(f"   阅读站: dist/index.html  ({len(stages)} 阶段)")
    print(f"   授课PPT: dist/slides/index.html")
    return 0

if __name__ == "__main__":
    if "--lint" in sys.argv:
        sys.exit(do_lint())
    sys.exit(do_build())
