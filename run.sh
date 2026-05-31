#!/usr/bin/env bash
# 一键构建并本地预览 AI-Learn 课程
set -e
cd "$(dirname "$0")"

PY=./.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "未找到虚拟环境，正在创建..."
  python3 -m venv .venv
  ./.venv/bin/pip install -q markdown pygments
fi

echo "▶ 构建课程..."
$PY build.py

echo ""
echo "▶ 启动本地服务器 http://localhost:8000"
echo "   课程阅读站: http://localhost:8000/index.html"
echo "   授课 PPT:   http://localhost:8000/slides/index.html"
echo "   (Ctrl+C 停止)"
cd dist && $PY -m http.server 8000
