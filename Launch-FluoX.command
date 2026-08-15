#!/bin/zsh
cd "$(dirname "$0")" || exit 1
if [ -x ".venv/bin/python" ]; then
  .venv/bin/python webapp.py
else
  python3 webapp.py
fi
