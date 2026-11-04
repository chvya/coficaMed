#!/usr/bin/env bash
set -e
echo "=== Instalando CodificaMed ==="
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo
echo "=== Listo ==="
echo "Siguiente paso: ollama pull qwen2.5:3b && ollama pull bge-m3"
