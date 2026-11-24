#!/usr/bin/env bash
# Скачивает готовую ONNX-модель и словарь токенизатора с Hugging Face в ./model
set -euo pipefail
BASE=https://huggingface.co/onnx-community/rubert-tiny-sentiment-balanced-ONNX/resolve/main
mkdir -p model
curl -L "$BASE/onnx/model_int8.onnx" -o model/model_int8.onnx   # ~12 МБ, int8-квантизация
curl -L "$BASE/vocab.txt"            -o model/vocab.txt          # ~240 КБ, словарь WordPiece
curl -L "$BASE/config.json"          -o model/config.json        # id2label: negative/neutral/positive
ls -lh model
