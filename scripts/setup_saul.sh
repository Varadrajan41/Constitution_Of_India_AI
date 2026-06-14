#!/usr/bin/env bash
# Download SaulLM-7B-Instruct (GGUF) and register it with Ollama.
#
# Requires: huggingface-hub CLI (`pip install -U "huggingface_hub[cli]"`) and ollama.
# Usage:    bash scripts/setup_saul.sh
set -euo pipefail

cd "$(dirname "$0")"

REPO="${SAUL_GGUF_REPO:-MaziyarPanahi/SaulLM-7B-Instruct-GGUF}"
FILE="${SAUL_GGUF_FILE:-SaulLM-7B-Instruct.Q4_K_M.gguf}"
TARGET="saul-7b-instruct.Q4_K_M.gguf"

if [ ! -f "$TARGET" ]; then
  echo "Downloading $FILE from $REPO ..."
  huggingface-cli download "$REPO" "$FILE" --local-dir . --local-dir-use-symlinks False
  # Normalise the filename the Modelfile expects.
  [ -f "$FILE" ] && mv -f "$FILE" "$TARGET" || true
fi

echo "Registering model with Ollama as 'saul-7b-instruct' ..."
ollama create saul-7b-instruct -f Modelfile.saul
echo "Done. Test with: ollama run saul-7b-instruct 'Hello'"
