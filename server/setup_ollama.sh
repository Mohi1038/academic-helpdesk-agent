#!/usr/bin/env bash
# Run INSIDE the GPU container on the remote server. Installs Ollama, starts it on the
# container's bridge interface, and pulls the models used by the helpdesk.
#
#   bash server/setup_ollama.sh            # install + start + pull
#   OLLAMA_MODELS=/data/ollama bash server/setup_ollama.sh   # keep models on a persistent volume
set -euo pipefail

export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"    # reachable from the host via the bridge IP
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"  # default 2k-4k silently truncates prompts
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-2}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
# On a shared multi-GPU box, pin Ollama to one GPU, e.g. CUDA_VISIBLE_DEVICES=1 bash server/setup_ollama.sh
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && export CUDA_VISIBLE_DEVICES
MODELS=(qwen3:8b llama3.1:8b granite3.3:8b nomic-embed-text)

nvidia-smi --query-gpu=name,memory.total --format=csv || {
  echo "No GPU visible in this container (was it started with --gpus all?)" >&2; exit 1; }

if ! command -v ollama >/dev/null; then
  SUDO=$([ "$(id -u)" -eq 0 ] || echo sudo)
  # the installer ships a .tar.zst archive
  command -v zstd >/dev/null || { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq zstd curl; }
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null; then
  mkdir -p "$OLLAMA_MODELS"
  nohup ollama serve > "$HOME/ollama.log" 2>&1 &
  for _ in $(seq 30); do curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null && break; sleep 1; done
fi

# The V100 (compute capability 7.0) must run on the bundled CUDA 12 runtime; a "cpu" library here means no GPU.
grep -iE "inference compute|library=" "$HOME/ollama.log" | tail -3 || true

for m in "${MODELS[@]}"; do ollama pull "$m"; done
ollama list

echo
echo "Ollama is listening on $(hostname -I | awk '{print $1}'):11434"
echo "From your laptop:  scripts/tunnel.sh <user@host> $(hostname -I | awk '{print $1}')"
