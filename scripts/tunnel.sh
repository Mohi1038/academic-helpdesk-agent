#!/usr/bin/env bash
# Forward localhost:11434 on this machine to Ollama inside the remote container.
# The container has no sshd, so we tunnel to the HOST and forward to the container's bridge IP.
#
#   scripts/tunnel.sh user@gpu-host [container-ip]     # container IP defaults to 172.17.0.4
#   (find the IP inside the container with: hostname -I)
set -euo pipefail

HOST="${1:?usage: scripts/tunnel.sh user@host [container-ip]}"
CONTAINER_IP="${2:-172.17.0.4}"

echo "Tunnel: localhost:11434 -> $HOST -> $CONTAINER_IP:11434  (Ctrl-C to stop)"
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L "11434:${CONTAINER_IP}:11434" "$HOST"
