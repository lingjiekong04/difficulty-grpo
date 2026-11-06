#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-train}" in
  status) exec python3 -m dagrpo.jobs status ;;
  train|pilot|matrix) exec python3 -m dagrpo.jobs start "${1:-train}" ;;
  *) echo 'Usage: bash scripts/train.sh [train|pilot|matrix|status]'; exit 1 ;;
esac
