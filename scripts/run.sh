#!/usr/bin/env bash
# Thin wrappers around the radioprotect-sm CLI for documented workflows.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"

case "${1:-}" in
  demo)
    radioprotect-sm run-all --demo
    ;;
  fetch)
    radioprotect-sm fetch-chembl "${@:2}"
    ;;
  train)
    radioprotect-sm train "${@:2}"
    ;;
  screen)
    radioprotect-sm screen "${@:2}"
    ;;
  test)
    pytest
    ;;
  *)
    echo "Usage: $0 {demo|fetch|train|screen|test}"
    exit 1
    ;;
esac
