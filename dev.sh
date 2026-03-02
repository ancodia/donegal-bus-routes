#!/usr/bin/env bash
# dev.sh — start the FastAPI backend and Vite frontend together.
# Usage: ./dev.sh
# Kill with Ctrl-C; both processes are cleaned up automatically.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

API_CMD=".venv/bin/uvicorn main:app --reload --host 127.0.0.1 --port 8000"
WEB_DIR="$ROOT/web"

# ── Preflight checks ─────────────────────────────────────────────────────────

if [[ ! -f "$ROOT/.venv/bin/uvicorn" ]]; then
  echo "ERROR: .venv not found. Run 'uv sync' first." >&2
  exit 1
fi

if [[ ! -d "$WEB_DIR/node_modules" ]]; then
  echo "node_modules missing — installing frontend dependencies..."
  (cd "$WEB_DIR" && npm install)
fi

# ── Process management ───────────────────────────────────────────────────────

API_PID=""
WEB_PID=""

cleanup() {
  echo ""
  echo "Shutting down..."
  [[ -n "$API_PID" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "$WEB_PID" ]] && kill "$WEB_PID" 2>/dev/null || true
  wait "$API_PID" "$WEB_PID" 2>/dev/null || true
}

trap cleanup INT TERM

# ── Start servers ────────────────────────────────────────────────────────────

echo "Starting API      → http://localhost:8000"
(cd "$ROOT" && $API_CMD) &
API_PID=$!

echo "Starting frontend → http://localhost:5173"
(cd "$WEB_DIR" && npm run dev) &
WEB_PID=$!

# Wait for either process to exit
wait -n "$API_PID" "$WEB_PID"
EXIT_CODE=$?

cleanup
exit $EXIT_CODE
