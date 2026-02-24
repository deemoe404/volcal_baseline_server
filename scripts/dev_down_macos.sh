#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_FILE=".run/app.pid"

if [[ -f "$PID_FILE" ]]; then
  APP_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" >/dev/null 2>&1; then
    kill "$APP_PID" >/dev/null 2>&1 || true
    sleep 1
    if kill -0 "$APP_PID" >/dev/null 2>&1; then
      kill -9 "$APP_PID" >/dev/null 2>&1 || true
    fi
    echo "Stopped app process $APP_PID"
  fi
  rm -f "$PID_FILE"
fi

docker compose -f docker-compose.yml -f docker-compose.macos.yml down --remove-orphans

echo "Stopped macOS dev stack."
