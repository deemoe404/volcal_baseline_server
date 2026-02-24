#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PUBLIC_HOST="${PUBLIC_HOST:-}"
VENV_DIR=".venv-macos"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PID_FILE=".run/app.pid"
LOG_FILE=".run/app.log"

usage() {
  cat <<'EOF'
Usage: scripts/dev_up_macos.sh --public-host <LAN_IP> [--venv-dir <path>] [--python-bin <python>]

Starts db/minio/tusd via Docker Compose and runs FastAPI app locally in a macOS venv.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --public-host)
      PUBLIC_HOST="${2:-}"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is intended for macOS only." >&2
  exit 1
fi

if [[ -z "$PUBLIC_HOST" ]]; then
  echo "--public-host is required (example: --public-host 192.168.1.26)." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found." >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required (for libomp)." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p .run

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" >/dev/null 2>&1; then
    echo "App already running with pid $EXISTING_PID. Run scripts/dev_down_macos.sh first." >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

docker compose -f docker-compose.yml -f docker-compose.macos.yml up -d db minio mc-init tusd

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r app/requirements.txt -r app/volcal_baseline/requirements.txt
OPENMP_ROOT="$(brew --prefix libomp)" python -m pip install --editable app/volcal_baseline/py4dgeo

export DATABASE_URL="postgresql+psycopg2://tus:tuspass@localhost:5432/tasks"
export TUS_ENDPOINT="http://${PUBLIC_HOST}:1080/files"
export MINIO_ENDPOINT="http://${PUBLIC_HOST}:9000"
export MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
export MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin123}"
export AWS_S3_ADDRESSING_STYLE="${AWS_S3_ADDRESSING_STYLE:-path}"
export TORCH_DEVICE="${TORCH_DEVICE:-auto}"
export TORCH_FALLBACK_CHAIN="${TORCH_FALLBACK_CHAIN:-mps,cpu}"
export VOLCAL_NUM_WORKERS="${VOLCAL_NUM_WORKERS:-0}"
export VOLCAL_MAX_ITER="${VOLCAL_MAX_ITER:-768}"
export VOLCAL_STABLE_COREPOINT_STRIDE="${VOLCAL_STABLE_COREPOINT_STRIDE:-4}"
export VOLCAL_MAX_COREPOINTS="${VOLCAL_MAX_COREPOINTS:-120000}"
export VOLCAL_NONROI_SAMPLE_STEP="${VOLCAL_NONROI_SAMPLE_STEP:-400}"
export PUBLIC_HOST

(
  cd app
  nohup "$ROOT_DIR/$VENV_DIR/bin/uvicorn" main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --proxy-headers >"$ROOT_DIR/$LOG_FILE" 2>&1 &
  echo $! >"$ROOT_DIR/$PID_FILE"
)

APP_PID="$(cat "$PID_FILE")"
for _ in $(seq 1 30); do
  if ! kill -0 "$APP_PID" >/dev/null 2>&1; then
    echo "Failed to start FastAPI app. Check $LOG_FILE" >&2
    exit 1
  fi

  if curl -sf "http://127.0.0.1:8000/docs" >/dev/null 2>&1; then
    break
  fi

  sleep 1
done

if ! curl -sf "http://127.0.0.1:8000/docs" >/dev/null 2>&1; then
  echo "FastAPI app did not become ready in time. Check $LOG_FILE" >&2
  exit 1
fi

echo "Server stack is up."
echo "API endpoint: http://${PUBLIC_HOST}:8000"
echo "tusd endpoint: http://${PUBLIC_HOST}:1080/files"
echo "MinIO endpoint: http://${PUBLIC_HOST}:9000"
echo "App logs: $LOG_FILE"
