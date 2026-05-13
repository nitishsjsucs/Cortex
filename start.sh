#!/usr/bin/env bash
# start.sh — Launch the full Cortex demo stack (no Docker required)
#
# Starts:
#   1. MLflow tracking server  → http://localhost:5000
#   2. FastAPI inference API   → http://localhost:8001  (/docs for Swagger)
#   3. Gradio ML demo          → http://localhost:7860
#   4. React web frontend      → http://localhost:4028
#
# Usage:
#   chmod +x start.sh
#   ./start.sh
#
# Stop all:
#   ./start.sh stop

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_ROOT/.logs"
PID_FILE="$REPO_ROOT/.pids"
mkdir -p "$LOG_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${BLUE}[Cortex]${NC} $*"; }
ok()    { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
err()   { echo -e "${RED}[ERROR ]${NC} $*"; }

# ── Stop mode ─────────────────────────────────────────────────────────────────
if [[ "$1" == "stop" ]]; then
  info "Stopping all Cortex services..."
  if [[ -f "$PID_FILE" ]]; then
    while IFS= read -r pid; do
      kill "$pid" 2>/dev/null && echo "  killed PID $pid" || true
    done < "$PID_FILE"
    rm -f "$PID_FILE"
  fi
  # Kill by port as fallback
  for port in 5000 7860 8001 4028; do
    pid=$(lsof -ti tcp:"$port" 2>/dev/null) && kill $pid 2>/dev/null && echo "  freed port $port" || true
  done
  ok "All services stopped."
  exit 0
fi

# ── Prerequisite checks ───────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || { err "python3 not found"; exit 1; }
command -v node    >/dev/null 2>&1 || { err "node not found";    exit 1; }
command -v npm     >/dev/null 2>&1 || { err "npm not found";     exit 1; }

# ── Install Python deps if needed ─────────────────────────────────────────────
info "Checking Python dependencies..."
python3 -c "import mlflow, fastapi, gradio, torch, transformers" 2>/dev/null || {
  info "Installing Python dependencies (first run)..."
  python3 -m pip install -r "$REPO_ROOT/ml/requirements.txt" -q
}
ok "Python deps ready"

# ── Install Node deps if needed ───────────────────────────────────────────────
info "Checking Node dependencies..."
if [[ ! -d "$REPO_ROOT/apps/web/node_modules" ]]; then
  info "Installing Node dependencies (first run)..."
  (cd "$REPO_ROOT/apps/web" && npm install --silent)
fi
ok "Node deps ready"

# ── Generate synthetic training data if missing ───────────────────────────────
if [[ ! -f "$REPO_ROOT/ml/data/arxiv.csv" ]]; then
  info "Generating synthetic training data..."
  PYTHONPATH="$REPO_ROOT" python3 -c "
from ml.data.arxiv_loader import _generate_synthetic_data
from pathlib import Path
Path('ml/data').mkdir(parents=True, exist_ok=True)
df = _generate_synthetic_data(n=500)
df.to_csv('ml/data/arxiv.csv', index=False)
print(f'  Generated {len(df)} samples')
"
fi

> "$PID_FILE"  # reset PID file

# ── 1. MLflow Tracking Server ─────────────────────────────────────────────────
info "Starting MLflow server on http://localhost:5000 ..."
PYTHONPATH="$REPO_ROOT" python3 -m mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri "$REPO_ROOT/mlruns" \
  --default-artifact-root "$REPO_ROOT/mlruns/artifacts" \
  > "$LOG_DIR/mlflow.log" 2>&1 &
echo $! >> "$PID_FILE"
sleep 2
ok "MLflow  → http://localhost:5000"

# ── 2. FastAPI Inference Server ───────────────────────────────────────────────
info "Starting FastAPI inference server on http://localhost:8001 ..."
PYTHONPATH="$REPO_ROOT" \
MODEL_CHECKPOINT="$REPO_ROOT/ml/outputs/models/best_model.pt" \
MLFLOW_TRACKING_URI="http://127.0.0.1:5000" \
python3 -m uvicorn ml.inference.api:app \
  --host 127.0.0.1 --port 8001 \
  > "$LOG_DIR/api.log" 2>&1 &
echo $! >> "$PID_FILE"
sleep 3
ok "API     → http://localhost:8001  (Swagger: http://localhost:8001/docs)"

# ── 3. Gradio ML Demo ─────────────────────────────────────────────────────────
info "Starting Gradio demo on http://localhost:7860 ..."
PYTHONPATH="$REPO_ROOT" \
MODEL_CHECKPOINT="$REPO_ROOT/ml/outputs/models/best_model.pt" \
python3 "$REPO_ROOT/ml/gradio_demo.py" --port 7860 \
  > "$LOG_DIR/gradio.log" 2>&1 &
echo $! >> "$PID_FILE"
sleep 4
ok "Gradio  → http://localhost:7860"

# ── 4. React Web Frontend ─────────────────────────────────────────────────────
info "Starting React web app on http://localhost:4028 ..."
(cd "$REPO_ROOT/apps/web" && npm run dev -- --host 127.0.0.1) \
  > "$LOG_DIR/web.log" 2>&1 &
echo $! >> "$PID_FILE"
sleep 3
ok "Web     → http://localhost:4028"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Cortex Demo Stack — All Services Up          ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║${NC}  React web app   →  http://localhost:4028            ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  Gradio ML demo  →  http://localhost:7860            ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  FastAPI server  →  http://localhost:8001/docs       ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  MLflow UI       →  http://localhost:5000            ${GREEN}║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║${NC}  Logs: .logs/   •   Stop: ./start.sh stop           ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
info "Watching logs (Ctrl+C to detach, services keep running)..."
tail -f "$LOG_DIR/api.log" "$LOG_DIR/gradio.log" "$LOG_DIR/web.log"
