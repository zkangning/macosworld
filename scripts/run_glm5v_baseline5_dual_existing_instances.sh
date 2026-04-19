#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.env" ] && [ "${MACOSWORLD_NO_AUTO_ENV:-0}" != "1" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

export MACOSWORLD_NO_AUTO_ENV=1
export PYTHONUNBUFFERED=1

USED_EN_RUNTIME="${USED_EN_RUNTIME:-$ROOT_DIR/run_log/baseline5_gemini3flash_20260417/gemini3flash_baseline5_snapshot_used_en_20260417.runtime.env}"
USED_APPS_RUNTIME="${USED_APPS_RUNTIME:-$ROOT_DIR/run_log/baseline5_gemini3flash_20260417/gemini3flash_baseline5_snapshot_usedApps_en_20260417.runtime.env}"

GLM_API_KEY="${GLM_API_KEY:?GLM_API_KEY is required}"
GLM_BASE_URL="${GLM_BASE_URL:-https://openrouter.ai/api/v1}"
GLM_THINKING_MODE="${GLM_THINKING_MODE:-on}"
GLM_PROXY_URL="${GLM_PROXY_URL:-}"
GUI_AGENT_NAME="${GUI_AGENT_NAME:-z-ai/glm-5v-turbo}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/results/glm5v_baseline5}"
RUN_NAME="${RUN_NAME:-glm5v_baseline5_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/run_log/$RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-20}"
OPENROUTER_APP_NAME="${OPENROUTER_APP_NAME:-macosworld-glm5v-baseline5}"
OPENROUTER_SITE_URL="${OPENROUTER_SITE_URL:-https://github.com/openai/codex}"

mkdir -p "$LOG_DIR" "$RESULT_ROOT"

load_runtime() {
  local runtime_file="$1"
  # shellcheck disable=SC1090
  source "$runtime_file"
}

launch_worker() {
  local runtime_file="$1"
  local label="$2"
  local log_file="$3"
  local state_file="$4"
  local target_result_dir="$5"
  local session_name="${RUN_NAME}_${label}"
  local ssh_pkey_value="${MACOSWORLD_SSH_PKEY:-/Users/zhangkangning/.ssh/macosworld-gemini-smoke-20260403-key.pem}"

  tmux kill-session -t "$session_name" >/dev/null 2>&1 || true
  tmux new-session -d -s "$session_name" \
    "cd '$ROOT_DIR' && \
     RUNTIME_FILE='$runtime_file' \
     TARGET_RESULT_DIR='$target_result_dir' \
     STATE_FILE='$state_file' \
     GLM_API_KEY='$GLM_API_KEY' \
     GLM_BASE_URL='$GLM_BASE_URL' \
     GLM_THINKING_MODE='$GLM_THINKING_MODE' \
     GLM_PROXY_URL='$GLM_PROXY_URL' \
     GUI_AGENT_NAME='$GUI_AGENT_NAME' \
     MAX_STEPS='$MAX_STEPS' \
     MACOSWORLD_SSH_PKEY='$ssh_pkey_value' \
     OPENROUTER_APP_NAME='$OPENROUTER_APP_NAME' \
     OPENROUTER_SITE_URL='$OPENROUTER_SITE_URL' \
     bash '$ROOT_DIR/scripts/run_glm5v_baseline5_worker.sh' > '$log_file' 2>&1"

  echo "$session_name" > "${log_file}.session"
  echo "launched $label tmux_session=$session_name log=$log_file"
}

launch_worker \
  "$USED_EN_RUNTIME" \
  "used_en" \
  "$LOG_DIR/${RUN_NAME}_used_en.log" \
  "$ROOT_DIR/run_log/${RUN_NAME}_used_en_instance_state.json" \
  "$RESULT_ROOT/snapshot_used_en"

launch_worker \
  "$USED_APPS_RUNTIME" \
  "usedApps_en" \
  "$LOG_DIR/${RUN_NAME}_usedApps_en.log" \
  "$ROOT_DIR/run_log/${RUN_NAME}_usedApps_en_instance_state.json" \
  "$RESULT_ROOT/snapshot_usedApps_en"
