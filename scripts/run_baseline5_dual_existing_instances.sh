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
GUI_AGENT_NAME="${GUI_AGENT_NAME:?GUI_AGENT_NAME is required}"
RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT is required}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/run_log/$RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-20}"
mkdir -p "$LOG_DIR" "$RESULT_ROOT"

load_runtime() {
  local runtime_file="$1"
  # shellcheck disable=SC1090
  source "$runtime_file"
}

launch_worker() {
  local runtime_file="$1"
  local label_override="$2"
  local log_file="$3"
  local state_file="$4"
  local target_result_dir="$5"

  (
    load_runtime "$runtime_file"

    export MACOSWORLD_INSTANCE_ID="$instance_id"
    export MACOSWORLD_SSH_HOST="$dns_name"
    export MACOSWORLD_SSH_PKEY="${MACOSWORLD_SSH_PKEY:-/Users/zhangkangning/.ssh/macosworld-gemini-smoke-20260403-key.pem}"
    export MACOSWORLD_GUI_AGENT_NAME="$GUI_AGENT_NAME"
    export MACOSWORLD_NO_AUTO_ENV=1
    export MACOSWORLD_PATHS_TO_EVAL_TASKS="$task_dir"
    export MACOSWORLD_LANGUAGES="task_en_env_en"
    export MACOSWORLD_BASE_SAVE_DIR="$target_result_dir"
    export MACOSWORLD_AWS_INSTANCE_STATE_FILE="$state_file"
    export MACOSWORLD_AWS_INITIAL_SNAPSHOT_NAME="$initial_snapshot"
    export MACOSWORLD_MAX_STEPS="$MAX_STEPS"
    exec bash "$ROOT_DIR/scripts/run_aws_eval.sh"
  ) > "$log_file" 2>&1 &

  echo "$!" > "${log_file}.pid"
  echo "launched $label_override pid=$(cat "${log_file}.pid") log=$log_file"
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
