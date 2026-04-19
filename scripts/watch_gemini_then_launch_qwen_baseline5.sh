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

USED_EN_GEMINI_DIR="${USED_EN_GEMINI_DIR:-./results/gemini3flash_baseline5_snapshot_used_en_20260417}"
USED_APPS_GEMINI_DIR="${USED_APPS_GEMINI_DIR:-./results/gemini3flash_baseline5_snapshot_usedApps_en_20260417}"
WATCH_LOG_DIR="${WATCH_LOG_DIR:-$ROOT_DIR/run_log/watchers}"
WATCH_LOG_FILE="${WATCH_LOG_FILE:-$WATCH_LOG_DIR/watch_gemini_then_launch_qwen_baseline5.log}"
QWEN_RESULT_ROOT="${QWEN_RESULT_ROOT:-$ROOT_DIR/results/qwen3_baseline5}"
QWEN_RUN_NAME="${QWEN_RUN_NAME:-qwen3_baseline5}"
QWEN_AGENT_NAME="${QWEN_AGENT_NAME:-qwen3-vl-235b-a22b-thinking}"
MAX_STEPS="${MAX_STEPS:-20}"
mkdir -p "$WATCH_LOG_DIR" "$QWEN_RESULT_ROOT"

ts() {
  date '+%F_%T'
}

check_done() {
  local base_save_dir="$1"
  local task_dir="$2"
  python utils/completion_checker.py \
    --base_save_dir "$base_save_dir" \
    --paths_to_eval_tasks "$task_dir" \
    --languages task_en_env_en
}

gemini_processes_done() {
  if pgrep -f "base_save_dir $USED_EN_GEMINI_DIR" >/dev/null 2>&1; then
    return 1
  fi
  if pgrep -f "base_save_dir $USED_APPS_GEMINI_DIR" >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

{
  echo "[$(ts)] watcher started"
  echo "[$(ts)] waiting for Gemini baseline5 completion"

  while true; do
    en_done="$(check_done "$USED_EN_GEMINI_DIR" ./tasks/baseline5_snapshot_used_en || true)"
    apps_done="$(check_done "$USED_APPS_GEMINI_DIR" ./tasks/baseline5_snapshot_usedApps_en || true)"
    echo "[$(ts)] completion status used_en=$en_done usedApps_en=$apps_done"

    if [[ "$en_done" == "True" && "$apps_done" == "True" ]]; then
      break
    fi
    sleep 120
  done

  echo "[$(ts)] Gemini result files are complete; waiting for Gemini processes to exit"
  while ! gemini_processes_done; do
    sleep 30
  done

  echo "[$(ts)] launching Qwen baseline5 on existing instances"
  GUI_AGENT_NAME="$QWEN_AGENT_NAME" \
  RESULT_ROOT="$QWEN_RESULT_ROOT" \
  RUN_NAME="$QWEN_RUN_NAME" \
  MAX_STEPS="$MAX_STEPS" \
  bash "$ROOT_DIR/scripts/run_baseline5_dual_existing_instances.sh"

  echo "[$(ts)] Qwen baseline5 launch command completed"
} >> "$WATCH_LOG_FILE" 2>&1
