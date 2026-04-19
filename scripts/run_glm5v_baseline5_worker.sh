#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MACOSWORLD_NO_AUTO_ENV=1
export PYTHONUNBUFFERED=1

RUNTIME_FILE="${RUNTIME_FILE:?RUNTIME_FILE is required}"
TARGET_RESULT_DIR="${TARGET_RESULT_DIR:?TARGET_RESULT_DIR is required}"
STATE_FILE="${STATE_FILE:?STATE_FILE is required}"
GLM_API_KEY="${GLM_API_KEY:?GLM_API_KEY is required}"
GLM_BASE_URL="${GLM_BASE_URL:-https://openrouter.ai/api/v1}"
GLM_THINKING_MODE="${GLM_THINKING_MODE:-on}"
GLM_PROXY_URL="${GLM_PROXY_URL:-}"
GUI_AGENT_NAME="${GUI_AGENT_NAME:-z-ai/glm-5v-turbo}"
MAX_STEPS="${MAX_STEPS:-20}"
OPENROUTER_APP_NAME="${OPENROUTER_APP_NAME:-macosworld-glm5v-baseline5}"
OPENROUTER_SITE_URL="${OPENROUTER_SITE_URL:-https://github.com/openai/codex}"

# shellcheck disable=SC1090
source "$RUNTIME_FILE"

export GLM_API_KEY
export GLM_BASE_URL
export GLM_THINKING_MODE
export GLM_PROXY_URL
export OPENROUTER_APP_NAME
export OPENROUTER_SITE_URL

export MACOSWORLD_INSTANCE_ID="$instance_id"
export MACOSWORLD_SSH_HOST="$dns_name"
export MACOSWORLD_SSH_PKEY="${MACOSWORLD_SSH_PKEY:-/Users/zhangkangning/.ssh/macosworld-gemini-smoke-20260403-key.pem}"
export MACOSWORLD_GUI_AGENT_NAME="$GUI_AGENT_NAME"
export MACOSWORLD_PATHS_TO_EVAL_TASKS="$task_dir"
export MACOSWORLD_LANGUAGES="task_en_env_en"
export MACOSWORLD_BASE_SAVE_DIR="$TARGET_RESULT_DIR"
export MACOSWORLD_AWS_INSTANCE_STATE_FILE="$STATE_FILE"
export MACOSWORLD_AWS_INITIAL_SNAPSHOT_NAME="$initial_snapshot"
export MACOSWORLD_MAX_STEPS="$MAX_STEPS"

exec bash "$ROOT_DIR/scripts/run_aws_eval.sh"
