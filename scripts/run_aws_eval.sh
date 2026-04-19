#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${MACOSWORLD_NO_AUTO_ENV:-0}" != "1" && -f "$repo_root/.env" ]]; then
  set -a
  source "$repo_root/.env"
  set +a
fi

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

require_var "MACOSWORLD_INSTANCE_ID"
require_var "MACOSWORLD_SSH_HOST"
require_var "MACOSWORLD_SSH_PKEY"
require_var "MACOSWORLD_GUI_AGENT_NAME"
require_var "MACOSWORLD_PATHS_TO_EVAL_TASKS"
require_var "MACOSWORLD_LANGUAGES"
require_var "MACOSWORLD_BASE_SAVE_DIR"

region="${AWS_DEFAULT_REGION:-${AWS_REGION:-ap-southeast-1}}"
export AWS_DEFAULT_REGION="$region"

if [[ ! -f "$MACOSWORLD_SSH_PKEY" ]]; then
  echo "SSH key not found: $MACOSWORLD_SSH_PKEY" >&2
  exit 1
fi

# Best-effort permission hardening. On this machine, parallel workers may touch the
# same PEM path concurrently, so do not fail the whole run if chmod is interrupted.
if [[ "$(stat -f '%Lp' "$MACOSWORLD_SSH_PKEY" 2>/dev/null || echo '')" != "400" ]]; then
  chmod 400 "$MACOSWORLD_SSH_PKEY" 2>/dev/null || true
fi
read -r -a task_dirs <<< "$MACOSWORLD_PATHS_TO_EVAL_TASKS"
read -r -a languages <<< "$MACOSWORLD_LANGUAGES"

conda_env="${MACOSWORLD_CONDA_ENV:-macosworld}"
python_cmd=(python)
if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "$conda_env" ]]; then
  python_cmd=(conda run -n "$conda_env" python)
fi

exec "${python_cmd[@]}" run.py \
  --instance_id "$MACOSWORLD_INSTANCE_ID" \
  --ssh_host "$MACOSWORLD_SSH_HOST" \
  --ssh_pkey "$MACOSWORLD_SSH_PKEY" \
  --gui_agent_name "$MACOSWORLD_GUI_AGENT_NAME" \
  --paths_to_eval_tasks "${task_dirs[@]}" \
  --languages "${languages[@]}" \
  --base_save_dir "$MACOSWORLD_BASE_SAVE_DIR" \
  --max-steps "${MACOSWORLD_MAX_STEPS:-20}" \
  --snapshot_recovery_timeout_seconds "${MACOSWORLD_SNAPSHOT_RECOVERY_TIMEOUT_SECONDS:-1200}" \
  --task_step_timeout "${MACOSWORLD_TASK_STEP_TIMEOUT:-120}"
