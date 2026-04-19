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
export AWS_PROFILE="${AWS_PROFILE:-macosworld-eval}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-southeast-1}"

HOST_EN="${HOST_EN:-h-04e49661c1901b945}"
HOST_APPS="${HOST_APPS:-h-0b29a903a5c49d362}"
AMI_EN="${AMI_EN:-ami-0132f892c5d80f6ba}"
AMI_APPS="${AMI_APPS:-ami-07f4fd69378358c18}"
SUBNET="${SUBNET:-subnet-0523d71491e819dbb}"
KEY_NAME="${KEY_NAME:-macosworld-gemini-smoke-20260403-key}"
SG_ID="${SG_ID:-sg-0324bde9759ab4283}"
PEM="${PEM:-/Users/zhangkangning/.ssh/macosworld-gemini-smoke-20260403-key.pem}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/run_log/fullbench_qwen}"
mkdir -p "$LOG_DIR"

ts() {
  date '+%F_%T'
}

wait_host_available() {
  local host_id="$1"
  while true; do
    local state
    state="$(aws ec2 describe-hosts --host-ids "$host_id" --query 'Hosts[0].State' --output text)"
    echo "[$(ts)] host $host_id state=$state"
    if [ "$state" = "available" ]; then
      return 0
    fi
    sleep 20
  done
}

wait_instance_running() {
  local instance_id="$1"
  while true; do
    local state
    state="$(aws ec2 describe-instances --instance-ids "$instance_id" --query 'Reservations[0].Instances[0].State.Name' --output text)"
    echo "[$(ts)] instance $instance_id state=$state"
    if [ "$state" = "running" ]; then
      return 0
    fi
    sleep 10
  done
}

wait_ssh_ready() {
  local host="$1"
  while ! ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 -i "$PEM" "ec2-user@$host" exit 0 >/dev/null 2>&1; do
    echo "[$(ts)] waiting for SSH on $host"
    sleep 15
  done
}

launch_worker() {
  local label="$1"
  local host_id="$2"
  local image_id="$3"
  local initial_snapshot="$4"
  local task_dir="$5"
  local result_dir="$6"
  local state_file="$7"
  local tag_name="$8"
  local log_file="$9"
  local runtime_file="${10}"

  {
    echo "[$(ts)] worker=$label waiting for host $host_id"
    wait_host_available "$host_id"

    local instance_id
    instance_id="$(aws ec2 run-instances \
      --image-id "$image_id" \
      --instance-type mac2.metal \
      --key-name "$KEY_NAME" \
      --security-group-ids "$SG_ID" \
      --subnet-id "$SUBNET" \
      --placement "Tenancy=host,HostId=$host_id,AvailabilityZone=ap-southeast-1a" \
      --count 1 \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$tag_name}]" \
      --query 'Instances[0].InstanceId' \
      --output text)"

    echo "[$(ts)] worker=$label launched instance=$instance_id"
    wait_instance_running "$instance_id"

    local dns_name
    dns_name="$(aws ec2 describe-instances --instance-ids "$instance_id" --query 'Reservations[0].Instances[0].PublicDnsName' --output text)"
    echo "[$(ts)] worker=$label dns=$dns_name"
    wait_ssh_ready "$dns_name"
    echo "[$(ts)] worker=$label ssh_ready"

    cat > "$runtime_file" <<EOF
label=$label
host_id=$host_id
instance_id=$instance_id
dns_name=$dns_name
image_id=$image_id
initial_snapshot=$initial_snapshot
task_dir=$task_dir
result_dir=$result_dir
state_file=$state_file
EOF

    (
      export MACOSWORLD_INSTANCE_ID="$instance_id"
      export MACOSWORLD_SSH_HOST="$dns_name"
      export MACOSWORLD_SSH_PKEY="$PEM"
      export MACOSWORLD_PATHS_TO_EVAL_TASKS="$task_dir"
      export MACOSWORLD_BASE_SAVE_DIR="$result_dir"
      export MACOSWORLD_AWS_INSTANCE_STATE_FILE="$state_file"
      export MACOSWORLD_AWS_INITIAL_SNAPSHOT_NAME="$initial_snapshot"
      exec bash "$ROOT_DIR/scripts/run_aws_eval.sh"
    )
  } > "$log_file" 2>&1
}

echo "[$(ts)] snapshot-aware dual launcher started"

launch_worker \
  "used_en" \
  "$HOST_EN" \
  "$AMI_EN" \
  "snapshot_used_en" \
  "./tasks/full_en_snapshot_used_en" \
  "./results/qwen3vl_full202_snapshot_used_en_20260407" \
  "./run_log/qwen3vl_full202_snapshot_used_en_20260407_instance_state.json" \
  "qwen3vl-full202-used-en-20260407" \
  "$LOG_DIR/qwen3vl_full202_snapshot_used_en_20260407.log" \
  "$LOG_DIR/qwen3vl_full202_snapshot_used_en_20260407.runtime.env" &
PID_EN=$!

launch_worker \
  "usedApps_en" \
  "$HOST_APPS" \
  "$AMI_APPS" \
  "snapshot_usedApps_en" \
  "./tasks/full_en_snapshot_usedApps_en" \
  "./results/qwen3vl_full202_snapshot_usedApps_en_20260407" \
  "./run_log/qwen3vl_full202_snapshot_usedApps_en_20260407_instance_state.json" \
  "qwen3vl-full202-usedApps-en-20260407" \
  "$LOG_DIR/qwen3vl_full202_snapshot_usedApps_en_20260407.log" \
  "$LOG_DIR/qwen3vl_full202_snapshot_usedApps_en_20260407.runtime.env" &
PID_APPS=$!

echo "[$(ts)] worker supervisor pids: used_en=$PID_EN usedApps_en=$PID_APPS"

while true; do
  echo "[$(ts)] heartbeat"
  ps -p "$PID_EN" -o pid=,etime=,command= || true
  ps -p "$PID_APPS" -o pid=,etime=,command= || true
  sleep 60
done
