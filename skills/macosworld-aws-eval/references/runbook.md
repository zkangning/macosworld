# macOSWorld AWS Runbook

This file contains the detailed commands and operator checklist for the repo at `/Users/zhangkangning/code_repos/macosworld`.

## 1. Quick Environment Check

```bash
cd /Users/zhangkangning/code_repos/macosworld
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate macosworld

AWS_PROFILE=macosworld-eval AWS_DEFAULT_REGION=ap-southeast-1 aws sts get-caller-identity
AWS_PROFILE=macosworld-eval AWS_DEFAULT_REGION=ap-southeast-1 \
  aws ec2 describe-hosts --filter Name=instance-type,Values=mac2.metal
AWS_PROFILE=macosworld-eval AWS_DEFAULT_REGION=ap-southeast-1 \
  aws ec2 describe-instances --filters Name=instance-type,Values=mac2.metal
```

## 2. Recommended Runtime Environment

```bash
export AWS_PROFILE=macosworld-eval
export AWS_DEFAULT_REGION=ap-southeast-1

export GEMINI_BASE_URL='https://runway.devops.rednote.life/openai/google'
export GEMINI_API_KEY='set outside the skill'
export GEMINI_THINKING_LEVEL='NONE'
export GEMINI_INCLUDE_THOUGHTS='false'

export MACOSWORLD_AWS_RESET_MODE='smart'
export MACOSWORLD_AWS_HARD_RESET_STRATEGY='relaunch'
export MACOSWORLD_AWS_INITIAL_SNAPSHOT_NAME='snapshot_used_en'
export MACOSWORLD_AWS_INSTANCE_STATE_FILE='./run_log/<run_name>_instance_state.json'
```

Notes:

- Keep the API key outside the skill and repo.
- For English-only smoke runs, `snapshot_used_en` is the right initial snapshot.

## 3. Recommended Smoke Launch

```bash
python -u run.py \
  --instance_id <bootstrap_instance_id> \
  --ssh_host <bootstrap_public_dns> \
  --ssh_pkey /Users/zhangkangning/.ssh/macosworld-gemini-smoke-20260403-key.pem \
  --gui_agent_name gemini-3-flash \
  --paths_to_eval_tasks ./tasks/smoke_gemini4_en \
  --languages task_en_env_en \
  --base_save_dir ./results/<run_name> \
  --max-steps 15 \
  --snapshot_recovery_timeout_seconds 1500 \
  --task_max_attempts 1
```

## 4. What Healthy Progress Looks Like

In logs:

- `Using soft reset`
- `SSH connectivity ready`
- `Connected to ...`
- `Capturing screenshot`
- `Calling GUI agent`
- `Actuating`
- `Evaluation result`

On disk:

- `context/step_001.png`
- `context/step_001_raw_response.txt`
- `context/step_001_parsed_actions.json`
- `eval_result.txt`

## 5. How To Monitor A Live Run

```bash
find ./results/<run_name> -name eval_result.txt -print | sort | xargs -I {} sh -c 'printf "%s: " "$1"; cat "$1"; echo' _ {}

find ./results/<run_name> -name fail.flag -print | sort

find ./results/<run_name> -path '*/context/step_*_raw_response.txt' | wc -l

AWS_PROFILE=macosworld-eval AWS_DEFAULT_REGION=ap-southeast-1 \
  aws ec2 describe-instance-status --include-all-instances --instance-ids <instance_id>
```

## 6. If AWS Mac Permissions Fail

Typical error:

- `explicit deny in a service control policy`

Then:

- do not keep modifying IAM users
- check Organizations SCP for `ec2:AllocateHosts`
- required capability is Region-specific and applies before instance launch

## 7. If ReplaceRootVolumeTask Is Mentioned

Current guidance:

- avoid using it as the default reset mechanism
- prefer the patched relaunch flow in `utils/run_task.py`

Reason:

- direct boot from benchmark AMIs was observed to work
- replace-root-volume was observed to cause `impaired` instance states and SSH loss on this setup

## 8. If VNC Fails But SSH Works

Check local versions:

```bash
python - <<'PY'
import paramiko, sshtunnel
print('paramiko', getattr(paramiko, '__version__', 'unknown'))
print('sshtunnel', getattr(sshtunnel, '__version__', 'unknown'))
print('has DSSKey', hasattr(paramiko, 'DSSKey'))
PY
```

Expected risky combo:

- `paramiko 4.x`
- `sshtunnel 0.4.0`

In this repo, `utils/VNCClient.py` should contain the compatibility shim for `paramiko.DSSKey`.

## 9. Cost Control

- one `mac2.metal` instance per Dedicated Host
- Dedicated Host has a 24-hour minimum allocation period
- do not allocate extra hosts for casual diagnostics unless necessary
- prefer reusing a healthy host

## 10. Cleanup

Terminate unused instances:

```bash
AWS_PROFILE=macosworld-eval AWS_DEFAULT_REGION=ap-southeast-1 \
  aws ec2 terminate-instances --instance-ids <instance_id>
```

Release hosts only after the minimum allocation period has expired:

```bash
AWS_PROFILE=macosworld-eval AWS_DEFAULT_REGION=ap-southeast-1 \
  aws ec2 release-hosts --host-ids <host_id>
```
