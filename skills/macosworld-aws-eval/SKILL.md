---
name: macosworld-aws-eval
description: Configure, run, debug, and cost-control macOSWorld evaluation on AWS EC2 Mac when the user mentions macOSWorld, macosworld, AWS smoke tests, Gemini Agent, EC2 Mac, Dedicated Host, ReplaceRootVolumeTask, 云测评, 或 wants reusable runbooks for this repository.
argument-hint: [repo-path-or-run-goal]
allowed-tools: Bash(*), Read, Write, Edit, Grep, Glob
---

# macOSWorld AWS Eval

Use this skill for the repository at `/Users/zhangkangning/code_repos/macosworld` when the task is about AWS-based evaluation, smoke tests, Gemini runs, EC2 Mac resources, or benchmark debugging.

## What This Skill Covers

- AWS account, Region, quota, and SCP readiness for EC2 Mac
- direct-boot launch patterns for `mac2.metal`
- Gemini 3 Flash gateway configuration for this machine
- the patched reset strategy in `utils/run_task.py`
- monitoring and interpreting smoke/full benchmark progress
- common macOSWorld-specific failure modes and their fixes

For exact commands, read `references/runbook.md`.

## Core Repo Facts

- Controller runs locally; the AWS Mac instance is only the remote GUI environment.
- Model API calls happen locally, not on the EC2 Mac instance.
- Main entrypoints:
  - `run.py`
  - `testbench.py`
  - `utils/run_task.py`
  - `utils/VNCClient.py`
  - `agent/gemini.py`
  - `constants.py`

## Agent Architecture Map

- `gemini-*`
  - routed by `agent/get_gui_agent.py`
  - implemented in `agent/gemini.py`
  - current preferred path for this machine
- `openai-*`
  - implemented in `agent/openai.py`
- `openai_omniparser-*`
  - implemented in `agent/openai_omniparser.py`
- `openai_cua-*`
  - implemented in `agent/openai_cua.py`
- `claude*`
  - implemented in `agent/anthropic.py`
- `showui*`
  - implemented in `agent/showui.py`
- `uitars*`
  - implemented in `agent/uitars.py`

For evaluation planning:

- the repo is not screenshot-only
- the common pattern is:
  - task text
  - one or more screenshots
  - local model call
  - parsed action list
  - VNC execution on the remote Mac

## Current Known-Good Baseline

- Preferred AWS Region: `ap-southeast-1`
- Validated path: English benchmark / smoke tests
- Benchmark AMIs come from `constants.py`
- `us-east-1` is not the default recommendation for this repo because the hardcoded benchmark AMIs are Region-specific and were validated in `ap-southeast-1`

For the current local workflow, baseline and test runs are fixed to five English domains only:

- `file_management`
- `media`
- `productivity`
- `sys_and_interface`
- `sys_apps`

Execution packaging is fixed by snapshot, not by reporting domain:

- `tasks/baseline5_snapshot_used_en`
  - `112` tasks
- `tasks/baseline5_snapshot_usedApps_en`
  - `31` tasks

Total fixed scope: `143` tasks.

Critical reporting rule:

- run scheduling and host allocation should use the two snapshot-based packages above
- final metrics must still be aggregated and reported by the five original domains
- do not collapse reporting into only `snapshot_used_en` vs `snapshot_usedApps_en`
- `productivity` spans both packages, so its results must be merged across the two runs before computing domain-level scores

## AWS Constraints That Matter

- EC2 Mac requires `Dedicated Host`
- `mac2.metal` means one Mac instance per Dedicated Host
- Dedicated Host has a 24-hour minimum allocation period
- IAM `AdministratorAccess` is not enough if Organizations SCP denies `ec2:AllocateHosts`
- The required quota is Region-specific:
  - `Running Dedicated mac2 Hosts`

## Current Reset Strategy

Do not default back to `ReplaceRootVolumeTask`.

Use the patched behavior in `utils/run_task.py`:

- `MACOSWORLD_AWS_RESET_MODE=smart`
- `MACOSWORLD_AWS_HARD_RESET_STRATEGY=relaunch`

Meaning:

- same-snapshot tasks:
  - use in-guest soft reset
- cross-snapshot tasks:
  - terminate the current instance
  - wait for the same host to become `available`
  - relaunch a new instance from the target AMI on that host

This exists because direct boot from the benchmark AMI worked, while `ReplaceRootVolumeTask` was observed to drive the instance into `impaired` / SSH-loss states on this setup.

## Important Local Patches To Preserve

- `utils/VNCClient.py`
  - `check_ssh_connectivity()` must check pure SSH, not tunnel-to-5900
  - includes a `paramiko.DSSKey` compatibility shim for `paramiko 4.x + sshtunnel 0.4.0`
- `utils/run_task.py`
  - split timeout budgets for reset, health, and SSH
  - persist bootstrap-instance to active-instance mapping in `run_log/*.json`
  - use `smart` soft-reset / relaunch-reset behavior
- `testbench.py`
  - sort tasks by snapshot to reduce cross-snapshot resets

If the user asks to debug regressions, verify these patches still exist before chasing AWS issues.

## Gemini Notes

- For this repo, `gemini-*` maps to `agent/gemini.py`
- Gemini step semantics are:
  - one screenshot
  - one model call
  - model may return multiple actions
  - all returned actions are executed before the next screenshot
- Gateway settings should come from environment variables, not hardcoded secrets
- If needed, recover the local gateway shape from `/Users/zhangkangning/code_repos/OSWorld/test_gemini.py`, but do not copy secrets into the skill
- Keep prompt and parser aligned with `execute_actions()` and `utils/VNCClient.py`; if they drift, fix parser first and then update the prompt to match

## Recommended Workflow

1. Verify AWS credentials, Region, quota, and SCP state.
2. Reuse an existing healthy host/instance if it already matches the target snapshot.
3. For the fixed five-domain benchmark subset, schedule runs using the two snapshot packages, not mixed-domain batches.
4. Prefer direct-boot instances from the target benchmark AMI.
5. For smoke runs, isolate:
   - result directory
   - runtime-state JSON file
6. Use `20` as the default `max_steps` budget for future evaluations unless there is an explicit experiment override.
7. Start with English-only smoke before larger runs.
8. Monitor:
   - `run.py` / `testbench.py` output
   - `eval_result.txt`
   - `context/step_*`
   - EC2 instance health
9. After execution, merge results back to the five reporting domains before computing pass rates or final tables.

## Result Interpretation

- `eval_result.txt = 100`: pass
- `eval_result.txt = 0`: benchmark ran, but task failed semantically
- `fail.flag`: framework/run-level failure
- multiple `step_*_raw_response.txt` files with no `fail.flag` means the agent loop itself worked
- multiple actions inside one `step_*_raw_response.txt` are normal; one outer step is one perception/decision round, not one atomic click

## Failure Patterns

- `explicit deny in a service control policy`
  - fix SCP; do not waste time changing IAM only
- `ReplaceRootVolumeTask` succeeds but SSH never returns
  - treat replace path as broken; use relaunch strategy
- SSH works but VNC fails with `paramiko.DSSKey`
  - local dependency compatibility issue; verify `utils/VNCClient.py` shim
- instance is `running` and EC2 health is `ok`, but launcher still loops on `waiting for SSH`
  - do not assume the AMI is broken first
  - first check the security group's ingress allowlist for port `22`
  - on this machine, the real cause was that the controller's public egress IP changed, but the security group still only allowed older `/32` entries
  - verify the controller IP with `curl -4 https://checkip.amazonaws.com`
  - compare it against `describe-security-groups` output for the benchmark SSH security group
  - if missing, add the current controller IP as a new `/32` ingress rule for TCP `22`
  - after updating the rule, retry raw SSH immediately before changing AMIs, hosts, or reset strategy
- host stuck `pending` after terminating an instance
  - usually wait; Mac host lifecycle is slower than normal EC2
- new direct-boot instance stays `initializing`
  - allow a several-minute window before declaring failure; compare with a known-good healthy instance if one exists

## SSH Triage Playbook

Use this exact order when EC2 Mac appears healthy but the benchmark is blocked on SSH:

1. Check instance state and EC2 health.
   - if `running` + `SystemStatus ok` + `InstanceStatus ok`, the problem may still be only network ingress
2. Test raw SSH from the controller.
   - use the exact PEM and DNS name
   - do not rely only on framework logs
3. Check the controller's current public IP.
   - example: `curl -4 -s https://checkip.amazonaws.com`
4. Inspect the benchmark security group ingress rules for port `22`.
   - if the controller IP is not listed, fix that before deeper debugging
5. Only after ingress is confirmed should you investigate:
   - AMI boot behavior
   - console output
   - launch template / subnet / NACL anomalies

Known-good diagnosis from April 17, 2026:

- instances:
  - `running`
  - `SystemStatus ok`
  - `InstanceStatus ok`
- symptom:
  - launcher logs kept printing `waiting for SSH`
- root cause:
  - security group `sg-0324bde9759ab4283` allowed only stale controller IPs
  - current controller IP was `103.14.247.231`
- fix:
  - authorize ingress `tcp/22` for `103.14.247.231/32`
- result:
  - both Mac instances became reachable over SSH immediately
  - launcher advanced into `run_aws_eval.sh`, then `run.py`, then `testbench.py`

## When To Read The Reference

Read `references/runbook.md` when you need:

- the exact smoke launch command
- AWS CLI inspection commands
- cleanup commands
- a concise monitor checklist
