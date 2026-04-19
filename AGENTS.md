# macOSWorld Agent Notes

## Active Baseline Default

- GUI agent: `gemini-3-flash`
- Default max steps: `20`
- Fixed evaluation scope: five English domains only
  - `file_management`
  - `media`
  - `productivity`
  - `sys_and_interface`
  - `sys_apps`
- Execution packages:
  - `./tasks/baseline5_snapshot_used_en`
  - `./tasks/baseline5_snapshot_usedApps_en`
- Reporting must still aggregate back to the five original domains.

## Runtime Architecture

- The controller runs locally in this repo.
- The AWS EC2 Mac instance is only the remote GUI environment.
- Model API calls happen locally, not on the EC2 Mac instance.
- Main entrypoints:
  - `run.py`
  - `testbench.py`
  - `utils/run_task.py`
  - `utils/VNCClient.py`
  - `agent/gemini.py`
  - `agent/openai.py`
  - `agent/qwen.py`
  - `agent/get_gui_agent.py`
  - `agent/gui_action_parser.py`
  - `constants.py`

The common loop is:

1. read task text
2. capture one or more screenshots
3. call the local model backend
4. parse actions locally
5. execute actions over VNC on the remote Mac
6. repeat until `done`, `fail`, timeout, or max steps

## Fixed Baseline5 Scheduling Rule

- Schedule runs by snapshot package, not by reporting domain.
- `tasks/baseline5_snapshot_used_en` contains `112` tasks.
- `tasks/baseline5_snapshot_usedApps_en` contains `31` tasks.
- Total fixed scope is `143` tasks.
- Final metrics must still be merged back to the five original domains.
- `productivity` spans both packages, so do not report only `snapshot_used_en` vs `snapshot_usedApps_en`.

## Gemini3Flash Gateway

The local Gemini agent uses the OpenAI-like gateway path implemented in [agent/gemini.py](/Users/zhangkangning/code_repos/macosworld/agent/gemini.py) when the following environment variables are set:

```bash
export GEMINI_BASE_URL='https://runway.devops.rednote.life/openai/google'
export GEMINI_API_KEY='e3003a57a815463696a2cceaecb7d941'
export GEMINI_THINKING_LEVEL='NONE'
export GEMINI_INCLUDE_THOUGHTS='false'
export GEMINI_SEED='0'
```

These values were aligned with `/Users/zhangkangning/code_repos/OSWorld/test_gemini.py`.

## GLM 5V OpenRouter Gateway

The GLM baseline path reuses the OpenAI-compatible architecture in [agent/gemini.py](/Users/zhangkangning/code_repos/macosworld/agent/gemini.py) and is routed from [agent/get_gui_agent.py](/Users/zhangkangning/code_repos/macosworld/agent/get_gui_agent.py).

Recommended environment:

```bash
export GLM_BASE_URL='https://openrouter.ai/api/v1'
export GLM_API_KEY='<set outside repo>'
export GLM_THINKING_MODE='on'
export OPENROUTER_APP_NAME='macosworld-glm5v-baseline5'
export OPENROUTER_SITE_URL='https://github.com/openai/codex'
export GUI_AGENT_NAME='z-ai/glm-5v-turbo'
```

Notes:

- `glm` currently shares the common `GEMINI_SYSTEM_PROMPT`; do not maintain a GLM-only prompt branch unless a provider constraint forces it.
- `GLM_THINKING_MODE='on'` should route through the OpenRouter reasoning flag path in [agent/gemini.py](/Users/zhangkangning/code_repos/macosworld/agent/gemini.py).
- Keep keys outside the repo even if local experiments happen from one workstation.

## AWS Dual-Host Baseline5 Run

Use the Gemini dual launcher:

```bash
bash scripts/run_gemini_baseline5_dual_by_snapshot.sh
```

For GLM 5V on existing instances, use:

```bash
bash scripts/run_glm5v_baseline5_dual_existing_instances.sh
```

This class of launcher:

- starts one `mac2.metal` instance on a host for `snapshot_used_en`
- starts one `mac2.metal` instance on a host for `snapshot_usedApps_en`
- runs the two fixed baseline5 packages in parallel
- keeps result directories, runtime env files, and logs separate

## AWS Constraints That Matter

- Preferred region: `ap-southeast-1`
- EC2 Mac requires `Dedicated Host`
- `mac2.metal` means one Mac instance per Dedicated Host
- Dedicated Host has a 24-hour minimum allocation period
- IAM `AdministratorAccess` is not enough if Organizations SCP denies `ec2:AllocateHosts`
- Required quota is region-specific:
  - `Running Dedicated mac2 Hosts`
- Benchmark AMIs are region-specific and validated in `ap-southeast-1`

## Current Reset Strategy

Do not default back to `ReplaceRootVolumeTask`.

Use the patched behavior in `utils/run_task.py`:

```bash
export MACOSWORLD_AWS_RESET_MODE='smart'
export MACOSWORLD_AWS_HARD_RESET_STRATEGY='relaunch'
```

Meaning:

- same-snapshot tasks use in-guest soft reset
- cross-snapshot tasks terminate the current instance, wait for the same host to become `available`, then relaunch a new instance from the target AMI on that host

This exists because direct boot from the benchmark AMI worked, while replace-root-volume was observed to push this setup into `impaired` or SSH-loss states.

## Important Local Patches To Preserve

- [utils/VNCClient.py](/Users/zhangkangning/code_repos/macosworld/utils/VNCClient.py)
  - `check_ssh_connectivity()` must check pure SSH, not tunnel-to-5900
  - includes the `paramiko.DSSKey` compatibility shim for `paramiko 4.x + sshtunnel 0.4.0`
- [utils/run_task.py](/Users/zhangkangning/code_repos/macosworld/utils/run_task.py)
  - split timeout budgets for reset, health, and SSH
  - persist bootstrap-instance to active-instance mapping in `run_log/*.json`
  - use `smart` soft-reset and relaunch-reset behavior
- [testbench.py](/Users/zhangkangning/code_repos/macosworld/testbench.py)
  - sort tasks by snapshot to reduce cross-snapshot resets
- [agent/gui_action_parser.py](/Users/zhangkangning/code_repos/macosworld/agent/gui_action_parser.py)
  - shared parser for `gemini`, `openai`, and `qwen`
  - keep prompt rules and parser behavior aligned

If a later regression appears, verify these patches still exist before chasing AWS or model-provider issues.

## General Agent Output Contract

The current implementation intentionally makes the output contract general rather than GLM-specific.

Shared rules live in [agent/gui_action_parser.py](/Users/zhangkangning/code_repos/macosworld/agent/gui_action_parser.py) and are injected into:

- [agent/gemini.py](/Users/zhangkangning/code_repos/macosworld/agent/gemini.py)
- [agent/openai.py](/Users/zhangkangning/code_repos/macosworld/agent/openai.py)
- [agent/qwen.py](/Users/zhangkangning/code_repos/macosworld/agent/qwen.py)

Important behavior:

- the model should emit one plaintext code block only
- no JSON, XML, HTML, function-call wrappers, or narration
- `left_click x y` and `double_click x y` are legal shorthand forms
- shorthand click forms are normalized into `move_to x y` plus the click action
- key aliases are normalized:
  - `Return -> enter`
  - `Escape -> esc`
  - `delete -> del`
  - `control -> ctrl`
- XML-style tags such as `<action_name>` and `<parameter_1>` are stripped and recovered where possible
- pipes, backticks, and mixed formatting wrappers are normalized away

Lock-screen rule:

- if the screenshot shows the macOS lock screen and a password field, the agent should unlock with `000000` and `enter`
- once the desktop or target app is visible, do not keep typing the password

## What Failed In Practice Before The Parser Fix

The first GLM baseline run exposed several recurring failure modes:

- click actions emitted as `left_click x y` or `double_click x y`
  - older parsing treated those as bare clicks and ignored the coordinates
- XML-ish wrappers such as `<action_name>` or `<function_calls>`
- narration mixed with actions
- repeated lock-screen password typing after the desktop was already visible
- premature `done`

The important lesson is that these are not GLM-only problems. The parser and prompt should stay general across providers.

## Monitoring And Result Interpretation

On disk:

- `eval_result.txt = 100`: pass
- `eval_result.txt = 0`: benchmark ran, but task failed semantically
- `fail.flag`: framework or run-level failure
- `context/step_*_raw_response.txt`: raw model output per perception round
- `context/step_*_parsed_actions.json`: parsed actions per round

Useful signals:

- multiple `step_*_raw_response.txt` files with no `fail.flag` means the agent loop itself worked
- multiple actions inside one response are normal
- one outer step is one perception-decision round, not one atomic click

Recommended monitor checklist:

1. inspect local `tmux` sessions
2. inspect `run.py` and `testbench.py` processes
3. count `eval_result.txt` and `fail.flag`
4. inspect `context/step_*`
5. inspect EC2 instance state and status checks
6. inspect the latest result-file timestamps before deciding a run is still active

## SSH Triage Playbook

Use this order when the EC2 Mac appears healthy but the benchmark is blocked on SSH:

1. Check instance state and EC2 health.
   - if `running` and both health checks are `ok`, the problem may still be network ingress
2. Test raw SSH from the controller.
3. Check the controller's current public IP.
   - for example: `curl -4 -s https://checkip.amazonaws.com`
4. Inspect the benchmark security group's port `22` ingress rules.
5. Only after ingress is confirmed should you investigate AMI boot behavior, console output, or subnet and NACL issues.

Known-good diagnosis from 2026-04-17:

- instances were `running`
- `SystemStatus` and `InstanceStatus` were `ok`
- launcher kept printing `waiting for SSH`
- root cause was stale `/32` ingress entries on `sg-0324bde9759ab4283`
- after authorizing the controller's current public IP on `tcp/22`, SSH became reachable immediately

## Stop And Cleanup Guidance

To stop an experiment safely:

1. kill the local `tmux` sessions
2. kill local `run.py` and `testbench.py` controllers if they are still alive
3. only terminate EC2 Mac instances if you also want to destroy the remote environment

Important distinction:

- stopping the local experiment does not require terminating the two Mac instances
- terminating an EC2 Mac instance can leave its Dedicated Host in `pending` for a while
- do not release hosts until the 24-hour minimum allocation window has expired

## Empirical Status Notes From The April 18-19, 2026 GLM Work

### Full GLM baseline attempt

Run root:

- [results/glm5v_baseline5](/Users/zhangkangning/code_repos/macosworld/results/glm5v_baseline5)
- [run_log/glm5v_baseline5_20260418_2220](/Users/zhangkangning/code_repos/macosworld/run_log/glm5v_baseline5_20260418_2220)

Observed state before the run was stopped:

- `28` tasks started
- `26` tasks reached a terminal artifact
- `4` pass
- `21` semantic fail
- `1` framework fail
- `2` task directories were interrupted without final artifact

This run is useful as a failure corpus, but not as a finished headline result.

### GLM smoke retest after parser and prompt generalization

Run root:

- [results/glm5v_smoke_retest_20260418_235823](/Users/zhangkangning/code_repos/macosworld/results/glm5v_smoke_retest_20260418_235823)
- [run_log/glm5v_smoke_retest_20260418_235823](/Users/zhangkangning/code_repos/macosworld/run_log/glm5v_smoke_retest_20260418_235823)

Selected cases:

- pass:
  - `03dd4300-a9e4-8b44-1339-6333dd82066d`
  - `09a146c8-d1cf-dbdb-54ce-b60ede93cdab`
- semantic fail:
  - `0156a050-0e42-591d-7dda-e64caa2f8ae6`
  - `05eb3ac3-9c69-8be8-a5d7-06c150839d29`

Interpretation:

- the parser and prompt fixes clearly improved legality and executability of model outputs
- the remaining failures in this smoke are now semantic task-solving failures rather than obvious output-format failures

## Project-Local Skill Copy

The project-local copy of the AWS evaluation skill should live under:

- `./skills/macosworld-aws-eval/SKILL.md`
- `./skills/macosworld-aws-eval/references/runbook.md`
- `./skills/macosworld-aws-eval/agents/openai.yaml`

When the external Codex skill and the project-local copy diverge, update both intentionally rather than letting them drift silently.
