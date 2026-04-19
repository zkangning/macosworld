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

## AWS Dual-Host Baseline5 Run

Use the Gemini dual launcher:

```bash
bash scripts/run_gemini_baseline5_dual_by_snapshot.sh
```

This launcher:

- starts one `mac2.metal` instance on a host for `snapshot_used_en`
- starts one `mac2.metal` instance on a host for `snapshot_usedApps_en`
- runs the two fixed baseline5 packages in parallel
- keeps result directories, runtime env files, and logs separate
