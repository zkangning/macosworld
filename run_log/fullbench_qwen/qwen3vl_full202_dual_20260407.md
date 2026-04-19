Start time: 2026-04-07 21:17 Asia/Shanghai

Goal:
- English main benchmark only
- 202 total cases
- Split evenly across two EC2 Mac hosts

Split:
- `tasks/full_en_split_a`: 101 cases
- `tasks/full_en_split_b`: 101 cases

Instances:
- Split A
  - host: `h-04e49661c1901b945`
  - instance: `i-0fda36f080d9726ef`
  - ssh host: `ec2-13-212-81-180.ap-southeast-1.compute.amazonaws.com`
  - persistent exec session: `20820`
- Split B
  - host: `h-0b29a903a5c49d362`
  - instance: `i-02bd558b7d87d893a`
  - ssh host: `ec2-13-212-86-114.ap-southeast-1.compute.amazonaws.com`
  - persistent exec session: `48544`

Result roots:
- `./results/qwen3vl_full202_split_a_20260407`
- `./results/qwen3vl_full202_split_b_20260407`

Runtime state files:
- `./run_log/qwen3vl_full202_split_a_20260407_instance_state.json`
- `./run_log/qwen3vl_full202_split_b_20260407_instance_state.json`

Configuration:
- model: `qwen3-vl-235b-a22b-thinking`
- image window: `3`
- reset mode: `smart`
- hard reset strategy: `relaunch`
- language: `task_en_env_en`

Duration estimate:
- Observed healthy 4-case smoke runtime on this repo: about 10 minutes total
- That smoke is too optimistic for full benchmark because the 202-case run contains harder tasks and cross-snapshot relaunch overhead
- Practical estimate for this dual-host 202-case run:
  - optimistic wall clock: `5-6 hours`
  - safer planning estimate: `6-8 hours`

Current launch status at write time:
- Split A: launched and attached to a healthy running instance
- Split B: supervisor launched; waiting for instance SSH readiness before starting benchmark
