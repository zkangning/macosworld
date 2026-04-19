# AWS Quickstart for macOSWorld

This note is the shortest path from a clean local checkout to a runnable AWS benchmark job.

## 1. What you need before starting

Collect these items first:

- A local `macosworld` conda environment created from `environment.yml`
- AWS account access keys available on the machine that runs the benchmark
- AWS region `ap-southeast-1`
- EC2 quota `Running Dedicated mac2 Hosts >= 1` in `ap-southeast-1`
- One allocated Dedicated Host in `ap-southeast-1a`
- One launched `mac2.metal` instance attached to that Dedicated Host
- The instance public DNS name
- The instance ID
- The SSH key pair `.pem` file used when launching the instance
- A security group that allows inbound SSH from the benchmark machine
- One GUI agent choice and its API credentials

Important:

- For benchmark execution, inbound SSH is required. Direct VNC exposure is optional because the testbench tunnels VNC over SSH.
- After launching the macOS instance, wait around 10 minutes before running the benchmark.
- Do not shut down the instance unless you are done. Shutdown or termination triggers cleanup delays on Mac hosts.

## 2. Fastest first smoke test

Use a single category and a single language first. This validates AWS connectivity, root-volume replacement, SSH tunneling, and agent credentials without burning through the full benchmark.

```bash
conda env create -f environment.yml
conda activate macosworld

export AWS_ACCESS_KEY_ID='...'
export AWS_SECRET_ACCESS_KEY='...'
export AWS_DEFAULT_REGION='ap-southeast-1'
export OPENAI_API_KEY='...'

chmod 400 /absolute/path/to/credential.pem

python run.py \
    --instance_id i-xxxxxxxxxxxxxxxxx \
    --ssh_host ec2-xx-xx-xx-xx.ap-southeast-1.compute.amazonaws.com \
    --ssh_pkey /absolute/path/to/credential.pem \
    --gui_agent_name gpt-4o-2024-08-06 \
    --paths_to_eval_tasks ./tasks/sys_apps \
    --languages task_en_env_en \
    --base_save_dir ./results/smoke_gpt4o \
    --max-steps 20 \
    --snapshot_recovery_timeout_seconds 1200 \
    --task_step_timeout 120
```

If this succeeds, scale up to more categories and languages.

## 3. Full benchmark command shape

```bash
python run.py \
    --instance_id i-xxxxxxxxxxxxxxxxx \
    --ssh_host ec2-xx-xx-xx-xx.ap-southeast-1.compute.amazonaws.com \
    --ssh_pkey /absolute/path/to/credential.pem \
    --gui_agent_name gpt-4o-2024-08-06 \
    --paths_to_eval_tasks ./tasks/sys_apps ./tasks/sys_and_interface ./tasks/productivity ./tasks/media ./tasks/file_management ./tasks/advanced ./tasks/multi_apps \
    --languages task_en_env_en task_zh_env_zh task_ar_env_ar task_ja_env_ja task_ru_env_ru \
    --base_save_dir ./results/gpt_4o \
    --max-steps 20 \
    --snapshot_recovery_timeout_seconds 1200 \
    --task_step_timeout 120
```

Notes:

- The safety subset should be run separately with `./tasks/safety` and `task_en_env_en`.
- `run.py` wraps `testbench.py`, restarts long-running sessions, and cleans interrupted task folders automatically.
- During each task, the testbench resets the machine by calling AWS `create_replace_root_volume_task(...)` with the macOSWorld AMI IDs defined in `constants.py`.

## 4. Minimal information to provide when asking for help

If you want someone else to get you to a runnable benchmark quickly, provide:

- Which GUI agent you want to evaluate
- Which API provider you want to use: OpenAI, Anthropic, Gemini, ShowUI, or UI-TARS
- Which task scope you want first: smoke test, full benchmark, or safety only
- Your AWS region
- Your Dedicated Host ID
- Your EC2 instance ID
- Your instance public DNS
- The absolute path to your `.pem` file
- Whether your security group already allows SSH from your current IP
- Where you want results written locally
- For Gemini, the path to `GOOGLE_APPLICATION_CREDENTIALS`
- Whether you want only environment setup, or also an end-to-end trial run

## 5. Cost and cleanup constraints

- A Dedicated Host starts billing when allocated.
- A Dedicated Host can only be released 24 hours after allocation.
- After terminating a Mac instance, you typically need to wait about 20 minutes before releasing the host.
- After termination or shutdown, launching a new Mac instance on that host can be blocked for roughly 1 hour by AWS cleanup.
