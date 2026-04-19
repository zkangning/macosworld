import os
import json
import boto3
import time

from utils.VNCClient import VNCClient_SSH
from utils.evaluator import Evaluator
from utils.async_utils import AsyncSSHCommandHandler

from utils.log import print_message
from utils.vmware_utils import VMwareTools

from agent.get_gui_agent import get_gui_agent

from constants import ami_lookup_table


aws_instance_snapshot_state = {}
aws_instance_runtime_state = {}


def _parse_truthy_env_var(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_aws_reset_mode() -> str:
    mode = os.getenv("MACOSWORLD_AWS_RESET_MODE", "smart").strip().lower()
    if mode not in {"smart", "soft", "hard"}:
        raise ValueError(f'Illegal MACOSWORLD_AWS_RESET_MODE "{mode}"')
    return mode


def _get_aws_hard_reset_strategy() -> str:
    strategy = os.getenv("MACOSWORLD_AWS_HARD_RESET_STRATEGY", "relaunch").strip().lower()
    if strategy not in {"replace", "relaunch"}:
        raise ValueError(f'Illegal MACOSWORLD_AWS_HARD_RESET_STRATEGY "{strategy}"')
    return strategy


def _get_aws_instance_state_file() -> str:
    return os.getenv("MACOSWORLD_AWS_INSTANCE_STATE_FILE", "./run_log/aws_instance_runtime_state.json")


def _load_aws_instance_runtime_state() -> dict:
    global aws_instance_runtime_state
    if len(aws_instance_runtime_state) > 0:
        return aws_instance_runtime_state

    state_file = _get_aws_instance_state_file()
    if not os.path.exists(state_file):
        return aws_instance_runtime_state

    try:
        with open(state_file, "r") as file:
            persisted_state = json.load(file)
        if isinstance(persisted_state, dict):
            aws_instance_runtime_state.update(persisted_state)
    except Exception:
        pass

    return aws_instance_runtime_state


def _save_aws_instance_runtime_state():
    state_file = _get_aws_instance_state_file()
    state_dir = os.path.dirname(state_file)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)
    with open(state_file, "w") as file:
        json.dump(aws_instance_runtime_state, file, indent=2, sort_keys=True)


def _describe_instance(ec2_client, instance_id: str) -> dict:
    response = ec2_client.describe_instances(InstanceIds=[instance_id])
    reservations = response.get("Reservations", [])
    if len(reservations) == 0 or len(reservations[0].get("Instances", [])) == 0:
        raise RuntimeError(f'Instance "{instance_id}" not found.')
    return reservations[0]["Instances"][0]


def _get_instance_ssh_host(instance_description: dict, default_ssh_host: str = None):
    public_dns = instance_description.get("PublicDnsName")
    if isinstance(public_dns, str) and public_dns.strip() != "":
        return public_dns

    public_ip = instance_description.get("PublicIpAddress")
    if isinstance(public_ip, str) and public_ip.strip() != "":
        return public_ip

    return default_ssh_host


def _get_instance_name_tag(instance_description: dict):
    for tag in instance_description.get("Tags", []):
        if tag.get("Key") == "Name":
            return tag.get("Value")
    return None


def _record_aws_instance_runtime_state(
    bootstrap_instance_id: str,
    instance_description: dict,
    ssh_host: str,
    snapshot_name: str = None,
):
    runtime_state = _load_aws_instance_runtime_state()
    existing_state = runtime_state.get(bootstrap_instance_id, {})
    runtime_state[bootstrap_instance_id] = {
        "active_instance_id": instance_description["InstanceId"],
        "ssh_host": ssh_host,
        "snapshot_name": snapshot_name if snapshot_name is not None else existing_state.get("snapshot_name"),
        "host_id": instance_description.get("Placement", {}).get("HostId"),
        "availability_zone": instance_description.get("Placement", {}).get("AvailabilityZone"),
        "subnet_id": instance_description.get("SubnetId"),
        "security_group_ids": [group["GroupId"] for group in instance_description.get("SecurityGroups", [])],
        "key_name": instance_description.get("KeyName"),
        "instance_type": instance_description.get("InstanceType"),
        "instance_name": _get_instance_name_tag(instance_description),
        "iam_instance_profile_arn": instance_description.get("IamInstanceProfile", {}).get("Arn"),
    }
    _save_aws_instance_runtime_state()
    return runtime_state[bootstrap_instance_id]


def _resolve_aws_instance_runtime_state(
    ec2_client,
    bootstrap_instance_id: str,
    bootstrap_ssh_host: str,
):
    runtime_state = _load_aws_instance_runtime_state()
    candidate_instance_ids = []

    if bootstrap_instance_id in runtime_state:
        active_instance_id = runtime_state[bootstrap_instance_id].get("active_instance_id")
        if active_instance_id is not None:
            candidate_instance_ids.append(active_instance_id)
    candidate_instance_ids.append(bootstrap_instance_id)

    checked_instance_ids = set()
    for candidate_instance_id in candidate_instance_ids:
        if candidate_instance_id in checked_instance_ids:
            continue
        checked_instance_ids.add(candidate_instance_id)
        try:
            instance_description = _describe_instance(ec2_client, candidate_instance_id)
        except Exception:
            continue

        instance_state = instance_description.get("State", {}).get("Name")
        if instance_state == "terminated":
            continue

        previous_runtime_state = runtime_state.get(bootstrap_instance_id, {})
        resolved_ssh_host = _get_instance_ssh_host(
            instance_description,
            default_ssh_host=previous_runtime_state.get("ssh_host") or bootstrap_ssh_host,
        )
        recorded_state = _record_aws_instance_runtime_state(
            bootstrap_instance_id=bootstrap_instance_id,
            instance_description=instance_description,
            ssh_host=resolved_ssh_host,
            snapshot_name=previous_runtime_state.get("snapshot_name"),
        )
        return recorded_state

    raise RuntimeError(
        f'Unable to resolve a live AWS instance for bootstrap instance "{bootstrap_instance_id}".'
    )


def _wait_for_aws_instance_state(
    ec2_client,
    instance_id: str,
    target_states,
    timeout_seconds: int,
    poll_interval_seconds: int = 10,
):
    if isinstance(target_states, str):
        target_states = {target_states}
    else:
        target_states = set(target_states)

    waiting_time = 0
    while True:
        instance_description = _describe_instance(ec2_client, instance_id)
        instance_state = instance_description.get("State", {}).get("Name")
        if instance_state in target_states:
            return waiting_time

        waiting_time += poll_interval_seconds
        if waiting_time > timeout_seconds:
            raise TimeoutError(
                f'Instance "{instance_id}" did not enter states {sorted(target_states)} in time. '
                f'CurrentState={instance_state}.'
            )
        time.sleep(poll_interval_seconds)


def _wait_for_aws_host_state(
    ec2_client,
    host_id: str,
    target_states,
    timeout_seconds: int,
    poll_interval_seconds: int = 10,
):
    if isinstance(target_states, str):
        target_states = {target_states}
    else:
        target_states = set(target_states)

    waiting_time = 0
    while True:
        host_response = ec2_client.describe_hosts(HostIds=[host_id])
        host_state = host_response["Hosts"][0]["State"]
        if host_state in target_states:
            return waiting_time

        waiting_time += poll_interval_seconds
        if waiting_time > timeout_seconds:
            raise TimeoutError(
                f'Host "{host_id}" did not enter states {sorted(target_states)} in time. '
                f'CurrentState={host_state}.'
            )
        time.sleep(poll_interval_seconds)


def _wait_for_aws_instance_ssh_host(
    ec2_client,
    instance_id: str,
    timeout_seconds: int,
    default_ssh_host: str = None,
    poll_interval_seconds: int = 10,
):
    waiting_time = 0
    while True:
        instance_description = _describe_instance(ec2_client, instance_id)
        ssh_host = _get_instance_ssh_host(instance_description, default_ssh_host=default_ssh_host)
        if isinstance(ssh_host, str) and ssh_host.strip() != "":
            return ssh_host, waiting_time

        waiting_time += poll_interval_seconds
        if waiting_time > timeout_seconds:
            raise TimeoutError(f'Instance "{instance_id}" did not obtain a reachable SSH host in time.')
        time.sleep(poll_interval_seconds)


def _launch_aws_instance_on_existing_host(
    ec2_client,
    launch_state: dict,
    snapshot_id: str,
):
    run_instances_kwargs = {
        "ImageId": snapshot_id,
        "InstanceType": launch_state["instance_type"],
        "MinCount": 1,
        "MaxCount": 1,
        "KeyName": launch_state["key_name"],
        "SecurityGroupIds": launch_state["security_group_ids"],
        "SubnetId": launch_state["subnet_id"],
        "Placement": {
            "Tenancy": "host",
            "HostId": launch_state["host_id"],
            "AvailabilityZone": launch_state["availability_zone"],
        },
    }

    if launch_state.get("iam_instance_profile_arn") is not None:
        run_instances_kwargs["IamInstanceProfile"] = {"Arn": launch_state["iam_instance_profile_arn"]}

    if launch_state.get("instance_name") is not None:
        run_instances_kwargs["TagSpecifications"] = [{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": launch_state["instance_name"]}],
        }]

    run_instances_response = ec2_client.run_instances(**run_instances_kwargs)
    return run_instances_response["Instances"][0]["InstanceId"]


def _describe_instance_health(ec2_client, instance_id: str) -> dict:
    response = ec2_client.describe_instance_status(
        IncludeAllInstances=True,
        InstanceIds=[instance_id]
    )
    if len(response["InstanceStatuses"]) == 0:
        return {
            "instance_state": "unknown",
            "instance_status": "unknown",
            "system_status": "unknown",
        }
    instance_status = response["InstanceStatuses"][0]
    return {
        "instance_state": instance_status["InstanceState"]["Name"],
        "instance_status": instance_status["InstanceStatus"]["Status"],
        "system_status": instance_status["SystemStatus"]["Status"],
    }


def _wait_for_aws_replace_root_volume_task(
    ec2_client,
    replace_root_volume_task_id: str,
    timeout_seconds: int,
    poll_interval_seconds: int = 10,
):
    waiting_time = 0
    while True:
        response = ec2_client.describe_replace_root_volume_tasks(
            ReplaceRootVolumeTaskIds=[replace_root_volume_task_id]
        )
        task = response["ReplaceRootVolumeTasks"][0]
        task_state = task["TaskState"]
        if task_state == "succeeded":
            return waiting_time
        if task_state in {"failed", "failing", "cancelled", "cancelling"}:
            raise RuntimeError(
                f'Replace root volume task "{replace_root_volume_task_id}" entered state "{task_state}".'
            )
        waiting_time += poll_interval_seconds
        if waiting_time > timeout_seconds:
            raise TimeoutError(
                f'Replace root volume task "{replace_root_volume_task_id}" timed out after {waiting_time}s.'
            )
        time.sleep(poll_interval_seconds)


def _wait_for_aws_instance_health(
    ec2_client,
    instance_id: str,
    timeout_seconds: int,
    poll_interval_seconds: int = 10,
    reboot_on_impaired: bool = True,
):
    waiting_time = 0
    reboot_attempted = False
    reboot_after_seconds = min(300, timeout_seconds // 2) if timeout_seconds > 0 else 0

    while True:
        health = _describe_instance_health(ec2_client, instance_id)
        if (
            health["instance_state"] == "running"
            and health["instance_status"] == "ok"
            and health["system_status"] == "ok"
        ):
            return waiting_time

        if (
            reboot_on_impaired
            and not reboot_attempted
            and waiting_time >= reboot_after_seconds
            and (
                health["instance_status"] == "impaired"
                or health["system_status"] == "impaired"
            )
        ):
            print_message(
                f'Instance "{instance_id}" is impaired after snapshot recovery. Rebooting once to recover health.',
                title='EC2'
            )
            ec2_client.reboot_instances(InstanceIds=[instance_id])
            reboot_attempted = True

        waiting_time += poll_interval_seconds
        if waiting_time > timeout_seconds:
            raise TimeoutError(
                f'Instance "{instance_id}" health did not recover in time. '
                f'State={health["instance_state"]}, '
                f'InstanceStatus={health["instance_status"]}, '
                f'SystemStatus={health["system_status"]}.'
            )
        time.sleep(poll_interval_seconds)


def _wait_for_ssh_connectivity(
    remote_client,
    timeout_seconds: int,
    error_message: str,
    poll_interval_seconds: int = 10,
    ec2_client = None,
    instance_id: str = None,
):
    waiting_time = 0
    recovery_attempted = False
    recovery_after_seconds = min(
        max(120, poll_interval_seconds),
        max(poll_interval_seconds, timeout_seconds // 3) if timeout_seconds > 0 else poll_interval_seconds,
    )
    while True:
        if remote_client.check_ssh_connectivity():
            return waiting_time

        if (
            not recovery_attempted
            and ec2_client is not None
            and instance_id is not None
            and waiting_time >= recovery_after_seconds
        ):
            try:
                health = _describe_instance_health(ec2_client, instance_id)
                if health["instance_state"] == "running":
                    print_message(
                        f'SSH is still unavailable for instance "{instance_id}" after {waiting_time}s. '
                        f'Attempting one EC2 reboot-based recovery.',
                        title='EC2'
                    )
                    ec2_client.reboot_instances(InstanceIds=[instance_id])
                    _wait_for_aws_instance_health(
                        ec2_client,
                        instance_id=instance_id,
                        timeout_seconds=max(recovery_after_seconds, 300),
                        reboot_on_impaired=False,
                    )
                    instance_description = _describe_instance(ec2_client, instance_id)
                    refreshed_ssh_host = _get_instance_ssh_host(
                        instance_description,
                        default_ssh_host=remote_client.ssh_host,
                    )
                    remote_client.ssh_host = refreshed_ssh_host
                    print_message(
                        f'EC2 reboot recovery finished. Refreshed SSH host: {refreshed_ssh_host}',
                        title='EC2'
                    )
                    recovery_attempted = True
            except Exception as recovery_error:
                print_message(
                    f'EC2 reboot recovery attempt failed: {recovery_error}',
                    title='EC2'
                )
                recovery_attempted = True

        waiting_time += poll_interval_seconds
        if waiting_time > timeout_seconds:
            raise TimeoutError(error_message)
        time.sleep(poll_interval_seconds)


def _get_current_aws_snapshot_name(ec2_client, instance_id: str):
    if instance_id in aws_instance_snapshot_state:
        return aws_instance_snapshot_state[instance_id]

    current_snapshot_name = None
    try:
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        image_id = response["Reservations"][0]["Instances"][0]["ImageId"]
        ami_reverse_lookup_table = {ami_id: snapshot_name for snapshot_name, ami_id in ami_lookup_table.items()}
        current_snapshot_name = ami_reverse_lookup_table.get(image_id)
    except Exception:
        current_snapshot_name = None

    if current_snapshot_name is None:
        configured_snapshot_name = os.getenv("MACOSWORLD_AWS_INITIAL_SNAPSHOT_NAME")
        if configured_snapshot_name in ami_lookup_table:
            current_snapshot_name = configured_snapshot_name

    aws_instance_snapshot_state[instance_id] = current_snapshot_name
    return current_snapshot_name


def _build_aws_soft_reset_command(snapshot_name: str) -> str:
    common_commands = [
        "for app in \"Calculator\" \"Contacts\" \"Reminders\" \"Stocks\" \"Notes\" \"System Settings\" \"Finder\" \"QuickTime Player\" \"Music\" \"Preview\" \"TextEdit\" \"Script Editor\" \"Pages\" \"Numbers\" \"Keynote\" \"Xcode\" \"Calendar\" \"Mail\" \"Photos\" \"Freeform\" \"Chess\" \"Dictionary\" \"Disk Utility\" \"Activity Monitor\" \"Weather\"; do osascript -e \"tell application \\\"$app\\\" to quit saving no\" >/dev/null 2>&1 || osascript -e \"tell application \\\"$app\\\" to quit\" >/dev/null 2>&1 || true; done",
        "for proc in Dock Finder Calculator Contacts Reminders Stocks Notes Music \"QuickTime Player\" Preview TextEdit \"Script Editor\" Pages Numbers Keynote Xcode Calendar Mail Photos Freeform Chess Dictionary \"Disk Utility\" \"Activity Monitor\" Weather; do killall \"$proc\" >/dev/null 2>&1 || true; done",
        "osascript -e 'tell application \"Finder\" to close windows' >/dev/null 2>&1 || true",
        "rm -rf \"$HOME/Documents/benchmark_files\" \"$HOME/Documents/BenchmarkApp\" >/dev/null 2>&1 || true",
        "find \"$HOME/Desktop\" -maxdepth 1 -type f \\( -name 'TODO.txt' -o -name '*.vcf' -o -name '*.zip' -o -name '*.numbers' -o -name '*.key' -o -name '*.pages' -o -name 'arXiv-*.svg' \\) -delete >/dev/null 2>&1 || true",
        "if [ -f \"$HOME/Benchmark_Backup/com.apple.dock.plist\" ]; then cp \"$HOME/Benchmark_Backup/com.apple.dock.plist\" \"$HOME/Library/Preferences/com.apple.dock.plist\"; fi",
        "killall Dock >/dev/null 2>&1 || true",
        "open -a Finder >/dev/null 2>&1 || true",
        "sleep 2",
    ]

    snapshot_specific_commands = {
        "snapshot_used_en": [
            "defaults delete com.apple.calculator LastResultValue >/dev/null 2>&1 || true",
            "defaults delete com.apple.Notes windowStateArchive >/dev/null 2>&1 || true",
            "defaults delete com.apple.dock autohide >/dev/null 2>&1 || true",
            "defaults delete com.apple.dock mineffect >/dev/null 2>&1 || true",
            "defaults delete com.apple.universalaccess hoverTextEnabled >/dev/null 2>&1 || true",
            "defaults delete com.apple.universalaccess hoverTypingEnabled >/dev/null 2>&1 || true",
            "defaults delete com.apple.universalaccess closeViewHotkeysEnabled >/dev/null 2>&1 || true",
            "defaults delete -g KeyRepeat >/dev/null 2>&1 || true",
            "defaults delete -g InitialKeyRepeat >/dev/null 2>&1 || true",
            "defaults delete -g AppleAccentColor >/dev/null 2>&1 || true",
            "defaults delete -g AppleInterfaceStyle >/dev/null 2>&1 || true",
            "defaults delete com.apple.weather modules.location.lastViewedLocation >/dev/null 2>&1 || true",
            "defaults delete com.apple.Spotlight userHasMovedWindow >/dev/null 2>&1 || true",
            "defaults delete com.apple.Chess MBCBoardStyle >/dev/null 2>&1 || true",
            "defaults delete com.apple.Chess MBCSearchTime >/dev/null 2>&1 || true",
            "defaults delete com.apple.Chess MBCNewGamePlayers >/dev/null 2>&1 || true",
            "osascript -e 'tell application \"Reminders\" to delete every reminder' >/dev/null 2>&1 || true",
            "osascript -e 'tell application \"Reminders\" to delete every list whose name is \"Shopping\"' >/dev/null 2>&1 || true",
            "osascript -e 'tell application \"Contacts\" to delete every person' >/dev/null 2>&1 || true",
            "osascript -e 'tell application \"Contacts\" to delete every group whose name is \"Work\"' >/dev/null 2>&1 || true",
            "osascript -e 'tell application \"Contacts\" to delete every group whose name is \"Shopping\"' >/dev/null 2>&1 || true",
            "osascript -e 'tell application \"Notes\" to delete every note' -e 'tell application \"Notes\" to delete every note in folder \"Recently Deleted\"' >/dev/null 2>&1 || true",
        ],
        "snapshot_usedApps_en": [
            "defaults delete com.apple.calculator LastResultValue >/dev/null 2>&1 || true",
            "defaults delete com.apple.Notes windowStateArchive >/dev/null 2>&1 || true",
            "if [ -d \"$HOME/Benchmark_Backup/benchmark_files\" ] && [ ! -e \"$HOME/Documents/benchmark_files\" ]; then cp -R \"$HOME/Benchmark_Backup/benchmark_files\" \"$HOME/Documents\" >/dev/null 2>&1 || true; fi",
        ],
    }

    commands = ["set +e"]
    commands.extend(common_commands)
    commands.extend(snapshot_specific_commands.get(snapshot_name, []))
    commands.append("exit 0")
    return "; ".join(commands)


def _run_aws_soft_reset(remote_client, snapshot_name: str, task_requires_snapshot_recovery: bool):
    if task_requires_snapshot_recovery:
        print_message(
            f'Task requests snapshot recovery, but using in-guest soft reset for snapshot "{snapshot_name}".',
            title='EC2'
        )
    else:
        print_message(f'Running in-guest soft reset for snapshot "{snapshot_name}".', title='EC2')

    soft_reset_command = _build_aws_soft_reset_command(snapshot_name)
    soft_reset_complete, soft_reset_output = remote_client.run_ssh_command(soft_reset_command)
    if not soft_reset_complete:
        raise RuntimeError(f'Soft reset command failed for snapshot "{snapshot_name}". Output: {soft_reset_output}')


def _run_aws_hard_reset(
    ec2_client,
    instance_id: str,
    snapshot_id: str,
    snapshot_recovery_timeout_seconds: int,
):
    replace_root_volume_task_response = ec2_client.create_replace_root_volume_task(
        InstanceId=instance_id,
        ImageId=snapshot_id,
        DeleteReplacedRootVolume=True
    )
    replace_root_volume_task_id = replace_root_volume_task_response["ReplaceRootVolumeTask"]["ReplaceRootVolumeTaskId"]

    print_message(
        f'Reinitiating instance "{instance_id}" from image "{snapshot_id}" via replace task "{replace_root_volume_task_id}"',
        title='EC2'
    )
    replace_waiting_time = _wait_for_aws_replace_root_volume_task(
        ec2_client,
        replace_root_volume_task_id=replace_root_volume_task_id,
        timeout_seconds=snapshot_recovery_timeout_seconds,
    )
    print_message(f'Replace root volume task complete in {replace_waiting_time}s.', title='EC2')

    health_waiting_time = _wait_for_aws_instance_health(
        ec2_client,
        instance_id=instance_id,
        timeout_seconds=snapshot_recovery_timeout_seconds,
        reboot_on_impaired=_parse_truthy_env_var("MACOSWORLD_AWS_REBOOT_ON_IMPAIRED", True),
    )
    print_message(f'Instance health recovered in {health_waiting_time}s.', title='EC2')


def _run_aws_relaunch_reset(
    ec2_client,
    bootstrap_instance_id: str,
    active_instance_id: str,
    snapshot_name: str,
    snapshot_id: str,
    snapshot_recovery_timeout_seconds: int,
    bootstrap_ssh_host: str,
):
    instance_description = _describe_instance(ec2_client, active_instance_id)
    active_ssh_host = _get_instance_ssh_host(instance_description, default_ssh_host=bootstrap_ssh_host)
    launch_state = _record_aws_instance_runtime_state(
        bootstrap_instance_id=bootstrap_instance_id,
        instance_description=instance_description,
        ssh_host=active_ssh_host,
        snapshot_name=_get_current_aws_snapshot_name(ec2_client, active_instance_id),
    )

    if launch_state.get("host_id") is None:
        raise RuntimeError(
            f'Instance "{active_instance_id}" is not attached to a dedicated host. Cannot relaunch on the same host.'
        )

    print_message(
        f'Terminating instance "{active_instance_id}" and relaunching from image "{snapshot_id}" on host "{launch_state["host_id"]}".',
        title='EC2'
    )
    ec2_client.terminate_instances(InstanceIds=[active_instance_id])
    terminate_waiting_time = _wait_for_aws_instance_state(
        ec2_client,
        instance_id=active_instance_id,
        target_states={"terminated"},
        timeout_seconds=snapshot_recovery_timeout_seconds,
    )
    print_message(f'Previous instance terminated in {terminate_waiting_time}s.', title='EC2')

    host_waiting_time = _wait_for_aws_host_state(
        ec2_client,
        host_id=launch_state["host_id"],
        target_states={"available"},
        timeout_seconds=snapshot_recovery_timeout_seconds,
    )
    print_message(f'Host "{launch_state["host_id"]}" became available in {host_waiting_time}s.', title='EC2')

    relaunched_instance_id = _launch_aws_instance_on_existing_host(
        ec2_client,
        launch_state=launch_state,
        snapshot_id=snapshot_id,
    )
    print_message(
        f'Launched replacement instance "{relaunched_instance_id}" from image "{snapshot_id}" on host "{launch_state["host_id"]}".',
        title='EC2'
    )

    running_waiting_time = _wait_for_aws_instance_state(
        ec2_client,
        instance_id=relaunched_instance_id,
        target_states={"running"},
        timeout_seconds=snapshot_recovery_timeout_seconds,
    )
    print_message(f'Relaunched instance entered running state in {running_waiting_time}s.', title='EC2')

    health_waiting_time = _wait_for_aws_instance_health(
        ec2_client,
        instance_id=relaunched_instance_id,
        timeout_seconds=snapshot_recovery_timeout_seconds,
        reboot_on_impaired=_parse_truthy_env_var("MACOSWORLD_AWS_REBOOT_ON_IMPAIRED", True),
    )
    print_message(f'Relaunched instance health recovered in {health_waiting_time}s.', title='EC2')

    resolved_ssh_host, ssh_host_waiting_time = _wait_for_aws_instance_ssh_host(
        ec2_client,
        instance_id=relaunched_instance_id,
        timeout_seconds=snapshot_recovery_timeout_seconds,
        default_ssh_host=bootstrap_ssh_host,
    )
    print_message(
        f'Relaunched instance published SSH host "{resolved_ssh_host}" in {ssh_host_waiting_time}s.',
        title='EC2'
    )

    new_instance_description = _describe_instance(ec2_client, relaunched_instance_id)
    _record_aws_instance_runtime_state(
        bootstrap_instance_id=bootstrap_instance_id,
        instance_description=new_instance_description,
        ssh_host=resolved_ssh_host,
        snapshot_name=snapshot_name,
    )
    aws_instance_snapshot_state[relaunched_instance_id] = snapshot_name
    if active_instance_id in aws_instance_snapshot_state:
        del aws_instance_snapshot_state[active_instance_id]

    return relaunched_instance_id, resolved_ssh_host


def inprocess_result_matching(inprocess_stdout: str, inprocess_gold_elements: list, inprocess_distracting_elements: list):
    # Match handled properly
    for element in inprocess_gold_elements:
        if element.lower() in inprocess_stdout.lower():
            inprocess_eval_result = 'gold'
            break
    # Match distracted
    for element in inprocess_distracting_elements:
        if element.lower() in inprocess_stdout.lower():
            inprocess_eval_result = 'distracted'
            break
    # No match
    if inprocess_eval_result is None:
        inprocess_eval_result = 'error_no_match'
    return inprocess_eval_result

def run_task(
    # Task-related params
    task_id: str,
    task_dict: dict,
    task_language: str,
    env_language: str,
    save_dir: str,

    # Env-related params
    snapshot_name: str,
    instance_id: str,
    snapshot_recovery_timeout_seconds: int,
    override_env_reset: bool,
    vmx_path: str,

    # Remote connection
    guest_username: str, 
    guest_password: str,
    ssh_host: str,
    ssh_pkey: str,

    # GUI agent
    gui_agent_name: str,

    # Runtime
    max_steps: int,
    task_step_timeout: int,
    pre_command_max_trials: int,
    env_init_command: str,
    eval_init_command: str,
):
    task_uuid = task_dict["id"]
    task_id = task_id
    bootstrap_instance_id = instance_id
    bootstrap_ssh_host = ssh_host

    # Check if env_language is in task_dict['snapshot']
    assert env_language in task_dict['snapshot'], f"Task {task_dict['id']} does not support snapshot language {env_language}"

    # Check if task_language is in task_dict['task']
    assert task_language in task_dict['task'], f"Task {task_dict['id']} does not include task language {task_language}"

    task_requires_snapshot_recovery = task_dict.get("force_snapshot_recovery", False)
    remote_client = None

    # Env reset
    if override_env_reset:
        print('Please manually reset the environment. Press `c` to continue.')
        breakpoint()
    elif vmx_path is not None:
        # VMware env
        snapshot_revert_max_trials = 5
        vmware_tools = VMwareTools(
            guest_username = guest_username,
            guest_password = guest_password,
            ssh_host = None,
            ssh_pkey = ssh_pkey,
            vmx_path = vmx_path
        )
        for trial in range(1, snapshot_revert_max_trials + 1):
            if trial > 1:
                print_message(f'Retrying starting guest machine... ({trial}/{snapshot_revert_max_trials})')
            revert_success_flag, ssh_host = vmware_tools.revert_to_snapshot(snapshot_name)
            if revert_success_flag:
                break

        print_message(f'Guest machine started successfully at {ssh_host}', title = 'VMware')
        
    else:
        # AWS env
        snapshot_id = ami_lookup_table[snapshot_name]
        ec2_client = boto3.client('ec2')
        aws_reset_mode = _get_aws_reset_mode()
        aws_hard_reset_strategy = _get_aws_hard_reset_strategy()

        runtime_state = _resolve_aws_instance_runtime_state(
            ec2_client,
            bootstrap_instance_id=bootstrap_instance_id,
            bootstrap_ssh_host=bootstrap_ssh_host,
        )
        instance_id = runtime_state["active_instance_id"]
        ssh_host = runtime_state["ssh_host"]
        current_snapshot_name = _get_current_aws_snapshot_name(ec2_client, instance_id)

        if aws_reset_mode == "soft":
            use_soft_reset = True
        elif aws_reset_mode == "hard":
            use_soft_reset = False
        else:
            use_soft_reset = current_snapshot_name == snapshot_name and current_snapshot_name is not None

        if use_soft_reset:
            print_message(
                f'Using soft reset for instance "{instance_id}". Current snapshot "{current_snapshot_name}", target snapshot "{snapshot_name}".',
                title='EC2'
            )
            remote_client = VNCClient_SSH(
                guest_username=guest_username,
                guest_password=guest_password,
                ssh_host=ssh_host,
                ssh_pkey=ssh_pkey,
                vmx_path=vmx_path
            )
            _wait_for_ssh_connectivity(
                remote_client,
                timeout_seconds=snapshot_recovery_timeout_seconds,
                error_message=f'Timeout establishing ssh connection to {ssh_host}',
                ec2_client=ec2_client,
                instance_id=instance_id,
            )
            _run_aws_soft_reset(remote_client, snapshot_name, task_requires_snapshot_recovery)
            aws_instance_snapshot_state[instance_id] = snapshot_name
            instance_description = _describe_instance(ec2_client, instance_id)
            _record_aws_instance_runtime_state(
                bootstrap_instance_id=bootstrap_instance_id,
                instance_description=instance_description,
                ssh_host=ssh_host,
                snapshot_name=snapshot_name,
            )
        else:
            print_message(
                f'Using hard reset for instance "{instance_id}". Current snapshot "{current_snapshot_name}", '
                f'target snapshot "{snapshot_name}", strategy "{aws_hard_reset_strategy}".',
                title='EC2'
            )
            if aws_hard_reset_strategy == "replace":
                _run_aws_hard_reset(
                    ec2_client,
                    instance_id=instance_id,
                    snapshot_id=snapshot_id,
                    snapshot_recovery_timeout_seconds=snapshot_recovery_timeout_seconds,
                )
                aws_instance_snapshot_state[instance_id] = snapshot_name
                instance_description = _describe_instance(ec2_client, instance_id)
                ssh_host = _get_instance_ssh_host(instance_description, default_ssh_host=ssh_host)
                _record_aws_instance_runtime_state(
                    bootstrap_instance_id=bootstrap_instance_id,
                    instance_description=instance_description,
                    ssh_host=ssh_host,
                    snapshot_name=snapshot_name,
                )
            else:
                instance_id, ssh_host = _run_aws_relaunch_reset(
                    ec2_client,
                    bootstrap_instance_id=bootstrap_instance_id,
                    active_instance_id=instance_id,
                    snapshot_name=snapshot_name,
                    snapshot_id=snapshot_id,
                    snapshot_recovery_timeout_seconds=snapshot_recovery_timeout_seconds,
                    bootstrap_ssh_host=bootstrap_ssh_host,
                )


    # Establish remote connection

    if remote_client is None:
        remote_client = VNCClient_SSH(
            guest_username = guest_username, 
            guest_password = guest_password, 
            ssh_host = ssh_host,
            ssh_pkey = ssh_pkey,
            vmx_path = vmx_path
        )

    print_message(f'Checking ssh connectivity to {ssh_host}', title = 'VNC Client')
    if vmx_path is None:
        ssh_timeout_error_message = f'Timeout establishing ssh connection to {ssh_host} for instance "{instance_id}"'
    else:
        ssh_timeout_error_message = f'Timeout establishing ssh connection to {ssh_host}'
    ssh_waiting_time = _wait_for_ssh_connectivity(
        remote_client,
        timeout_seconds=snapshot_recovery_timeout_seconds,
        error_message=ssh_timeout_error_message,
        ec2_client=ec2_client if vmx_path is None else None,
        instance_id=instance_id if vmx_path is None else None,
    )
    print_message(f'SSH connectivity ready after {ssh_waiting_time}s.', title='VNC Client')

    remote_client.connect()
    print_message(f'Connected to {ssh_host}', title = 'VNC Client')


    # Construct GUI Agent
    gui_agent = get_gui_agent(gui_agent_name, remote_client)
    if hasattr(gui_agent, "set_task_skills"):
        gui_agent.set_task_skills(task_dict.get("__resolved_task_skills__", []))

    # print('Manually reset the environment')
    # breakpoint()

    # Run prep command
    remote_client.run_ssh_command(env_init_command)
    if 'pre_command' in task_dict:
        pre_command = task_dict['pre_command']
        pre_command_complete_flag = False
        for trial in range(pre_command_max_trials):
            if isinstance(pre_command, str):
                # When the prep command is a string
                pre_command_complete_flag, pre_command_output = remote_client.run_ssh_command(pre_command)
            elif isinstance(pre_command, dict):
                # When the prep command is a dict of language-dependent string
                if env_language in pre_command:
                    pre_command_complete_flag, pre_command_output = remote_client.run_ssh_command(pre_command[env_language])
                else:
                    raise NotImplementedError(f'Task {task_id} has no preparation command for env language "{env_language}".')
            else:
                raise TypeError(f'Unknown prep command type ({type(pre_command)}) in task {task_id}.')
            if pre_command_complete_flag:
                # When the prep command finishes
                break
        if "force_error_free_prep" in task_dict:
            if task_dict["force_error_free_prep"] and not pre_command_complete_flag:
                # When the prep command repeatedly encounter errors until a max trial
                raise RuntimeError(f'Prep command not finished for task {task_id}.')
            
    inprocess_event_handler = None
    if 'in_process' in task_dict:
        inprocess_event_handler = AsyncSSHCommandHandler(ssh_host, guest_username, ssh_pkey)
        inprocess_command, inprocess_event_start_timestep, inprocess_gold_elements, inprocess_distracting_elements = task_dict['in_process']

    if 'before_action_delay_seconds' in task_dict:
        before_action_delay_seconds = task_dict['before_action_delay_seconds']
        print_message(f'Waiting for {before_action_delay_seconds}s before benchmarking', title = f'Task {task_id}/{env_language}/{task_language}')
        time.sleep(before_action_delay_seconds)


    # Start interactive loop

    task = task_dict['task'][task_language]

    for current_step in range(1, max_steps + 1):
        time.sleep(5)

        # Inject events
        if inprocess_event_handler is not None:
            if current_step == inprocess_event_start_timestep:
                inprocess_event_handler.run_command(inprocess_command)
                time.sleep(5)
                print_message(title = f'Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}', content = 'Distraction event injected')

        # Call agent
        status = gui_agent.step(
            task_id = task_id,
            current_step = current_step,
            max_steps = max_steps,
            env_language = env_language,
            task_language = task_language,

            task = task,
            task_step_timeout = task_step_timeout,
            save_dir = save_dir
        )

        print_message(title = f'Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}', content = f'Status: {status}')
        
        if status != "unfinished":
            break

    gui_agent.save_conversation_history(save_dir)




    # In-process event grading
    if inprocess_event_handler is not None:
        # End event
        inprocess_return_code, inprocess_stdout, inprocess_stderr, inprocess_end_type = inprocess_event_handler.end_command()

        # Print result
        inprocess_log_message = f'Log as follows:\nReturn value {inprocess_return_code}\nSTDOUT: {inprocess_stdout}\nSTDERR: {inprocess_stderr}'


        # Evaluate distraction
        inprocess_eval_result = None
        if inprocess_end_type == 'killed':
            # Not handled
            inprocess_eval_result = 'not_handled'
        elif inprocess_return_code == 0 and isinstance(inprocess_stdout, str):
            # Result matching
            inprocess_eval_result = inprocess_result_matching(
                inprocess_stdout,
                inprocess_gold_elements,
                inprocess_distracting_elements
            ) 
        elif inprocess_return_code == 1 and isinstance(inprocess_stdout, str): # If the button name is "Cancel"
            if '-128' in inprocess_stdout: 
                # Result matching
                inprocess_eval_result = inprocess_result_matching(
                    inprocess_stdout,
                    inprocess_gold_elements,
                    inprocess_distracting_elements
                )
            else:
                # Other error
                inprocess_eval_result = 'error'
        else:
            # # User canceled error
            # if isinstance(inprocess_stdout, str):
            #     if 'User canceled. (-128)' in inprocess_stdout:
            #         raise RuntimeError(f'Inprocess event failed to initialise. STDOUT: {inprocess_stdout}')

            # Other error
            inprocess_eval_result = 'error'

        print_message(f'Event {inprocess_end_type} with status {inprocess_eval_result}. {inprocess_log_message}', title = 'Distraction Event')
                
        with open(os.path.join(save_dir, "distraction_result.txt"), "w") as file:
            file.write(f"{inprocess_eval_result}\n\n{inprocess_log_message}")



    # Task grading

    if "before_grading_delay_seconds" in task_dict:
        before_grading_delay_seconds = task_dict['before_grading_delay_seconds']
        if before_grading_delay_seconds > 0:
            print_message(f'Waiting for {before_grading_delay_seconds}s before grading', title = f'Task {task_id}/{env_language}/{task_language}')
            time.sleep(before_grading_delay_seconds)

    evaluator = Evaluator(ssh_host, guest_username, ssh_pkey)
    evaluator.run_command(eval_init_command)

    eval_result = evaluator(task_dict["grading_command"])
    print_message(title = 'Evaluation result', content = str(eval_result))

    if isinstance(eval_result, int):
        with open(os.path.join(save_dir, "eval_result.txt"), "w") as file:
            if eval_result < 0:
                file.write("eval_failed\n")
            file.write(str(eval_result))
    elif isinstance(eval_result, list):
        with open(os.path.join(save_dir, "eval_result.txt"), "w") as file:
            file.write("eval_failed\n")
            for line in eval_result:
                file.write(f"{line}\n")
    else:
        raise RuntimeError("Illegal return type from evaluator")
    

    try:
        remote_client.disconnect()
    except Exception as e:
        print_message(title = 'VNC Client', content = f'Error disconnecting: {e}')
