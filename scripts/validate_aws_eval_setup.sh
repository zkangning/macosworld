#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${MACOSWORLD_NO_AUTO_ENV:-0}" != "1" && -f "$repo_root/.env" ]]; then
  set -a
  source "$repo_root/.env"
  set +a
fi

region="${AWS_DEFAULT_REGION:-${AWS_REGION:-ap-southeast-1}}"
export AWS_DEFAULT_REGION="$region"

echo "== AWS identity =="
aws sts get-caller-identity --output table
echo

echo "== Dedicated mac2 host quota =="
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-5D8DADF5 \
  --query 'Quota.{QuotaCode:QuotaCode,Value:Value,Adjustable:Adjustable}' \
  --output table
echo

echo "== Default VPC =="
aws ec2 describe-vpcs \
  --filters Name=isDefault,Values=true \
  --query 'Vpcs[].{VpcId:VpcId,Cidr:CidrBlock,State:State}' \
  --output table
echo

echo "== Default subnets =="
aws ec2 describe-subnets \
  --filters Name=default-for-az,Values=true \
  --query 'Subnets[].{SubnetId:SubnetId,Az:AvailabilityZone,MapPublicIpOnLaunch:MapPublicIpOnLaunch,VpcId:VpcId}' \
  --output table
echo

echo "== Security groups =="
aws ec2 describe-security-groups \
  --query 'SecurityGroups[].{GroupId:GroupId,GroupName:GroupName,VpcId:VpcId,Description:Description}' \
  --output table
echo

echo "== Key pairs =="
aws ec2 describe-key-pairs \
  --query 'KeyPairs[].{KeyName:KeyName,KeyType:KeyType,CreateTime:CreateTime}' \
  --output table
echo

echo "== Dedicated hosts =="
aws ec2 describe-hosts \
  --filter Name=instance-type,Values=mac2.metal \
  --query 'Hosts[].{HostId:HostId,Az:AvailabilityZone,State:State,Instances:Instances[].InstanceId}' \
  --output table
echo

echo "== mac2.metal instances =="
aws ec2 describe-instances \
  --filters Name=instance-type,Values=mac2.metal Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[].Instances[].{InstanceId:InstanceId,State:State.Name,Az:Placement.AvailabilityZone,HostId:Placement.HostId,PublicDns:PublicDnsName,KeyName:KeyName,LaunchTime:LaunchTime}' \
  --output table
echo

echo "== AMI visibility from constants.py =="
python3 - <<'PY' | while IFS=$'\t' read -r snapshot_name image_id; do
from constants import ami_lookup_table
for snapshot_name, image_id in ami_lookup_table.items():
    print(f"{snapshot_name}\t{image_id}")
PY
  found="$(aws ec2 describe-images --image-ids "$image_id" --query 'length(Images)' --output text 2>/dev/null || echo 0)"
  if [[ "$found" == "1" ]]; then
    printf '[ok]   %-24s %s\n' "$snapshot_name" "$image_id"
  else
    printf '[warn] %-24s %s not visible in %s\n' "$snapshot_name" "$image_id" "$region"
  fi
done
echo

if [[ -n "${MACOSWORLD_SSH_PKEY:-}" ]]; then
  echo "== Local SSH key =="
  if [[ -f "$MACOSWORLD_SSH_PKEY" ]]; then
    ls -l "$MACOSWORLD_SSH_PKEY"
  else
    echo "Missing local SSH key: $MACOSWORLD_SSH_PKEY"
  fi
  echo
fi

if [[ -n "${MACOSWORLD_INSTANCE_ID:-}" ]]; then
  echo "== Requested instance =="
  aws ec2 describe-instances \
    --instance-ids "$MACOSWORLD_INSTANCE_ID" \
    --query 'Reservations[].Instances[].{InstanceId:InstanceId,State:State.Name,Az:Placement.AvailabilityZone,HostId:Placement.HostId,PublicDns:PublicDnsName,KeyName:KeyName,LaunchTime:LaunchTime}' \
    --output table
  echo
fi

cat <<'EOF'
Interpretation:
- If quota Value is 0, you cannot allocate a mac2 Dedicated Host yet.
- If there is no key pair, create one before launching the Mac instance.
- If there is no Dedicated Host or mac2.metal instance, the benchmark cannot start.
- Any [warn] AMI means the corresponding language or snapshot family may fail during root-volume replacement.
EOF
