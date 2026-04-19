# Baseline5 Fixed Packages

This repository now fixes the English baseline/test scope to the following five domains only:

- `file_management`
- `media`
- `productivity`
- `sys_and_interface`
- `sys_apps`

The tasks are split into two stable packages by snapshot:

- `tasks/baseline5_snapshot_used_en`
  - count: `112`
  - domains:
    - `file_management`: `29`
    - `productivity`: `16`
    - `sys_and_interface`: `29`
    - `sys_apps`: `38`

- `tasks/baseline5_snapshot_usedApps_en`
  - count: `31`
  - domains:
    - `media`: `12`
    - `productivity`: `19`

Total fixed scope: `143` tasks.

Operational note:

- Run these two packages on separate EC2 Mac hosts/instances whenever possible.
- This avoids repeated switching between `snapshot_used_en` and `snapshot_usedApps_en`.
- The package directories contain symlinks back to the canonical task JSON files.
- This split is only for execution and scheduling.
- Final metric reporting must still be grouped by the original five domains.
- In particular, `productivity` spans both snapshot packages and must be merged back before reporting domain-level scores.
