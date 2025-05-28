#!/usr/bin/env bash
# ?????????? PPO ? SAC
# ???./run_train.sh [ppo|sac] [config_path]

ALG=${1:-ppo}
CONFIG=${2:-trainer/configs/${ALG}_hover.yaml}

echo "Training algorithm: ${ALG}"
echo "Using config file: ${CONFIG}"

# ?? conda ??
# ???? PowerShell ??? `conda init powershell`
conda activate rov_rl_py311

if ($ALG -eq 'ppo') {
  python trainer/train_ppo.py -c $CONFIG
} elseif ($ALG -eq 'sac') {
  python trainer/train_sac.py -c $CONFIG
} else {
  Write-Error "Unknown algorithm '$ALG'. Use 'ppo' or 'sac'."
  exit 1
}
