#!/usr/bin/env bash
# ????????? evaluate.py ??? checkpoint ???
# ???./run_eval.sh <checkpoint_path>

if ([string]::IsNullOrWhiteSpace($args[0])) {
  Write-Host "Usage: $0 <checkpoint_path>"
  exit 1
}
$CKPT_PATH = $args[0]

echo "Evaluating checkpoint: $CKPT_PATH"

# ?? conda ??
conda activate rov_rl_py311

python evaluate.py -c $CKPT_PATH
