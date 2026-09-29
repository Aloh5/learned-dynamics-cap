#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
python_binary="${PYTHON_BINARY:-python3}"
matlab_binary="${MATLAB_BINARY:-matlab}"
intlab_dir="${INTLAB_DIR:?Set INTLAB_DIR to your INTLAB installation directory}"
"$python_binary" "$script_dir/fixed_parameter_candidate.py" --newton-steps 0
"$matlab_binary" -batch "addpath('$intlab_dir'); startintlab; run('$script_dir/proof_MG_fixed_parameter.m');" -logfile "$script_dir/outputs_fixed_parameter/cap.log"
"$python_binary" -m unittest discover -s "$script_dir" -p 'test_fixed_parameter.py' -v
