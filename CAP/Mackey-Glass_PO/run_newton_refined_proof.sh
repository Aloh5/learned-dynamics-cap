#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
python_binary="${PYTHON_BINARY:-python3}"
matlab_binary="${MATLAB_BINARY:-/Applications/MATLAB_R2024b.app/bin/matlab}"
intlab_dir="${INTLAB_PATH:?set INTLAB_PATH to your INTLAB installation}"
export MG_FIXED_CAP_OUTPUT_DIR="$script_dir/outputs_newton_refined"
export MG_CAP_RADIUS="${MG_CAP_RADIUS:-1.2e-12}"
mkdir -p "$MG_FIXED_CAP_OUTPUT_DIR"
# Remove downstream summaries before rebuilding, so a failed run leaves no
# stale success summary from a different candidate.
rm -f "$MG_FIXED_CAP_OUTPUT_DIR/newton_refinement_summary.json" "$MG_FIXED_CAP_OUTPUT_DIR/newton_refinement_manifest.json"
"$python_binary" "$script_dir/fixed_parameter_candidate.py" --newton-steps 5 --order 80 --output-dir "$MG_FIXED_CAP_OUTPUT_DIR"
"$matlab_binary" -batch "addpath('$intlab_dir'); startintlab; run('$script_dir/proof_MG_newton_refined.m');" -logfile "$MG_FIXED_CAP_OUTPUT_DIR/cap.log"
"$python_binary" "$script_dir/summarize_newton_refinement.py"
"$python_binary" -m unittest discover -s "$script_dir" -p 'test_*.py' -v
