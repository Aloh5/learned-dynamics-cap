#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
python_binary="${PYTHON_BINARY:-python3}"
matlab_binary="${MATLAB_BINARY:-/Applications/MATLAB_R2024b.app/bin/matlab}"
intlab_dir="${INTLAB_PATH:?set INTLAB_PATH to your INTLAB installation}"
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export KS_REFINED_OUTPUT_DIR="$script_dir/outputs_newton_refined"
export KS_REFINED_RADIUS="${KS_REFINED_RADIUS:-7e-9}"
mkdir -p "$KS_REFINED_OUTPUT_DIR"
rm -f "$KS_REFINED_OUTPUT_DIR/ks_newton_refined_certificate.json" "$KS_REFINED_OUTPUT_DIR/newton_refinement_manifest.json"
"$python_binary" -u "$script_dir/newton_refine.py" --output-dir "$KS_REFINED_OUTPUT_DIR"
"$matlab_binary" -batch "addpath('$intlab_dir'); startintlab; run('$script_dir/proof_KS_newton_refined.m');" -logfile "$KS_REFINED_OUTPUT_DIR/cap.log"
"$python_binary" "$script_dir/summarize_newton_refinement.py"
"$python_binary" -m unittest discover -s "$script_dir" -p 'test_*.py' -v
