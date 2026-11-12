#!/usr/bin/env bash
# Certify the Newton-refined sensor-surrogate orbit (see sensor_proof_candidate.py).
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export KS_CAP_OUTPUT_DIR="$script_dir/outputs_sensor_proof"
if [ ! -f "$KS_CAP_OUTPUT_DIR/ks_neumann_cap_candidate.mat" ]; then
  python -u "$script_dir/sensor_proof_candidate.py"
fi
python -u "$script_dir/scan_block_proof.py" \
  --candidate "$KS_CAP_OUTPUT_DIR/ks_neumann_cap_candidate.mat" \
  --output "$KS_CAP_OUTPUT_DIR/ks_neumann_fourier_proof_data.mat"
matlab_binary="${MATLAB_BINARY:-/Applications/MATLAB_R2024b.app/bin/matlab}"
intlab_dir="${INTLAB_PATH:?set INTLAB_PATH to your INTLAB installation}"
"$matlab_binary" -batch "addpath('$intlab_dir'); startintlab; run('$script_dir/proof_KS_sensor_refined.m')"
