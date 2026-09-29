#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export KS_CAP_OUTPUT_DIR="$script_dir/outputs_surrogate"
export KS_REQUIRE_SURROGATE=1
"${PYTHON_BINARY:-python3}" -u "$script_dir/generate_surrogate_candidate.py" --resume
"${PYTHON_BINARY:-python3}" -u "$script_dir/scan_block_proof.py" \
  --candidate "$KS_CAP_OUTPUT_DIR/ks_neumann_cap_candidate.mat" \
  --output "$KS_CAP_OUTPUT_DIR/ks_neumann_fourier_proof_data.mat"
matlab_binary="${MATLAB_BINARY:-matlab}"
intlab_dir="${INTLAB_DIR:?Set INTLAB_DIR to your INTLAB installation directory}"
"$matlab_binary" -batch "addpath('$intlab_dir'); startintlab; run('$script_dir/proof_KS_neumann.m')"
"${PYTHON_BINARY:-python3}" -m unittest discover -s "$script_dir" -p 'test_*.py' -v
