#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' 'Jetson environment inventory:'
printf 'Architecture: '; uname -m
printf 'Kernel: '; uname -r
printf 'Python: '; python3 --version

if [ -f /etc/nv_tegra_release ]; then
  cat /etc/nv_tegra_release
else
  echo '/etc/nv_tegra_release not found; verify JetPack/L4T manually.'
fi

cat <<'EOF'

Before installing ONNX Runtime:
  1. Record JetPack/L4T and Python versions.
  2. Choose a wheel/build compiled for the device architecture and desired execution provider.
  3. Confirm available providers in Python with:
       import onnxruntime as ort
       print(ort.get_available_providers())
  4. Validate model input/output names and shapes before camera deployment.

This repository intentionally does not promise CUDA/TensorRT acceleration unless the
selected runtime/provider is actually installed and benchmarked on the target device.
EOF
