#!/usr/bin/env bash
set -euo pipefail

python3 render_remote_figures.py \
  results-remote/20260921T152059.619153+0000 \
  --fallback \
    results-remote/20260916T075736.053003+0000 \
    results-remote/20260902T181411.384560+0000 \
    results-remote/20260908T125528.666457+0000 \
  --prefer-successful \
  --figures runtime-io \
  --out results-remote-figures
