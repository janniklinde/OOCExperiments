#!/bin/bash
set -euo pipefail

# load config
# shellcheck disable=SC1091
source ./sysds.conf

# setup 10kx10k sparsity 1 matrics
for id in {1..3} do
  SYSDS_CMD=( $SYSDS_CMD_COMMON "$SYSDS_JAR" -f ./datagen.dml -exec singlenode -args 10000 10000 1.0 $id )
  output=$("${SYSDS_CMD[@]}")
done
