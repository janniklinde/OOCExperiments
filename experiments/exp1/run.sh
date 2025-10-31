#!/bin/bash
set -euo pipefail

# shellcheck disable=SC1091
source ../../sysds.conf   # adjust path

echo "mode,rep1,rep2,rep3" > results.csv
modes=("cp" "ooc")

for mode in "${modes[@]}"; do
  row="$mode"
  for rep in {1..3}; do
    start=$(date +%s%N)

    # pick jar + optional flag
    if [[ $mode == "ooc" ]]; then
      jar="$SYSDS_JAR_OOC"
      oocflag="-ooc"
    else
      jar="$SYSDS_JAR_CP"
      oocflag=""
    fi

    file=./exp.dml

    # base cmd
    cmd=(
      "${SYSDS_CMD_COMMON[@]}"
      "$jar"
      -f "$file"
      -exec singlenode
    )

    # optional flag
    if [[ -n $oocflag ]]; then
      cmd+=("$oocflag")
    fi

    # args
    cmd+=( -args 10000 10000 1.0 "../../data/" )

    printf 'RUN CMD: %q ' "${cmd[@]}"; echo
    output=$("${cmd[@]}")

    echo "$output"

    exec_time=$(echo "$output" | grep -oP 'Total execution time:\s*\K[0-9.]+')
    result=$(echo "$output" | grep -oP 'Result:\s*\K[0-9.]+')

    end=$(date +%s%N)
    dur_ms=$(( (end - start) / 1000000 ))

    row="$row,$exec_time"
    echo "ExecTime: $exec_time"
    echo "Result: $result"
  done
  echo "$row" >> results.csv
done
