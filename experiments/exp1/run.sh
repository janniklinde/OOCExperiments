#!/bin/bash
set -euo pipefail

# load config
# shellcheck disable=SC1091
source ../sysds.conf

echo "mode,rep1,rep2,rep3" > results.csv
modes=("cp" "ooc")

for mode in "${modes[@]}"; do
  row="$mode"
  for rep in {1..3}; do
    start=$(date +%s%N)

    if [ "$mode" = "ooc" ]; then
      SYSDS_JAR="$SYSDS_JAR_OOC"
      OOCFlag="-ooc"
    else
      SYSDS_JAR="$SYSDS_JAR_CP"
      OOCFlag=""
    fi

    # build the cmd from config
    # $FILE in your original config is basically your DML script
    FILE=./exp.dml
    SYSDS_CMD=( $SYSDS_CMD_COMMON "$SYSDS_JAR" -f "$FILE" -exec singlenode $OOCFlag -args 10000 10000 1.0 "../data/" )

    # run and capture
    output=$("${SYSDS_CMD[@]}")
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
