#!/bin/bash
set -euo pipefail

# find all config files
confs=(../../sysds_*.conf)

# header
echo "mode,conf,rep1,rep2,rep3" > results.csv

modes=("cp" "ooc")

for conf in "${confs[@]}"; do
  # load this config
  # shellcheck disable=SC1090
  source "$conf"

  # expect SYSDS_CONFIG_NAME to be set by the conf
  cfg="${SYSDS_CONFIG_NAME:-$(basename "$conf" .conf)}"

  for mode in "${modes[@]}"; do
    row="$mode,$cfg"
    for rep in {1..1}; do
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

      cmd=(
        "${SYSDS_CMD_COMMON[@]}"
        "$jar"
        -f "$file"
        -exec singlenode
      )

      if [[ -n $oocflag ]]; then
        cmd+=("$oocflag")
      fi

      # args (keep as in your script)
      cmd+=( -explain hops -stats -args 1000000 1000 1.0 "../../data/" )

      printf 'RUN CMD (%s %s): %q ' "$cfg" "$mode" "${cmd[@]}"; echo
      output=$("${cmd[@]}")

      echo "$output"

      exec_time=$(echo "$output" | grep -oP 'Total execution time:\s*\K[0-9.]+')
      result=$(echo "$output" | grep -oP 'Result:\s*\K[-+0-9.eE]+')

      end=$(date +%s%N)
      dur_ms=$(( (end - start) / 1000000 ))

      row="$row,$exec_time"
      echo "ExecTime: $exec_time ms(raw: $dur_ms)"
      echo "Result: $result"
    done
    echo "$row" >> results.csv
  done
done
