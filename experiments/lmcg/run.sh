#!/bin/bash
set -uo pipefail

# find all config files
confs=(../../sysds_*.conf)

# header
echo "mode,conf,rep1,rep2,rep3" > results.csv

modes=("cp" "ooc" "HYBRID")

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

      file=./exp.dml

      if [[ $mode == "HYBRID" ]]; then
        # Reuse per-config JVM sizing from SYSDS_CMD_COMMON
        # (strip leading "java" and trailing "-jar" from that array).
        hybrid_base_opts=("${SYSDS_CMD_COMMON[@]}")
        if [[ ${#hybrid_base_opts[@]} -gt 0 && ${hybrid_base_opts[0]} == "java" ]]; then
          hybrid_base_opts=("${hybrid_base_opts[@]:1}")
        fi
        if [[ ${#hybrid_base_opts[@]} -gt 0 && ${hybrid_base_opts[-1]} == "-jar" ]]; then
          unset 'hybrid_base_opts[-1]'
        fi

        hybrid_all_opts=(
          "${hybrid_base_opts[@]}"
          --add-opens=java.base/java.nio=ALL-UNNAMED
          --add-opens=java.base/java.io=ALL-UNNAMED
          --add-opens=java.base/java.util=ALL-UNNAMED
          --add-opens=java.base/java.lang=ALL-UNNAMED
          --add-opens=java.base/java.lang.ref=ALL-UNNAMED
          --add-opens=java.base/java.lang.invoke=ALL-UNNAMED
          --add-opens=java.base/java.util.concurrent=ALL-UNNAMED
          --add-opens=java.base/sun.nio.ch=ALL-UNNAMED
          -Dspark.master=local[*]
          -Dspark.app.name=SystemDS-local
        )

        printf -v hybrid_opts '%s ' "${hybrid_all_opts[@]}"
        hybrid_opts="${hybrid_opts% }"

        cmd=( systemds "$file" -explain -args 1000000 1000 1.0 "../../data/" )

        printf 'RUN CMD (%s %s): SYSTEMDS_STANDALONE_OPTS="%s" SYSDS_DISTRIBUTED=0 SYSDS_EXEC_MODE=hybrid %q ' \
          "$cfg" "$mode" "$hybrid_opts" "${cmd[@]}"
        echo
        if output=$(SYSTEMDS_STANDALONE_OPTS="$hybrid_opts" SYSDS_DISTRIBUTED=0 SYSDS_EXEC_MODE=hybrid "${cmd[@]}" 2>&1); then
          echo "$output"
          exec_time=$(echo "$output" | grep -oP 'Total execution time:\s*\K[0-9.]+')
          result=$(echo "$output" | grep -oP 'Result:\s*\K[-+0-9.eE]+')
          [[ -z $exec_time ]] && exec_time="nan"
          [[ -z $result ]] && result="nan"
          status="ok"
        else
          echo "$output" >&2
          echo "Run failed (cfg: $cfg mode: $mode rep: $rep); storing nan" >&2
          exec_time="nan"
          result="nan"
          status="failed"
        fi
      else
        # pick jar + optional flag
        if [[ $mode == "ooc" ]]; then
          jar="$SYSDS_JAR_OOC"
          oocflags=(-ooc -oocStats -oocLogEvents)
        else
          jar="$SYSDS_JAR_CP"
          oocflags=()
        fi

        cmd=(
          "${SYSDS_CMD_COMMON[@]}"
          "$jar"
          -f "$file"
          -exec singlenode
          -config ./config.xml
        )

        if ((${#oocflags[@]})); then
          cmd+=("${oocflags[@]}")
        fi

        # args (keep as in your script)
        cmd+=( -explain -stats -args 1000000 1000 1.0 "../../data/" )

        printf 'RUN CMD (%s %s): %q ' "$cfg" "$mode" "${cmd[@]}"; echo
        if output=$("${cmd[@]}" 2>&1); then
          echo "$output"
          exec_time=$(echo "$output" | grep -oP 'Total execution time:\s*\K[0-9.]+')
          result=$(echo "$output" | grep -oP 'Result:\s*\K[-+0-9.eE]+')
          [[ -z $exec_time ]] && exec_time="nan"
          [[ -z $result ]] && result="nan"
          status="ok"
        else
          echo "$output" >&2
          echo "Run failed (cfg: $cfg mode: $mode rep: $rep); storing nan" >&2
          exec_time="nan"
          result="nan"
          status="failed"
        fi
      fi

      end=$(date +%s%N)
      dur_ms=$(( (end - start) / 1000000 ))

      row="$row,$exec_time"
      echo "ExecTime: $exec_time ms(raw: $dur_ms) [$status]"
      echo "Result: $result"
      rm -r ./tmp
      rm -r ./scratch_space
    done
    echo "$row" >> results.csv
  done
done
