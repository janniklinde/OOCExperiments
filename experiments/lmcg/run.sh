#!/bin/bash
set -uo pipefail

# find all config files
confs=(../../sysds_*.conf)

# header
echo "mode,conf,rep1,rep2,rep3" > results.csv

modes=("cp" "ooc" "HYBRID" "SPARK")

# reasonable Spark defaults (override in sysds_*.conf if needed)
: "${SYSDS_SPARK_EXEC_MEM_FRAC:=0.85}"
: "${SYSDS_SPARK_DRIVER_MEM_FRAC:=0.85}"
: "${SYSDS_SPARK_MEM_FRACTION:=0.70}"
: "${SYSDS_SPARK_STORAGE_FRACTION:=0.60}"

get_xmx_mb() {
  local t n u
  for t in "$@"; do
    if [[ $t =~ ^-Xmx([0-9]+)([gGmMkK])$ ]]; then
      n="${BASH_REMATCH[1]}"
      u="${BASH_REMATCH[2]}"
      case "$u" in
        g|G) echo $((n * 1024)); return ;;
        m|M) echo "$n"; return ;;
        k|K) echo $((n / 1024)); return ;;
      esac
    fi
  done
  echo 1024
}

for conf in "${confs[@]}"; do
  # load this config
  # shellcheck disable=SC1090
  source "$conf"

  # expect SYSDS_CONFIG_NAME to be set by the conf
  cfg="${SYSDS_CONFIG_NAME:-$(basename "$conf" .conf)}"

  for mode in "${modes[@]}"; do
    # Common script args:
    # 1=rownum 2=colnum 3=sparsity 4=data_dir 5=optional write path
    run_args=(1000000 1000 1.0 "../../data/" "out")
    out_path=""
    if [[ ${#run_args[@]} -ge 5 ]]; then
      out_path="${run_args[4]}"
    fi

    row="$mode,$cfg"
    for rep in {1..1}; do
      start=$(date +%s%N)

      file=./exp.dml

      if [[ $mode == "HYBRID" || $mode == "SPARK" ]]; then
        if [[ $mode == "HYBRID" ]]; then
          exec_mode="hybrid"
          app_name="SystemDS-local-hybrid"
        else
          exec_mode="spark"
          app_name="SystemDS-local-spark"
        fi

        # Reuse per-config JVM sizing from SYSDS_CMD_COMMON
        # (strip leading "java" and trailing "-jar" from that array).
        hybrid_base_opts=("${SYSDS_CMD_COMMON[@]}")
        if [[ ${#hybrid_base_opts[@]} -gt 0 && ${hybrid_base_opts[0]} == "java" ]]; then
          hybrid_base_opts=("${hybrid_base_opts[@]:1}")
        fi
        if [[ ${#hybrid_base_opts[@]} -gt 0 && ${hybrid_base_opts[-1]} == "-jar" ]]; then
          unset 'hybrid_base_opts[-1]'
        fi

        xmx_mb="$(get_xmx_mb "${SYSDS_CMD_COMMON[@]}")"
        spark_exec_mb="$(awk -v x="$xmx_mb" -v f="$SYSDS_SPARK_EXEC_MEM_FRAC" 'BEGIN{
          v = x * f;
          if (v < 256) v = 256;
          printf "%d", v;
        }')"
        spark_driver_mb="$(awk -v x="$xmx_mb" -v f="$SYSDS_SPARK_DRIVER_MEM_FRAC" 'BEGIN{
          v = x * f;
          if (v < 256) v = 256;
          printf "%d", v;
        }')"
        spark_mem_fraction="$(awk -v f="$SYSDS_SPARK_MEM_FRACTION" 'BEGIN{
          if (f <= 0.01) f = 0.01;
          if (f >= 0.99) f = 0.99;
          printf "%.4f", f;
        }')"
        spark_storage_fraction="$(awk -v s="$SYSDS_SPARK_STORAGE_FRACTION" 'BEGIN{
          if (s <= 0.01) s = 0.01;
          if (s >= 0.99) s = 0.99;
          printf "%.4f", s;
        }')"

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
          -Dspark.app.name=$app_name
          "-Dspark.executor.memory=${spark_exec_mb}m"
          "-Dspark.driver.memory=${spark_driver_mb}m"
          "-Dspark.memory.fraction=${spark_mem_fraction}"
          "-Dspark.memory.storageFraction=${spark_storage_fraction}"
        )

        printf -v hybrid_opts '%s ' "${hybrid_all_opts[@]}"
        hybrid_opts="${hybrid_opts% }"

        cmd=( systemds "$file" -explain -args "${run_args[@]}" )

        printf '%q ' env SYSTEMDS_STANDALONE_OPTS="$hybrid_opts" SYSDS_DISTRIBUTED=0 SYSDS_EXEC_MODE="$exec_mode" "${cmd[@]}"
        echo
        if output=$(env SYSTEMDS_STANDALONE_OPTS="$hybrid_opts" SYSDS_DISTRIBUTED=0 SYSDS_EXEC_MODE="$exec_mode" "${cmd[@]}" 2>&1); then
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
          oocflags=(-ooc -oocStats)
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
        cmd+=( -explain -stats -args "${run_args[@]}" )

        printf '%q ' "${cmd[@]}"
        echo
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

      # Cleanup optional output target and its metadata sidecar.
      if [[ -n "$out_path" && "$out_path" != "/" && "$out_path" != "." ]]; then
        [[ -e "$out_path" ]] && rm -rf "$out_path"
        [[ -e "${out_path}.mtd" ]] && rm -f "${out_path}.mtd"
      fi

      rm -r ./tmp
      rm -r ./scratch_space
    done
    echo "$row" >> results.csv
  done
done
