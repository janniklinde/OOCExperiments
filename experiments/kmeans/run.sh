#!/bin/bash
set -uo pipefail

# find all config files
confs=(../../sysds_*.conf)

# header
echo "mode,conf,rep1,rep2,rep3" > results.csv

modes=("cp" "ooc" "HYBRID" "SPARK")

# reasonable Spark defaults (override in sysds_*.conf if needed)
: "${SYSDS_SPARK_EXEC_MEM_FRAC:=0.80}"
: "${SYSDS_SPARK_DRIVER_MEM_FRAC:=0.80}"
: "${SYSDS_SPARK_MEM_FRACTION:=0.70}"
: "${SYSDS_SPARK_STORAGE_FRACTION:=0.60}"
: "${SYSDS_LOCAL_BUDGET_FRAC:=0.20}"
: "${SYSDS_LOCAL_BUDGET_MB:=}"
: "${SYSDS_RUN_TIMEOUT_SEC:=200}"
: "${SYSDS_SKIP_TOKEN:=skip}"

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

mem_value_to_mb() {
  local v n u
  v="${1:-}"
  if [[ $v =~ ^([0-9]+)([gGmMkK])?$ ]]; then
    n="${BASH_REMATCH[1]}"
    u="${BASH_REMATCH[2]:-m}"
    case "$u" in
      g|G) echo $((n * 1024)); return 0 ;;
      m|M) echo "$n"; return 0 ;;
      k|K) echo $((n / 1024)); return 0 ;;
    esac
  fi
  return 1
}

calc_spark_local_budgets_mb() {
  local xmx_mb local_target_mb
  xmx_mb="$1"

  if [[ -n ${SYSDS_LOCAL_BUDGET_MB:-} ]]; then
    local_target_mb="$(mem_value_to_mb "$SYSDS_LOCAL_BUDGET_MB" || true)"
  fi
  if [[ -z ${local_target_mb:-} ]]; then
    local_target_mb="$(awk -v x="$xmx_mb" -v f="$SYSDS_LOCAL_BUDGET_FRAC" 'BEGIN{
      v = x * f;
      if (v < 256) v = 256;
      printf "%d", v;
    }')"
  fi

  read -r spark_exec_mb spark_driver_mb sysds_local_budget_mb <<<"$(awk -v x="$xmx_mb" -v ef="$SYSDS_SPARK_EXEC_MEM_FRAC" -v df="$SYSDS_SPARK_DRIVER_MEM_FRAC" -v lb="$local_target_mb" 'BEGIN{
    minE=1024; minD=1024; minL=256;
    hardE=256; hardD=256; hardL=128;

    e = int(x * ef);
    d = int(x * df);
    l = int(lb);

    if (e < minE) e = minE;
    if (d < minD) d = minD;
    if (l < minL) l = minL;

    sum = e + d + l;
    if (sum > x) {
      scale = x / sum;
      e = int(e * scale);
      d = int(d * scale);
      l = int(l * scale);
      if (e < minE) e = minE;
      if (d < minD) d = minD;
      if (l < minL) l = minL;
    }

    sum = e + d + l;
    while (sum > x) {
      if (l > minL) { l--; sum--; continue; }
      if (e >= d && e > minE) { e--; sum--; continue; }
      if (d > minD) { d--; sum--; continue; }
      if (e > minE) { e--; sum--; continue; }
      if (l > hardL) { l--; sum--; continue; }
      if (e >= d && e > hardE) { e--; sum--; continue; }
      if (d > hardD) { d--; sum--; continue; }
      break;
    }

    if (e < hardE) e = hardE;
    if (d < hardD) d = hardD;
    if (l < hardL) l = hardL;

    if (e + d + l > x) {
      l = x - e - d;
      if (l < hardL) l = hardL;
    }

    printf "%d %d %d", e, d, l;
  }')"
}

SKIP_HINT_SHOWN=0
RUN_OUTPUT=""
RUN_OUTCOME=""

run_with_timeout_skip() {
  local log_file pid start_s now_s input rc
  local -a run_cmd
  run_cmd=("$@")
  RUN_OUTPUT=""
  RUN_OUTCOME="failed"

  log_file="$(mktemp)"

  if [[ -r /dev/tty && $SKIP_HINT_SHOWN -eq 0 ]]; then
    echo "Run control: type '${SYSDS_SKIP_TOKEN}' + Enter to skip a run (timeout: ${SYSDS_RUN_TIMEOUT_SEC}s)." > /dev/tty
    SKIP_HINT_SHOWN=1
  fi

  if command -v setsid >/dev/null 2>&1; then
    setsid "${run_cmd[@]}" >"$log_file" 2>&1 &
  else
    "${run_cmd[@]}" >"$log_file" 2>&1 &
  fi
  pid=$!
  start_s=$(date +%s)

  while kill -0 "$pid" 2>/dev/null; do
    now_s=$(date +%s)
    if (( now_s - start_s >= SYSDS_RUN_TIMEOUT_SEC )); then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      sleep 1
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      RUN_OUTPUT="$(cat "$log_file")"
      rm -f "$log_file"
      RUN_OUTCOME="timeout"
      return 1
    fi

    if [[ -r /dev/tty ]]; then
      if IFS= read -r -t 1 input < /dev/tty; then
        if [[ "${input,,}" == "${SYSDS_SKIP_TOKEN,,}" ]]; then
          kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
          sleep 1
          kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
          wait "$pid" 2>/dev/null || true
          RUN_OUTPUT="$(cat "$log_file")"
          rm -f "$log_file"
          RUN_OUTCOME="skipped"
          return 1
        fi
      fi
    else
      sleep 1
    fi
  done

  if wait "$pid"; then
    rc=0
  else
    rc=$?
  fi
  RUN_OUTPUT="$(cat "$log_file")"
  rm -f "$log_file"

  if (( rc == 0 )); then
    RUN_OUTCOME="ok"
    return 0
  fi
  RUN_OUTCOME="failed"
  return 1
}

run_output_has_error_marker() {
  local output_text="$1"
  grep -qE 'An Error Occurred|DMLException' <<<"$output_text"
}

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
      config_args=()
      if [[ -f ./config.xml ]]; then
        config_args=(-config ./config.xml)
      fi

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
        calc_spark_local_budgets_mb "$xmx_mb"
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

        cmd=( systemds "$file" "${config_args[@]}" -local_budget "${sysds_local_budget_mb}m" -stats -explain )

        printf '%q ' env SYSTEMDS_STANDALONE_OPTS="$hybrid_opts" SYSDS_DISTRIBUTED=0 SYSDS_EXEC_MODE="$exec_mode" "${cmd[@]}"
        echo
        run_with_timeout_skip env SYSTEMDS_STANDALONE_OPTS="$hybrid_opts" SYSDS_DISTRIBUTED=0 SYSDS_EXEC_MODE="$exec_mode" "${cmd[@]}"
        output="$RUN_OUTPUT"
        reported_error=0
        if [[ $RUN_OUTCOME == "ok" ]] && run_output_has_error_marker "$output"; then
          RUN_OUTCOME="failed"
          reported_error=1
        fi
        if [[ $RUN_OUTCOME == "ok" ]]; then
          echo "$output"
          exec_time=$(echo "$output" | grep -oP 'Total execution time:\s*\K[0-9.]+')
          result=$(echo "$output" | grep -oP 'Result:\s*\K[-+0-9.eE]+')
          [[ -z $exec_time ]] && exec_time="nan"
          [[ -z $result ]] && result="nan"
          status="ok"
        else
          [[ -n $output ]] && echo "$output" >&2
          if (( reported_error )); then
            echo "Run reported DMLException/An Error Occurred (cfg: $cfg mode: $mode rep: $rep); storing nan" >&2
          elif [[ $RUN_OUTCOME == "timeout" ]]; then
            echo "Run timed out after ${SYSDS_RUN_TIMEOUT_SEC}s (cfg: $cfg mode: $mode rep: $rep); storing nan" >&2
          elif [[ $RUN_OUTCOME == "skipped" ]]; then
            echo "Run skipped by user (cfg: $cfg mode: $mode rep: $rep); storing nan" >&2
          else
            echo "Run failed (cfg: $cfg mode: $mode rep: $rep); storing nan" >&2
          fi
          exec_time="nan"
          result="nan"
          status="$RUN_OUTCOME"
        fi
      else
        # pick jar + optional flags
        if [[ $mode == "ooc" ]]; then
          jar="$SYSDS_JAR_OOC"
          oocflags=(-ooc)
        else
          jar="$SYSDS_JAR_CP"
          oocflags=()
        fi

        cmd=(
          "${SYSDS_CMD_COMMON[@]}"
          "$jar"
          -f "$file"
          -exec singlenode
          "${config_args[@]}"
        )

        if ((${#oocflags[@]})); then
          cmd+=("${oocflags[@]}")
        fi

        printf '%q ' "${cmd[@]}"
        echo
        run_with_timeout_skip "${cmd[@]}"
        output="$RUN_OUTPUT"
        reported_error=0
        if [[ $RUN_OUTCOME == "ok" ]] && run_output_has_error_marker "$output"; then
          RUN_OUTCOME="failed"
          reported_error=1
        fi
        if [[ $RUN_OUTCOME == "ok" ]]; then
          echo "$output"
          exec_time=$(echo "$output" | grep -oP 'Total execution time:\s*\K[0-9.]+')
          result=$(echo "$output" | grep -oP 'Result:\s*\K[-+0-9.eE]+')
          [[ -z $exec_time ]] && exec_time="nan"
          [[ -z $result ]] && result="nan"
          status="ok"
        else
          [[ -n $output ]] && echo "$output" >&2
          if (( reported_error )); then
            echo "Run reported DMLException/An Error Occurred (cfg: $cfg mode: $mode rep: $rep); storing nan" >&2
          elif [[ $RUN_OUTCOME == "timeout" ]]; then
            echo "Run timed out after ${SYSDS_RUN_TIMEOUT_SEC}s (cfg: $cfg mode: $mode rep: $rep); storing nan" >&2
          elif [[ $RUN_OUTCOME == "skipped" ]]; then
            echo "Run skipped by user (cfg: $cfg mode: $mode rep: $rep); storing nan" >&2
          else
            echo "Run failed (cfg: $cfg mode: $mode rep: $rep); storing nan" >&2
          fi
          exec_time="nan"
          result="nan"
          status="$RUN_OUTCOME"
        fi
      fi

      end=$(date +%s%N)
      dur_ms=$(( (end - start) / 1000000 ))

      row="$row,$exec_time"
      echo "ExecTime: $exec_time ms(raw: $dur_ms) [$status]"
      echo "Result: $result"

      if [[ -e ./out ]]; then
        rm -rf ./out
      fi
      if [[ -e ./out.mtd ]]; then
        rm -f ./out.mtd
      fi
      rm -rf ./tmp
      rm -rf ./scratch_space
    done
    echo "$row" >> results.csv
  done
done
