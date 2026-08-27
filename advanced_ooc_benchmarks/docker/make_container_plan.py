#!/usr/bin/env python3
"""Rewrite a benchmark plan's host-specific paths for the container.

Only five lines of a plan are tied to a particular machine: `root`, the three
`tools` entries, and `environment.SPARK_HOME`. Everything else derives from
`${plan.root}` and `${tools.*}`, so a targeted line rewrite - rather than a YAML
round-trip - produces a container plan that is byte-identical to the source plan
apart from those paths, comments included.

Usage: make_container_plan.py [SOURCE_PLAN] [-o TARGET_PLAN]
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def substitute(text, pattern, replacement, key, required):
    """Replace the single line matching `pattern`, or fail loudly."""
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if not matches:
        if required:
            raise SystemExit(f"Plan has no {key} entry to rewrite")
        return text, False
    if len(matches) > 1:
        raise SystemExit(f"Plan has {len(matches)} {key} entries; expected exactly one")
    start, end = matches[0].span()
    return text[:start] + replacement + text[end:], True


def spark_home(python):
    """Locate the pyspark package of the container interpreter, if installed."""
    try:
        result = subprocess.run(
            [python, "-c", "import os, pyspark; print(os.path.dirname(pyspark.__file__))"],
            text=True, capture_output=True)
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def main(argv=None):
    plan_dir = Path(os.environ.get("BENCH_PLAN_DIR", Path(__file__).resolve().parent.parent))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?",
                        default=os.environ.get("BENCHMARK_SOURCE_PLAN",
                                               str(plan_dir / "benchmark-plan.yaml")))
    parser.add_argument("-o", "--output", default=os.environ.get("BENCHMARK_PLAN", ""))
    parser.add_argument("--root", default=os.environ.get("BENCH_CONTAINER_ROOT", "/bench/data"))
    parser.add_argument("--python", default=os.environ.get("BENCH_CONTAINER_PYTHON",
                                                           "/opt/bench-venv/bin/python"))
    parser.add_argument("--systemds-jar", default=os.environ.get("BENCH_CONTAINER_JAR",
                                                                 "/opt/systemds/SystemDS.jar"))
    parser.add_argument("--spark-submit", default=os.environ.get(
        "BENCH_CONTAINER_SPARK_SUBMIT", "/opt/bench-venv/bin/spark-submit"))
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    if not source.is_file():
        raise SystemExit(f"Missing source plan: {source}")
    target = Path(args.output).resolve() if args.output else \
        source.with_name(source.stem + ".container" + source.suffix)

    text = source.read_text(encoding="utf-8")
    text, _ = substitute(text, r"^root:[ \t]*\S.*$", f"root: {args.root}",
                         "root", required=True)
    text, _ = substitute(text, r"^  python:[ \t]*\S.*$", f"  python: {args.python}",
                         "tools.python", required=True)
    text, _ = substitute(text, r"^  systemds_jar:[ \t]*\S.*$",
                         f"  systemds_jar: {args.systemds_jar}",
                         "tools.systemds_jar", required=True)
    text, _ = substitute(text, r"^  spark_submit:[ \t]*\S.*$",
                         f"  spark_submit: {args.spark_submit}",
                         "tools.spark_submit", required=False)
    home = spark_home(args.python)
    if home:
        text, _ = substitute(text, r"^  SPARK_HOME:[ \t]*\S.*$", f"  SPARK_HOME: {home}",
                             "environment.SPARK_HOME", required=False)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"wrote container plan {target} (from {source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
