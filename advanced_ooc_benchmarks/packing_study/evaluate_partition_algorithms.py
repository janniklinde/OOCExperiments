#!/usr/bin/env python3
"""Compare partition-aware OOC execution across benchmark algorithm kernels."""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SUITE = Path(__file__).resolve().parents[1]
JAVA = Path("/opt/devcon/env/java/current/bin/java")
JAR = Path("/workspace/systemds/target/SystemDS.jar")
DATA = Path("/workspace/data_dir")
SPEC = importlib.util.spec_from_file_location("cgroup", SUITE / "vaex/test_memory_sweep.py")
CGROUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CGROUP)

SCRIPTS = {
    "kmeans": '''source("kmeans/implementation.dml") as impl;
X=read($1); [C,Y,v]=impl::m_kmeans_fixed(X=X,k=8,iterations=1); print("value="+v);
''',
    "lmcg": '''source("lmcg/implementation.dml") as impl;
X=read($1); y=ifelse(X[,1]>0,1,-1); [b,v]=impl::m_lmcg_fixed(X=X,y=y,reg=1e-7,iterations=1,tolerance=0); print("value="+v);
''',
    "l2svm": '''source("l2svm/implementation.dml") as impl;
X=read($1); y=ifelse(X[,1]>0,1,-1); w=impl::m_l2svm(X=X,Y=y,intercept=FALSE,epsilon=1e-6,reg=1,maxIterations=1,maxii=2,verbose=FALSE); print("value="+sqrt(sum(w^2)));
''',
    "multilogreg": '''source("multilogreg/implementation.dml") as impl;
X=read($1); y=1+(X[,1]>0); B=impl::m_multiLogReg(X=X,Y=y,icpt=0,tol=1e-6,reg=1,maxi=1,maxii=2,verbose=FALSE); print("value="+sqrt(sum(B^2)));
''',
    "gnmf": '''source("gnmf/implementation.dml") as impl;
X=abs(read($1)); [W,H]=impl::m_gnmf(X=X,rank=8,iterations=1,epsilon=1e-8,seed=7); print("value="+(sum(W)+sum(H)));
''',
    "randomized_svd": '''X=read($1); k=min(8,ncol(X)); O=rand(rows=ncol(X),cols=k,pdf="normal",seed=7); Y=X%*%O; G=t(Y)%*%Y; L=cholesky(G); Q=Y%*%t(inv(L)); B=t(Q)%*%X; [U,S,V]=svd(B); print("value="+sum(S));
''',
    "pca": '''source("pca/implementation.dml") as impl;
X=read($1); [S,C,E]=impl::m_pca_covariance(X=X,k=min(8,ncol(X))); print("value="+sum(E));
''',
}


def config(case, partition_bytes):
    path = case / "config.xml"
    path.write_text(f'''<root>
<sysds.localtmpdir>{case}/tmp</sysds.localtmpdir>
<sysds.scratch>{case}/scratch</sysds.scratch>
<sysds.defaultblocksize>1000</sysds.defaultblocksize>
<sysds.ooc.io.direct>true</sysds.ooc.io.direct>
<sysds.ooc.io.reader.buffersize>8388608</sysds.ooc.io.reader.buffersize>
<sysds.ooc.io.reader.threads>8</sysds.ooc.io.reader.threads>
<sysds.ooc.io.reader.poolsize>8</sysds.ooc.io.reader.poolsize>
<sysds.ooc.materialized.partition.bytes>{partition_bytes}</sysds.ooc.materialized.partition.bytes>
<sysds.ooc.memory.broker.max>536870912</sysds.ooc.memory.broker.max>
<sysds.ooc.memory.prefetch.max>134217728</sysds.ooc.memory.prefetch.max>
<sysds.ooc.memory.cache.fraction.soft>0.35</sysds.ooc.memory.cache.fraction.soft>
<sysds.ooc.memory.cache.fraction.hard>0.45</sysds.ooc.memory.cache.fraction.hard>
<sysds.ooc.source.replay.memory>134217728</sysds.ooc.source.replay.memory>
<sysds.ooc.source.bulksize>134217728</sysds.ooc.source.bulksize>
<sysds.ooc.source.replay.prefetch>8</sysds.ooc.source.replay.prefetch>
</root>''')
    return path


def extract(pattern, log, cast=float):
    values = re.findall(pattern, log)
    return cast(values[-1]) if values else None


def run(root, dataset_name, dataset, algorithm, partition_bytes, heap_gib, memory_gib, timeout):
    mode = "partition" if partition_bytes else "ordinary"
    case = root / dataset_name / algorithm / mode
    case.mkdir(parents=True)
    (case / "java-tmp").mkdir()
    script = case / "run.dml"
    script.write_text(SCRIPTS[algorithm])
    drop = subprocess.run([sys.executable, str(SUITE / "drop_caches.py"), str(dataset)],
                          capture_output=True, text=True)
    (case / "drop-caches.log").write_text(drop.stdout + drop.stderr)
    if drop.returncode:
        raise RuntimeError(f"cold-cache precondition failed for {dataset}")
    cmd = [str(JAVA), f"-Xms{heap_gib}g", f"-Xmx{heap_gib}g", "-XX:+UseG1GC",
           "-XX:G1HeapRegionSize=16m", "-XX:ActiveProcessorCount=8",
           "--add-modules=jdk.incubator.vector", f"-Djava.io.tmpdir={case}/java-tmp",
           "-Dsysds.ooc.watchdog.primitives=true",
           "-jar", str(JAR), "-f", str(script), "-exec", "singlenode",
           "-config", str(config(case, partition_bytes)), "-ooc", "-oocStats", "-stats",
           "-args", str(dataset)]
    print(f"Starting {dataset_name}/{algorithm}/{mode}", flush=True)
    result = CGROUP.run(Path("/sys/fs/cgroup/devcon"), case, "systemds",
                        memory_gib * 1024**3, cmd, timeout, min_free_bytes=12*1024**3)
    log = (case / "systemds.log").read_text(errors="replace")
    if result["status"] == "ok" and ("An Error Occurred" in log or "Exception in thread" in log):
        result["status"] = "failed"
    result.update(dataset=dataset_name, algorithm=algorithm, mode=mode,
                  partition_bytes=partition_bytes,
                  value=extract(r"value=([^\n]+)", log),
                  source_gb=extract(r"source scans:\s*\d+ \(time [\d.]+ sec, ([\d.]+) GB\)", log),
                  load_gb=extract(r"loadFromDisk:\s*\d+ \(time [\d.]+ sec, ([\d.]+) GB\)", log),
                  write_gb=extract(r"evict writes:\s*\d+ \(time [\d.]+ sec, ([\d.]+) GB\)", log),
                  tasks=extract(r"reader tasks:\s*(\d+)", log, int),
                  errors=[line[:500] for line in log.splitlines()
                          if any(x in line for x in ("An Error", "Exception", "OutOfMemory"))][:8])
    (case / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    for leaf in ("tmp", "scratch", "java-tmp"):
        target = case / leaf
        if target.exists():
            shutil.rmtree(target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--algorithms", nargs="+", choices=sorted(SCRIPTS), default=sorted(SCRIPTS))
    parser.add_argument("--modes", nargs="+", choices=("ordinary", "partition"), default=("ordinary", "partition"))
    parser.add_argument("--partition-kib", type=int, default=512)
    parser.add_argument("--heap-gib", type=int, default=4)
    parser.add_argument("--memory-gib", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    os.chdir(SUITE)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root = args.results or DATA / "partition-algorithm-study" / stamp
    root.mkdir(parents=True)
    summary = {"jar": str(JAR), "datasets": args.dataset, "heap_gib": args.heap_gib,
               "memory_gib": args.memory_gib, "partition_kib": args.partition_kib, "results": []}
    for dataset_name, raw_path in args.dataset:
        dataset = Path(raw_path).resolve()
        if not Path(str(dataset) + ".mtd").exists():
            parser.error(f"missing metadata for {dataset}")
        for algorithm in args.algorithms:
            for mode in args.modes:
                result = run(root, dataset_name, dataset, algorithm,
                             args.partition_kib * 1024 if mode == "partition" else 0,
                             args.heap_gib, args.memory_gib, args.timeout)
                summary["results"].append(result)
                (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps({key: result.get(key) for key in
                                  ("dataset", "algorithm", "mode", "status", "wall_seconds",
                                   "read_bytes", "write_bytes", "source_gb", "load_gb", "write_gb",
                                   "tasks", "value")}), flush=True)
    print(root)


if __name__ == "__main__":
    main()
