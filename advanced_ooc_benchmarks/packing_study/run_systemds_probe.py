#!/usr/bin/env python3
"""Isolated real-OOC probe using an already prepared skinny SystemDS matrix."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SUITE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('cg', SUITE / 'vaex/test_memory_sweep.py')
CG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CG)
JAVA = '/opt/devcon/env/java/current/bin/java'
PYTHON = '/opt/devcon/env/python/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--jar', type=Path, required=True)
    parser.add_argument('--blocksize', type=int, required=True)
    parser.add_argument('--reader-buffer-kib', type=int, required=True)
    parser.add_argument('--source-bulk-mib', type=int, default=64)
    parser.add_argument('--workload', choices=('scan', 'kmeans', 'matmul', 'broadcast', 'join'), default='scan')
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    root = args.results.resolve()
    if root.exists():
        parser.error(f'results path exists: {root}')
    if not Path(str(args.input) + '.mtd').exists() or not args.jar.is_file():
        parser.error('prepared native input or SystemDS JAR is missing')
    os.chdir(SUITE)
    root.mkdir(parents=True)
    config = root / 'SystemDS-config.xml'
    settings = {
        'localtmpdir': root / 'tmp',
        'scratch': root / 'scratch',
        'defaultblocksize': args.blocksize,
        'ooc.io.direct': 'true',
        'ooc.io.reader.threads': 8,
        'ooc.io.reader.poolsize': 8,
        'ooc.io.reader.buffersize': args.reader_buffer_kib * 1024,
        'ooc.memory.broker.max': 268435456,
        'ooc.memory.prefetch.max': 67108864,
        'ooc.memory.cache.fraction.soft': 0.04,
        'ooc.memory.cache.fraction.hard': 0.06,
        'ooc.source.replay.memory': args.source_bulk_mib * 1024**2,
        'ooc.source.bulksize': args.source_bulk_mib * 1024**2,
        'ooc.source.replay.prefetch': 8,
        'ooc.materialized.partition.bytes': 0,
    }
    config.write_text('<root>\n' + ''.join(f'<sysds.{key}>{value}</sysds.{key}>\n'
                                           for key, value in settings.items()) + '</root>\n')
    if args.workload == 'scan':
        script = root / 'scan.dml'
        script.write_text('X=read($1); s=0; for(i in 1:$2) { s=s+sum(X > (i/100)); } print("scan checksum: "+s);\n')
        parameters = [args.input, args.iterations]
    elif args.workload == 'kmeans':
        script = SUITE / 'kmeans/systemds.dml'
        (root / 'outputs').mkdir()
        parameters = [args.input, 8, args.iterations, root / 'outputs/C', root / 'outputs/Y']
    else:
        (root / 'outputs').mkdir()
        script = root / 'kernel.dml'
        kernels = {
            'matmul': 'B=rand(rows=ncol(X), cols=8, min=0, max=1, seed=7); Y=X%*%B;',
            'broadcast': 'v=rowSums(X); Y=X*v;',
            'join': 'T=X>0.5; Y=X*T;',
        }
        script.write_text('X=read($1); ' + kernels[args.workload] +
                          ' write(Y,$2,format="binary"); print("kernel checksum: "+sum(Y));\n')
        parameters = [args.input, root / 'outputs/Y']
    cold = subprocess.run([PYTHON, str(SUITE / 'drop_caches.py'), str(args.input)],
                          capture_output=True, text=True)
    (root / 'drop-caches.log').write_text(cold.stdout + cold.stderr)
    if cold.returncode:
        raise RuntimeError('Cold-cache precondition failed')
    (root / 'java-tmp').mkdir()
    command = [JAVA, '-Xms4g', '-Xmx4g', '-XX:+UseG1GC', '-XX:G1HeapRegionSize=16m',
               '-XX:ActiveProcessorCount=8', '--add-modules=jdk.incubator.vector',
               f'-Djava.io.tmpdir={root}/java-tmp', '-jar', str(args.jar.resolve()),
               '-f', str(script), '-exec', 'singlenode', '-config', str(config),
               '-ooc', '-oocStats', '-stats', '-args', *map(str, parameters)]
    result = CG.run(Path('/sys/fs/cgroup/devcon'), root, 'systemds', 6 * 1024**3,
                    command, args.timeout, min_free_bytes=20 * 1024**3)
    log = (root / 'systemds.log').read_text(errors='replace')
    if result['status'] == 'ok' and ('An Error Occurred' in log or 'Exception in thread' in log):
        result['status'] = 'failed'
    result['checksum'] = re.findall(r'(?:scan checksum:|kernel checksum:|inertia=)\s*([^\n]+)', log)
    for label, pattern in {
        'source_scans': r'source scans:\s*(\d+) \(time ([\d.]+) sec, ([\d.]+) GB\)',
        'load_from_disk': r'loadFromDisk:\s*(\d+) \(time ([\d.]+) sec, ([\d.]+) GB\)',
        'evict_writes': r'evict writes:\s*(\d+) \(time ([\d.]+) sec, ([\d.]+) GB\)',
        'get_calls': r'get calls:\s*(\d+)',
        'put_calls': r'put calls:\s*(\d+)',
    }.items():
        result[label] = re.findall(pattern, log)[-1:]
    (root / 'probe.json').write_text(json.dumps(result, indent=2) + '\n')
    for leaf in ('tmp', 'scratch', 'java-tmp', 'outputs'):
        if (root / leaf).exists():
            shutil.rmtree(root / leaf)
    print(json.dumps({key: result[key] for key in ('status', 'wall_seconds', 'read_bytes',
                                                  'write_bytes', 'checksum', 'source_scans',
                                                  'load_from_disk', 'evict_writes')}))
    return result['status'] != 'ok'


if __name__ == '__main__':
    sys.exit(main())
