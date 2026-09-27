#!/usr/bin/env python3
"""Run an isolated Twitter PageRank source-pack probe under cgroup v2."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

SUITE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('cg', SUITE / 'vaex/test_memory_sweep.py')
CG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CG)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--dangling', type=Path, required=True)
    parser.add_argument('--jar', type=Path, required=True)
    parser.add_argument('--partition-mib', type=int, required=True)
    parser.add_argument('--reader-threads', type=int, default=16)
    parser.add_argument('--source-bulk-mib', type=int, default=1024)
    parser.add_argument('--reader-buffer-mib', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--jfr-settings', type=Path)
    args = parser.parse_args()
    if args.results.exists():
        parser.error(f'results path exists: {args.results}')
    for path in (args.graph, args.dangling):
        if not Path(str(path) + '.mtd').is_file():
            parser.error(f'missing native input: {path}')
    graph_metadata = json.loads(Path(str(args.graph) + '.mtd').read_text())
    blocksize = int(graph_metadata['rows_in_block'])
    if min(args.reader_threads, args.source_bulk_mib, args.reader_buffer_mib) < 1:
        parser.error('reader threads, source bulk, and reader buffer must be positive')
    root = args.results.resolve()
    root.mkdir(parents=True)
    for name in ('tmp', 'scratch', 'java-tmp', 'outputs'):
        (root / name).mkdir()
    config = root / 'config.xml'
    config.write_text('<root>\n' + ''.join(f'<sysds.{key}>{value}</sysds.{key}>\n' for key, value in {
        'defaultblocksize': blocksize,
        'ooc.sparse.coo': 'true',
        'ooc.materialized.partition.bytes': args.partition_mib * 1024**2,
        'ooc.io.direct': 'true',
        'ooc.io.reader.buffersize': args.reader_buffer_mib * 1024**2,
        'ooc.io.reader.threads': args.reader_threads,
        'ooc.io.reader.poolsize': args.reader_threads,
        'ooc.source.replay.memory': 1024**3,
        'ooc.source.bulksize': args.source_bulk_mib * 1024**2,
        'ooc.source.replay.prefetch': 4,
        'ooc.memory.broker.max': 4 * 1024**3,
        'ooc.memory.prefetch.max': 1024**3,
        'localtmpdir': root / 'tmp',
        'scratch': root / 'scratch',
    }.items()) + '</root>\n')
    cold = subprocess.run([str(SUITE / 'drop_caches.py'), str(args.graph)],
                          capture_output=True, text=True)
    (root / 'drop-caches.log').write_text(cold.stdout + cold.stderr)
    if cold.returncode:
        raise RuntimeError(f'cold-cache precondition failed: {cold.stderr}')
    os.chdir(SUITE)
    command = ['/opt/devcon/env/java/current/bin/java', '-Xms12g', '-Xmx12g',
               '-XX:+UseG1GC', '-XX:G1HeapRegionSize=32m', '-XX:ActiveProcessorCount=16',
               f'-Djava.io.tmpdir={root}/java-tmp',
               '--add-modules=jdk.incubator.vector', '-jar', str(args.jar.resolve()),
               '-f', 'pagerank/systemds.dml', '-exec', 'singlenode', '-config', str(config),
               '-ooc', '-oocStats', '-stats', '-args', str(args.graph.resolve()),
               str(args.dangling.resolve()), '0.85', str(args.iterations),
               str(root / 'outputs/rank')]
    if args.jfr_settings:
        command[1:1] = [f'-XX:StartFlightRecording=filename={root}/source.jfr,'
                        f'settings={args.jfr_settings.resolve()},dumponexit=true']
    result = CG.run(Path('/sys/fs/cgroup/devcon'), root, 'pagerank', 16 * 1024**3,
                    command, args.timeout, min_free_bytes=30 * 1024**3)
    log = (root / 'pagerank.log').read_text(errors='replace')
    if result['status'] == 'ok' and ('An Error Occurred' in log or
                                     'org.apache.sysds.runtime.DMLRuntimeException:' in log or
                                     not (root / 'outputs/rank.mtd').is_file()):
        result['status'] = 'failed'
    result['source_scans'] = re.findall(r'source scans:\s*(\d+) \(time ([\d.]+) sec, ([\d.]+) GB\)', log)
    result['reloads'] = re.findall(r'loadFromDisk:\s*(\d+) \(time ([\d.]+) sec, ([\d.]+) GB\)', log)
    result['puts'] = re.findall(r'put calls:\s*(\d+)', log)
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'wall_seconds', 'read_bytes',
                                                  'write_bytes', 'source_scans', 'reloads', 'puts')}))
    return result['status'] != 'ok'


if __name__ == '__main__':
    raise SystemExit(main())
