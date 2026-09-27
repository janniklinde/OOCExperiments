#!/usr/bin/env python3
"""Run the isolated row-major source and eviction-time packing prototypes."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
JAVA = Path('/opt/devcon/env/java/current/bin')
SPEC = importlib.util.spec_from_file_location('cg', SUITE / 'vaex/test_memory_sweep.py')
CG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CG)


def cases(geometry):
    small = [
        ('small80-source-tile', 2_000_000, 10, 1000, 80, 'sequential', 'source'),
        ('small80-source-512', 2_000_000, 10, 1000, 512, 'sequential', 'source'),
        ('small80-source-2048', 2_000_000, 10, 1000, 2048, 'sequential', 'source'),
        ('small80-reactive-seq', 2_000_000, 10, 1000, 512, 'sequential', 'reactive'),
        ('small80-reactive-window', 2_000_000, 10, 1000, 512, 'window', 'reactive'),
        ('small80-reactive-global', 2_000_000, 10, 1000, 512, 'global', 'reactive'),
        ('small8-source-tile', 2_000_000, 10, 100, 8, 'sequential', 'source'),
        ('small8-source-512', 2_000_000, 10, 100, 512, 'sequential', 'source'),
        ('small8-reactive-seq', 2_000_000, 10, 100, 512, 'sequential', 'reactive'),
        ('small8-reactive-window', 2_000_000, 10, 100, 512, 'window', 'reactive'),
        ('small8-reactive-global', 2_000_000, 10, 100, 512, 'global', 'reactive'),
    ]
    large = [
        ('large80-source-tile', 20_000_000, 10, 1000, 80, 'sequential', 'source'),
        ('large80-source-512', 20_000_000, 10, 1000, 512, 'sequential', 'source'),
        ('large80-reactive-seq', 20_000_000, 10, 1000, 512, 'sequential', 'reactive'),
    ]
    huge = [
        ('huge80-source-tile', 100_000_000, 10, 1000, 80, 'sequential', 'source'),
        ('huge80-source-512', 100_000_000, 10, 1000, 512, 'sequential', 'source'),
    ]
    return {'small': small, 'large': large, 'huge': huge, 'all': small + large}[geometry]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--geometry', choices=['small', 'large', 'huge', 'all'], default='all')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--read-repetitions', type=int, default=1)
    parser.add_argument('--hot-mib', type=int, default=8)
    parser.add_argument('--gap', type=int, default=32)
    parser.add_argument('--case', action='append')
    parser.add_argument('--keep-packs', action='store_true')
    parser.add_argument('--read-only', action='store_true',
                        help='skip in-memory operators (required for the 8 GB huge cases)')
    args = parser.parse_args()
    selected = [case for case in cases(args.geometry) if not args.case or case[0] in args.case]
    if args.case and set(args.case) - {case[0] for case in selected}:
        parser.error('unknown --case for the selected geometry')
    if args.geometry == 'huge' and not args.read_only:
        parser.error('--geometry huge requires --read-only under the 6 GiB cgroup')
    if not selected:
        parser.error('no cases selected')
    root = args.results.resolve()
    root.mkdir(parents=True, exist_ok=False)
    classes = root / 'classes'
    classes.mkdir()
    subprocess.run([str(JAVA / 'javac'), '-d', str(classes), str(HERE / 'ReactivePackExperiment.java')],
                   check=True)
    for name, rows, cols, tile_rows, target_kib, arrival, kind in selected:
        directory = root / name
        directory.mkdir()
        command = [str(JAVA / 'java'), '-Xms2g', '-Xmx4g', '-XX:ActiveProcessorCount=8',
                   '-cp', str(classes), 'ReactivePackExperiment', str(directory), str(rows), str(cols),
                   str(tile_rows), str(target_kib), str(args.hot_mib), str(args.gap), arrival, '8',
                   str(args.repetitions), kind, str(args.read_repetitions), str(args.read_only).lower()]
        print(f'Starting {name}', flush=True)
        result = CG.run(Path('/sys/fs/cgroup/devcon'), directory, 'packing', 6 * 1024**3,
                        command, 300 if args.geometry == 'huge' else 120,
                        min_free_bytes=20 * 1024**3)
        if not args.keep_packs:
            for name_in_case in ('x.pack', 'b.pack'):
                (directory / name_in_case).unlink(missing_ok=True)
        if result['status'] != 'ok':
            print(f'Stopping after {name}: {result["status"]}', file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
