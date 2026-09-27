#!/usr/bin/env python3
import csv
import io
import statistics
import sys
from pathlib import Path

records = []
for directory in sys.argv[1:]:
    for path in sorted(Path(directory).glob('*.log')):
        text = path.read_text()
        lines = text.splitlines()
        if 'Exception in thread' in text or 'AssertionError' in text:
            print(f'FAILED: {path}', file=sys.stderr)
            continue
        header = next((i for i, line in enumerate(lines) if line.startswith('phase,algorithm,')), None)
        if header is None:
            continue
        cols = next((int(line.split('=')[1].split()[0]) for line in lines if line.startswith('# columns=')), 2)
        for row in csv.DictReader(io.StringIO('\n'.join(line for line in lines[header:] if not line.startswith('#')))):
            if row['phase'] not in ('prepare', 'reorder', 'iteration'):
                continue
            row['cols'] = cols
            row['source'] = str(path)
            records.append(row)

reference = {}
groups = {}
for row in records:
    key = (row['algorithm'], row['rows'], row['cols'], row['rank'], row['cache_bytes'], row['policy'])
    groups.setdefault(key, []).append(row)
    if row['phase'] == 'iteration':
        validation = (row['algorithm'], row['rows'], row['cols'], row['rank'], row['iteration'])
        value = float(row['checksum'])
        old = reference.setdefault(validation, value)
        if abs(value - old) > 1e-8 * max(1, abs(old)):
            raise ValueError(f'Checksum mismatch: {validation}: {value} vs {old}')

print('| Algorithm | Shape | Rank/k | Cache MiB | Policy | Iteration s | CPU s | Reload GB | Write GB | Tasks | Prep s | Reorder s |')
print('|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|')
for key, group in groups.items():
    algo, rows, cols, rank, cache, policy = key
    iterations = [row for row in group if row['phase'] == 'iteration']
    if len(iterations) < 3:
        print(f'INCOMPLETE: {key}', file=sys.stderr)
        continue
    def median(field):
        return statistics.median(float(row[field]) for row in iterations)
    prep = sum(float(row['seconds']) for row in group if row['phase'] == 'prepare')
    reorder = sum(float(row['seconds']) for row in group if row['phase'] == 'reorder')
    print(f'| {algo} | {rows} × {cols} | {rank} | {int(cache)/1048576:g} | {policy} | '
          f'{median("seconds"):.3f} | {median("cpu_seconds"):.2f} | '
          f'{median("read_bytes")/1e9:.3f} | {median("write_bytes")/1e9:.3f} | '
          f'{median("tasks"):.0f} | {prep:.3f} | {reorder:.3f} |')

if records:
    print(f'\nChecksums agree across policies for {len(reference)} iteration/shape combinations.')
    total = sum(int(row.get('process_read_bytes') or 0) for row in records if row['phase'] == 'iteration')
    print(f'Total /proc storage reads across measured iterations: {total} bytes.')
