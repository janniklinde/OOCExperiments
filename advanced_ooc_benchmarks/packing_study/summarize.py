#!/usr/bin/env python3
"""Summarize prototype fill, cold reads, and warm operator timings."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    print('case,tile_bytes,packs,mean_pack_bytes,underfilled,candidate_pairs,'
          'actual_pairs,full_read_calls,full_physical_MB,full_read_ms,stride10_physical_MB,stride10_read_ms,'
          'scan_tile_ms,scan_pack_ms,join_tile_ms,join_pack_ms,mm_tile_ms,mm_pack_ms,status')
    for case in sorted(args.results.iterdir()):
        log = case / 'packing.log'
        if not log.exists():
            continue
        prep = None
        reads = {}
        runs = {}
        checksums = {}
        for row in csv.reader(log.open()):
            if not row:
                continue
            if row[0] == 'PREP':
                prep = row
            elif row[0] == 'READ':
                reads.setdefault(int(row[2]), []).append(row)
            elif row[0] == 'RUN':
                key = (row[3], row[4])
                runs.setdefault(key, []).append(float(row[6]))
                checksums.setdefault(row[3], []).append(float(row[7]))
        if prep is None:
            continue
        rows, cols, tile_rows = map(int, prep[3:6])
        packs = int(prep[10])
        total_bytes = rows * cols * 8
        valid = all(max(values) - min(values) <= max(1, abs(values[0])) * 1e-10
                    for values in checksums.values())
        metrics = json.loads((case / 'packing.metrics.json').read_text())
        read_all = reads[1]
        read_stride = reads[10]
        new_layout = len(read_all[0]) == 9
        call_field = 4 if new_layout else 3
        physical_field = 6 if new_layout else 5
        time_field = 7 if new_layout else 6
        values = [case.name, min(tile_rows, rows) * cols * 8, packs,
                  round(total_bytes / packs), prep[12], prep[15],
                  prep[16] if len(prep) > 16 else '', read_all[0][call_field],
                  round(median(int(row[physical_field]) for row in read_all) / 1e6, 2),
                  round(median(float(row[time_field]) for row in read_all) * 1000, 3),
                  round(median(int(row[physical_field]) for row in read_stride) / 1e6, 2),
                  round(median(float(row[time_field]) for row in read_stride) * 1000, 3)]
        for op in ('scan', 'broadcast_join', 'matmul'):
            for mode in ('tile', 'pack'):
                sample = runs.get((op, mode), [])
                values.append(round(median(sample[1:] if len(sample) > 1 else sample) * 1000, 3)
                              if sample else '')
        values.append((metrics['status'] + ('-read-only' if not runs else '')) if valid
                      else 'checksum_mismatch')
        print(','.join(map(str, values)))


if __name__ == '__main__':
    main()
