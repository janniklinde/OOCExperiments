package org.apache.sysds.runtime.ooc.cache.io;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class RawSourceScanProbe {
	public static void main(String[] args) throws Exception {
		Path directory = Path.of(args[0]);
		int threads = Integer.parseInt(args[1]);
		long splitBytes = Long.parseLong(args[2]) * 1024 * 1024;
		long startTime = System.nanoTime();
		long total = 0;
		try(var pool = Executors.newFixedThreadPool(threads); var files = Files.list(directory)) {
			var tasks = new ArrayList<Future<Long>>();
			for(Path path : files.filter(Files::isRegularFile)
				.filter(p -> !p.getFileName().toString().startsWith(".")).sorted().toList()) {
				long size = Files.size(path);
				for(long offset = 0; offset < size; offset += splitBytes) {
					long begin = offset;
					long end = Math.min(size, offset + splitBytes);
					tasks.add(pool.submit(() -> {
						try(var input = new OOCDirectInputStream(path, 1024 * 1024, true)) {
							input.seek(begin);
							byte[] buffer = new byte[1024 * 1024];
							long count = 0;
							while(count < end - begin) {
								int n = input.read(buffer, 0, (int) Math.min(buffer.length, end - begin - count));
								if(n < 0)
									throw new IllegalStateException("early EOF: " + path);
								count += n;
							}
							return count;
						}
					}));
				}
			}
			for(var task : tasks)
				total += task.get();
		}
		double seconds = (System.nanoTime() - startTime) / 1e9;
		System.out.printf("bytes=%d seconds=%.3f GB_per_second=%.3f%n", total, seconds, total / seconds / 1e9);
	}
}
