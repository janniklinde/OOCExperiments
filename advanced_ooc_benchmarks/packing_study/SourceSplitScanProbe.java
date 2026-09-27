package org.apache.sysds.runtime.ooc.cache.io;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;

/** Tests independent, sync-aligned reads of byte ranges within SequenceFiles. */
public final class SourceSplitScanProbe {
	private record Split(Path path, long start, long end, long fileLength) { }
	private record Result(Split split, long blocks, long empty, double seconds) { }

	private static Result scan(Split split, int bufferBytes) throws IOException {
		long startTime = System.nanoTime();
		long blocks = 0;
		long empty = 0;
		try(FSDataInputStream input = new FSDataInputStream(
			new OOCDirectInputStream(split.path(), bufferBytes, true));
			SequenceFile.Reader reader = new SequenceFile.Reader(new JobConf(),
				SequenceFile.Reader.stream(input), SequenceFile.Reader.length(split.fileLength()))) {
			if(split.start() != 0)
				reader.sync(split.start());
			MatrixIndexes indexes = new MatrixIndexes();
			MatrixBlock block = new MatrixBlock();
			while(true) {
				long position = reader.getPosition();
				if(!reader.next(indexes, block))
					break;
				// Match Hadoop's SequenceFileRecordReader boundary rule: the first
				// record after a sync past the split end belongs to the next split.
				if(position >= split.end() && reader.syncSeen())
					break;
				blocks++;
				if(block.isEmptyBlock(false))
					empty++;
				block = new MatrixBlock();
			}
		}
		return new Result(split, blocks, empty, (System.nanoTime() - startTime) / 1e9);
	}

	public static void main(String[] args) throws Exception {
		if(args.length != 4)
			throw new IllegalArgumentException("usage: SourceSplitScanProbe DIRECTORY THREADS SPLIT_MIB BUFFER_MIB");
		Path directory = Path.of(args[0]);
		int threads = Integer.parseInt(args[1]);
		long splitBytes = Long.parseLong(args[2]) * 1024 * 1024;
		int bufferBytes = Integer.parseInt(args[3]) * 1024 * 1024;
		if(threads < 1 || splitBytes < 1 || bufferBytes < 1)
			throw new IllegalArgumentException("threads, split size, and buffer size must be positive");
		MatrixBlock.setUseCOO(true);
		List<Path> files;
		try(var entries = Files.list(directory)) {
			files = entries.filter(Files::isRegularFile)
				.filter(path -> !path.getFileName().toString().startsWith("."))
				.sorted().toList();
		}
		List<Split> splits = new ArrayList<>();
		for(Path file : files) {
			long size = Files.size(file);
			for(long offset = 0; offset < size; offset += splitBytes)
				splits.add(new Split(file, offset, Math.min(size, offset + splitBytes), size));
		}
		long startTime = System.nanoTime();
		List<Result> results = new ArrayList<>();
		try(var pool = Executors.newFixedThreadPool(threads)) {
			List<Future<Result>> futures = new ArrayList<>();
			for(Split split : splits)
				futures.add(pool.submit((Callable<Result>) () -> scan(split, bufferBytes)));
			for(Future<Result> future : futures)
				results.add(future.get());
		}
		results.sort(Comparator.comparingDouble(Result::seconds).reversed());
		long blocks = 0;
		long empty = 0;
		for(Result result : results) {
			blocks += result.blocks();
			empty += result.empty();
		}
		for(Result result : results.subList(0, Math.min(8, results.size())))
			System.out.printf("slowest file=%s start=%d end=%d blocks=%d seconds=%.3f%n",
				result.split().path().getFileName(), result.split().start(), result.split().end(),
				result.blocks(), result.seconds());
		System.out.printf("total files=%d splits=%d threads=%d split_mib=%s blocks=%d empty=%d seconds=%.3f%n",
			files.size(), splits.size(), threads, args[2], blocks, empty,
			(System.nanoTime() - startTime) / 1e9);
	}
}
