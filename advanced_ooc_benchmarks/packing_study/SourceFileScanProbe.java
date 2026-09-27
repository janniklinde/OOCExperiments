package org.apache.sysds.runtime.ooc.cache.io;

import java.nio.file.Files;
import java.nio.file.Path;
import java.io.IOException;
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

/** Isolates the file-parallel OOC source scan without PageRank computation or materialization. */
public final class SourceFileScanProbe {
	private record Result(String file, long bytes, long blocks, long empty, double seconds) { }

	private static long fileSize(Path path) {
		try {
			return Files.size(path);
		}
		catch(IOException error) {
			throw new RuntimeException(error);
		}
	}

	private static Result scan(Path path, int bufferBytes) throws Exception {
		long start = System.nanoTime();
		long blocks = 0;
		long empty = 0;
		try(FSDataInputStream input = new FSDataInputStream(new OOCDirectInputStream(path, bufferBytes, true));
			SequenceFile.Reader reader = new SequenceFile.Reader(new JobConf(),
				SequenceFile.Reader.stream(input), SequenceFile.Reader.length(Files.size(path)))) {
			MatrixIndexes indexes = new MatrixIndexes();
			MatrixBlock block = new MatrixBlock();
			while(reader.next(indexes, block)) {
				blocks++;
				if(block.isEmptyBlock(false))
					empty++;
				block = new MatrixBlock();
			}
		}
		return new Result(path.getFileName().toString(), Files.size(path), blocks, empty,
			(System.nanoTime() - start) / 1e9);
	}

	public static void main(String[] args) throws Exception {
		if(args.length < 2 || args.length > 4)
			throw new IllegalArgumentException("usage: SourceFileScanProbe DIRECTORY THREADS [BUFFER_MIB] [largest]");
		Path directory = Path.of(args[0]);
		int threads = Integer.parseInt(args[1]);
		int bufferBytes = (args.length >= 3 ? Integer.parseInt(args[2]) : 8) * 1024 * 1024;
		if(threads < 1 || bufferBytes < 1)
			throw new IllegalArgumentException("threads and buffer size must be positive");
		MatrixBlock.setUseCOO(true);
		List<Path> files;
		try(var entries = Files.list(directory)) {
			files = entries.filter(Files::isRegularFile)
				.filter(path -> !path.getFileName().toString().startsWith("."))
				.sorted().toList();
		}
		if(args.length == 4 && args[3].equals("largest"))
			files = files.stream().sorted(Comparator.comparingLong(SourceFileScanProbe::fileSize).reversed()).toList();
		long start = System.nanoTime();
		List<Result> results = new ArrayList<>();
		try(var pool = Executors.newFixedThreadPool(threads)) {
			List<Future<Result>> futures = new ArrayList<>();
			for(Path file : files)
				futures.add(pool.submit((Callable<Result>) () -> scan(file, bufferBytes)));
			for(Future<Result> future : futures)
				results.add(future.get());
		}
		results.sort(Comparator.comparingDouble(Result::seconds).reversed());
		long bytes = 0;
		long blocks = 0;
		long empty = 0;
		for(Result result : results) {
			bytes += result.bytes();
			blocks += result.blocks();
			empty += result.empty();
			System.out.printf("file=%s bytes=%d blocks=%d empty=%d seconds=%.3f%n",
				result.file(), result.bytes(), result.blocks(), result.empty(), result.seconds());
		}
		System.out.printf("total files=%d threads=%d order=%s bytes=%d blocks=%d empty=%d seconds=%.3f%n",
			files.size(), threads, args.length == 4 ? args[3] : "name", bytes, blocks, empty,
			(System.nanoTime() - start) / 1e9);
	}
}
