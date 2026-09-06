import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.math.BigInteger;

import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.NodeIterator;

public class WebGraphToPageRank {
	private static final class LittleEndianOutput implements Closeable {
		private final FileChannel _channel;
		private final ByteBuffer _buffer = ByteBuffer.allocateDirect(8 << 20).order(ByteOrder.LITTLE_ENDIAN);

		LittleEndianOutput(Path path) throws IOException {
			_channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
				StandardOpenOption.WRITE);
		}

		void writeByte(int value) throws IOException {
			require(1);
			_buffer.put((byte)value);
		}

		void writeInt(int value) throws IOException {
			require(Integer.BYTES);
			_buffer.putInt(value);
		}

		void writeLong(long value) throws IOException {
			require(Long.BYTES);
			_buffer.putLong(value);
		}

		void writeDouble(double value) throws IOException {
			require(Double.BYTES);
			_buffer.putDouble(value);
		}

		private void require(int bytes) throws IOException {
			if(_buffer.remaining() < bytes)
				flush();
		}

		private void flush() throws IOException {
			_buffer.flip();
			while(_buffer.hasRemaining())
				_channel.write(_buffer);
			_buffer.clear();
		}

		@Override
		public void close() throws IOException {
			flush();
			_channel.close();
		}
	}

	public static void main(String[] args) throws Exception {
		if(args.length != 4 && args.length != 6) {
			System.err.println("usage: WebGraphToPageRank ORIGINAL_BASENAME TRANSPOSE_BASENAME OUTPUT_DIR NAME [MULTIPLIER OFFSET]");
			System.exit(2);
		}
		String originalBase = args[0];
		String transposeBase = args[1];
		Path output = Path.of(args[2]);
		String name = args[3];
		long multiplier = args.length == 6 ? Long.parseLong(args[4]) : 1;
		long permutationOffset = args.length == 6 ? Long.parseLong(args[5]) : 0;
		Files.createDirectories(output.resolve("csr"));
		Files.createDirectories(output.resolve("coo"));

		ImmutableGraph original = ImmutableGraph.loadOffline(originalBase);
		int vertices = original.numNodes();
		if(BigInteger.valueOf(multiplier).gcd(BigInteger.valueOf(vertices)).longValueExact() != 1)
			throw new IllegalArgumentException("Permutation multiplier must be coprime with the vertex count");
		long inverse = BigInteger.valueOf(multiplier).modInverse(BigInteger.valueOf(vertices)).longValueExact();
		int[] outDegree = new int[vertices];
		long edges = 0;
		NodeIterator nodes = original.nodeIterator();
		while(nodes.hasNext()) {
			int node = nodes.nextInt();
			int degree = nodes.outdegree();
			outDegree[permute(node, multiplier, permutationOffset, vertices)] = degree;
			edges += degree;
		}
		System.err.printf("%s: %,d vertices, %,d edges%n", name, vertices, edges);

		long dangling = 0;
		try(LittleEndianOutput flags = new LittleEndianOutput(output.resolve("dangling.u8"))) {
			for(int degree : outDegree) {
				boolean isDangling = degree == 0;
				flags.writeByte(isDangling ? 1 : 0);
				if(isDangling)
					dangling++;
			}
		}

		ImmutableGraph transpose = args.length == 6 ? ImmutableGraph.loadMapped(transposeBase)
			: ImmutableGraph.loadOffline(transposeBase);
		if(transpose.numNodes() != vertices)
			throw new IOException("Original and transpose have different vertex counts");
		long written = 0;
		try(LittleEndianOutput rowPtr = new LittleEndianOutput(output.resolve("csr/row_ptr.i64"));
			LittleEndianOutput columns = new LittleEndianOutput(output.resolve("csr/col_idx.i32"));
			LittleEndianOutput csrValues = new LittleEndianOutput(output.resolve("csr/values.f64"));
			LittleEndianOutput sources = new LittleEndianOutput(output.resolve("coo/src.int32"));
			LittleEndianOutput destinations = new LittleEndianOutput(output.resolve("coo/dst.int32"));
			LittleEndianOutput cooValues = new LittleEndianOutput(output.resolve("coo/values.f64"))) {
			rowPtr.writeLong(0);
			NodeIterator rows = args.length == 6 ? null : transpose.nodeIterator();
			for(int destination = 0; destination < vertices; destination++) {
				int oldDestination = args.length == 6
					? unpermute(destination, inverse, permutationOffset, vertices) : rows.nextInt();
				int degree = args.length == 6 ? transpose.outdegree(oldDestination) : rows.outdegree();
				int[] predecessors = args.length == 6 ? transpose.successorArray(oldDestination) : rows.successorArray();
				for(int i = 0; i < degree; i++) {
					int source = permute(predecessors[i], multiplier, permutationOffset, vertices);
					if(outDegree[source] == 0)
						throw new IOException("Transpose contains an edge from a dangling source " + source);
					double value = 1.0 / outDegree[source];
					columns.writeInt(source);
					csrValues.writeDouble(value);
					sources.writeInt(source);
					destinations.writeInt(destination);
					cooValues.writeDouble(value);
					written++;
				}
				rowPtr.writeLong(written);
				if((destination + 1) % 1_000_000 == 0)
					System.err.printf("rows %,d/%,d, edges %,d/%,d%n", destination + 1, vertices, written, edges);
			}
		}
		if(written != edges)
			throw new IOException("Transpose edge count " + written + " differs from original " + edges);

		String metadata = String.format("{\n"
			+ "  \"name\": \"%s\",\n"
			+ "  \"vertices\": %d,\n"
			+ "  \"edges\": %d,\n"
			+ "  \"input_edges\": %d,\n"
			+ "  \"nonzeros\": %d,\n"
			+ "  \"dangling_vertices\": %d,\n"
			+ "  \"vertex_permutation\": {\"multiplier\": %d, \"offset\": %d},\n"
			+ "  \"orientation\": \"CSR stores G=P^T: rows are destinations, columns are sources\",\n"
			+ "  \"normalization\": \"non-dangling source columns sum to one; dangling.u8 marks empty columns\",\n"
			+ "  \"dtype\": {\"row_ptr\": \"int64\", \"col_idx\": \"int32\", \"values\": \"float64\"}\n"
			+ "}\n", name, vertices, edges, edges, edges, dangling, multiplier, permutationOffset);
		Files.writeString(output.resolve("metadata.json"), metadata, StandardCharsets.UTF_8);
		System.err.printf("prepared %,d edges and %,d dangling vertices%n", written, dangling);
	}

	private static int permute(int vertex, long multiplier, long offset, int vertices) {
		return (int)Math.floorMod(multiplier * vertex + offset, vertices);
	}

	private static int unpermute(int vertex, long inverse, long offset, int vertices) {
		return (int)Math.floorMod(inverse * Math.floorMod(vertex - offset, vertices), vertices);
	}
}
