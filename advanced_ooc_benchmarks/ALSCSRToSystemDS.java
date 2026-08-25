import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.SequenceFile;
import org.apache.sysds.runtime.data.SparseBlockCOO;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;

/** Stream canonical little-endian CSR arrays into SystemDS binary matrix blocks. */
public class ALSCSRToSystemDS {
	private static final class LittleEndianReader implements Closeable {
		private final FileChannel channel;
		private final ByteBuffer buffer = ByteBuffer.allocateDirect(8 << 20)
			.order(ByteOrder.LITTLE_ENDIAN);

		LittleEndianReader(Path path) throws IOException {
			channel = FileChannel.open(path, StandardOpenOption.READ);
			buffer.limit(0);
		}

		private void require(int bytes) throws IOException {
			if(buffer.remaining() >= bytes)
				return;
			buffer.compact();
			while(buffer.position() < bytes && channel.read(buffer) >= 0) { }
			buffer.flip();
			if(buffer.remaining() < bytes)
				throw new IOException("Unexpected end of CSR file");
		}

		int readInt() throws IOException {
			require(Integer.BYTES);
			return buffer.getInt();
		}

		long readLong() throws IOException {
			require(Long.BYTES);
			return buffer.getLong();
		}

		double readDouble() throws IOException {
			require(Double.BYTES);
			return buffer.getDouble();
		}

		@Override
		public void close() throws IOException {
			channel.close();
		}
	}

	public static void main(String[] args) throws Exception {
		if(args.length != 6) {
			System.err.println("usage: ALSCSRToSystemDS CSR_DIR OUTPUT_FILE ROWS COLS BLOCKSIZE NNZ");
			System.exit(2);
		}
		Path csr = Path.of(args[0]);
		String output = args[1];
		long numRows = Long.parseLong(args[2]);
		long numCols = Long.parseLong(args[3]);
		int blocksize = Integer.parseInt(args[4]);
		long expectedNnz = Long.parseLong(args[5]);
		int rowBlocks = Math.toIntExact((numRows + blocksize - 1) / blocksize);
		int colBlocks = Math.toIntExact((numCols + blocksize - 1) / blocksize);
		long offset = 0;

		try(LittleEndianReader rows = new LittleEndianReader(csr.resolve("row_ptr.i64"));
			LittleEndianReader columns = new LittleEndianReader(csr.resolve("col_idx.i32"));
			LittleEndianReader values = new LittleEndianReader(csr.resolve("values.f64"));
			SequenceFile.Writer writer = IOUtilFunctions.getSeqWriter(
				new org.apache.hadoop.fs.Path(output), new Configuration(), 1)) {
			if(rows.readLong() != 0)
				throw new IOException("CSR row pointer must start at zero");
			for(int blockRow = 0; blockRow < rowBlocks; blockRow++) {
				int rowsInBlock = (int)Math.min(blocksize, numRows - (long)blockRow * blocksize);
				SparseBlockCOO[] blocks = new SparseBlockCOO[colBlocks];
				for(int localRow = 0; localRow < rowsInBlock; localRow++) {
					long nextOffset = rows.readLong();
					for(long position = offset; position < nextOffset; position++) {
						int column = columns.readInt();
						if(column < 0 || column >= numCols)
							throw new IOException("CSR column outside declared matrix: " + column);
						int blockCol = column / blocksize;
						if(blocks[blockCol] == null)
							blocks[blockCol] = new SparseBlockCOO(rowsInBlock);
						blocks[blockCol].append(localRow, column % blocksize, values.readDouble());
					}
					offset = nextOffset;
				}
				for(int blockCol = 0; blockCol < colBlocks; blockCol++) {
					SparseBlockCOO sparse = blocks[blockCol];
					if(sparse == null)
						continue;
					int colsInBlock = (int)Math.min(blocksize,
						numCols - (long)blockCol * blocksize);
					writer.append(new MatrixIndexes(blockRow + 1L, blockCol + 1L),
						new MatrixBlock(rowsInBlock, colsInBlock, sparse.size(), sparse));
				}
			}
		}
		if(offset != expectedNnz)
			throw new IOException("CSR contains " + offset + " entries, expected " + expectedNnz);
	}
}
