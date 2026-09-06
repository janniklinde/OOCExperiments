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

public class CSRToSystemDS {
	private static final class LittleEndianReader implements Closeable {
		private final FileChannel _channel;
		private final ByteBuffer _buffer = ByteBuffer.allocateDirect(8 << 20).order(ByteOrder.LITTLE_ENDIAN);

		LittleEndianReader(Path path, long offset) throws IOException {
			_channel = FileChannel.open(path, StandardOpenOption.READ);
			_channel.position(offset);
			_buffer.limit(0);
		}

		private void require(int bytes) throws IOException {
			if(_buffer.remaining() >= bytes)
				return;
			_buffer.compact();
			while(_buffer.position() < bytes && _channel.read(_buffer) >= 0) { }
			_buffer.flip();
			if(_buffer.remaining() < bytes)
				throw new IOException("Unexpected end of file");
		}

		int readInt() throws IOException {
			require(Integer.BYTES);
			return _buffer.getInt();
		}

		long readLong() throws IOException {
			require(Long.BYTES);
			return _buffer.getLong();
		}

		double readDouble() throws IOException {
			require(Double.BYTES);
			return _buffer.getDouble();
		}

		@Override
		public void close() throws IOException {
			_channel.close();
		}
	}

	private static long rowOffset(Path rowPtr, long row) throws IOException {
		try(FileChannel channel = FileChannel.open(rowPtr, StandardOpenOption.READ)) {
			ByteBuffer value = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
			channel.read(value, row * Long.BYTES);
			value.flip();
			return value.getLong();
		}
	}

	public static void main(String[] args) throws Exception {
		if(args.length != 6) {
			System.err.println("usage: CSRToSystemDS CSR_DIR OUTPUT_FILE VERTICES BLOCKSIZE FIRST_BLOCK LAST_BLOCK");
			System.exit(2);
		}
		Path csr = Path.of(args[0]);
		String output = args[1];
		long vertices = Long.parseLong(args[2]);
		int blocksize = Integer.parseInt(args[3]);
		int firstBlock = Integer.parseInt(args[4]);
		int lastBlock = Integer.parseInt(args[5]);
		long firstRow = (long)firstBlock * blocksize;
		long offset = rowOffset(csr.resolve("row_ptr.i64"), firstRow);
		long nextOffset = offset;
		try(LittleEndianReader rows = new LittleEndianReader(csr.resolve("row_ptr.i64"),
			(firstRow + 1) * Long.BYTES);
			LittleEndianReader cols = new LittleEndianReader(csr.resolve("col_idx.i32"), offset * Integer.BYTES);
			LittleEndianReader vals = new LittleEndianReader(csr.resolve("values.f64"), offset * Double.BYTES);
			SequenceFile.Writer writer = IOUtilFunctions.getSeqWriter(new org.apache.hadoop.fs.Path(output),
				new Configuration(), 1)) {
			for(int blockRow = firstBlock; blockRow < lastBlock; blockRow++) {
				int rowsInBlock = (int)Math.min(blocksize, vertices - (long)blockRow * blocksize);
				int colBlocks = (int)((vertices + blocksize - 1) / blocksize);
				SparseBlockCOO[] blocks = new SparseBlockCOO[colBlocks];
				for(int localRow = 0; localRow < rowsInBlock; localRow++) {
					nextOffset = rows.readLong();
					for(long position = offset; position < nextOffset; position++) {
						int column = cols.readInt();
						int blockCol = column / blocksize;
						if(blocks[blockCol] == null)
							blocks[blockCol] = new SparseBlockCOO(rowsInBlock);
						blocks[blockCol].append(localRow, column % blocksize, vals.readDouble());
					}
					offset = nextOffset;
				}
				for(int blockCol = 0; blockCol < colBlocks; blockCol++) {
					SparseBlockCOO sparse = blocks[blockCol];
					if(sparse == null)
						continue;
					int colsInBlock = (int)Math.min(blocksize, vertices - (long)blockCol * blocksize);
					writer.append(new MatrixIndexes(blockRow + 1L, blockCol + 1L),
						new MatrixBlock(rowsInBlock, colsInBlock, sparse.size(), sparse));
				}
			}
		}
	}
}
