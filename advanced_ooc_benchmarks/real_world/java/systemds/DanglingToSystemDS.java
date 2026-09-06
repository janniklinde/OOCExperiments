import java.io.BufferedInputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.SequenceFile;
import org.apache.sysds.runtime.data.SparseBlockCOO;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;

public class DanglingToSystemDS {
	public static void main(String[] args) throws Exception {
		if(args.length != 4) {
			System.err.println("usage: DanglingToSystemDS DANGLING_U8 OUTPUT_FILE VERTICES BLOCKSIZE");
			System.exit(2);
		}
		Path input = Path.of(args[0]);
		String output = args[1];
		long vertices = Long.parseLong(args[2]);
		int blocksize = Integer.parseInt(args[3]);
		try(InputStream flags = new BufferedInputStream(Files.newInputStream(input), 8 << 20);
			SequenceFile.Writer writer = IOUtilFunctions.getSeqWriter(new org.apache.hadoop.fs.Path(output),
				new Configuration(), 1)) {
			long blocks = (vertices + blocksize - 1) / blocksize;
			for(long block = 0; block < blocks; block++) {
				int rows = (int)Math.min(blocksize, vertices - block * blocksize);
				SparseBlockCOO sparse = new SparseBlockCOO(rows);
				for(int row = 0; row < rows; row++) {
					int flag = flags.read();
					if(flag < 0)
						throw new IllegalStateException("Unexpected end of dangling bitmap");
					if(flag != 0)
						sparse.append(row, 0, 1);
				}
				if(sparse.size() > 0)
					writer.append(new MatrixIndexes(block + 1, 1), new MatrixBlock(rows, 1, sparse.size(), sparse));
			}
			if(flags.read() >= 0)
				throw new IllegalStateException("Dangling bitmap is longer than vertex count");
		}
	}
}
