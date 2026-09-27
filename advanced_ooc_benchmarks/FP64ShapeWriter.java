import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.SequenceFile;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;

/** Bounded row-band preparation, preserving the exact canonical FP64 value stream. */
public class FP64ShapeWriter {
    public static void main(String[] args) throws Exception {
        Path raw = Path.of(args[0]), output = Path.of(args[1]);
        long rows = Long.parseLong(args[2]);
        int cols = Integer.parseInt(args[3]), parts = 22;
        int blen = args.length > 6 ? Integer.parseInt(args[6]) : 1000;
        if(blen <= 0) throw new IllegalArgumentException("Blocksize must be positive");
        boolean labels = Boolean.parseBoolean(args[4]);
        boolean labelsOnly = args.length > 5 && Boolean.parseBoolean(args[5]);
        if(labelsOnly && !labels) throw new IllegalArgumentException("Label-only mode needs labels");
        if(Files.size(raw) != rows * cols * 8)
            throw new IllegalArgumentException("Input size disagrees with geometry");
        if(!labelsOnly) Files.createDirectories(output.resolve("X"));
        if(labels) Files.createDirectories(output.resolve("y"));
        long rowBlocks = (rows + blen - 1) / blen;
        int colBlocks = (cols + blen - 1) / blen;
        SequenceFile.Writer[] writers = new SequenceFile.Writer[parts];
        SequenceFile.Writer[] yWriters = new SequenceFile.Writer[parts];
        Configuration conf = new Configuration();
        try(FileChannel source = FileChannel.open(raw, StandardOpenOption.READ)) {
            for(int i=0; i<parts; i++) {
                if(!labelsOnly) writers[i] = IOUtilFunctions.getSeqWriter(new org.apache.hadoop.fs.Path(
                    output.resolve("X").resolve(String.format("part-%05d",i)).toString()), conf, 1);
                if(labels) yWriters[i] = IOUtilFunctions.getSeqWriter(new org.apache.hadoop.fs.Path(
                    output.resolve("y").resolve(String.format("part-%05d",i)).toString()), conf, 1);
            }
            ByteBuffer band = ByteBuffer.allocate(blen * cols * 8).order(ByteOrder.LITTLE_ENDIAN);
            for(long br=0; br<rowBlocks; br++) {
                int nr = (int)Math.min(blen, rows-br*blen);
                band.clear().limit(nr * cols * 8);
                while(band.hasRemaining()) {
                    if(source.read(band) < 0) throw new IllegalStateException("Unexpected EOF");
                }
                band.flip();
                int part = (int)Math.min(parts-1, br*parts/rowBlocks);
                MatrixBlock y = labels ? new MatrixBlock(nr,1,false).allocateDenseBlock() : null;
                if(labelsOnly)
                    for(int r=0; r<nr; r++) y.getDenseBlockValues()[r] = band.getDouble(r*cols*8)>=0 ? 1 : -1;
                for(int bc=0; !labelsOnly && bc<colBlocks; bc++) {
                    int nc = Math.min(blen,cols-bc*blen);
                    MatrixBlock block = new MatrixBlock(nr,nc,false).allocateDenseBlock();
                    double[] values = block.getDenseBlockValues();
                    for(int r=0; r<nr; r++) {
                        for(int c=0; c<nc; c++)
                            values[r*nc+c] = band.getDouble((r*cols+bc*blen+c)*8);
                        if(labels && bc==0) y.getDenseBlockValues()[r] = values[r*nc]>=0 ? 1 : -1;
                    }
                    block.recomputeNonZeros();
                    writers[part].append(new MatrixIndexes(br+1,bc+1),block);
                }
                if(labels) {
                    y.recomputeNonZeros();
                    yWriters[part].append(new MatrixIndexes(br+1,1),y);
                }
                if(br % 10000 == 0) System.out.println("Prepared row blocks " + br + "/" + rowBlocks);
            }
        }
        finally {
            for(SequenceFile.Writer writer : writers) if(writer!=null) writer.close();
            for(SequenceFile.Writer writer : yWriters) if(writer!=null) writer.close();
        }
        for(String name : labelsOnly ? new String[]{"y"} : labels ? new String[]{"X","y"} : new String[]{"X"}) {
            int ncols = name.equals("X") ? cols : 1;
            Files.writeString(output.resolve(name+".mtd"),
                "{\"data_type\":\"matrix\",\"value_type\":\"double\",\"rows\":"+rows+
                ",\"cols\":"+ncols+",\"rows_in_block\":"+blen+",\"cols_in_block\":"+blen+",\"nnz\":"+
                (rows*ncols)+",\"format\":\"binary\"}\n");
        }
    }
}
