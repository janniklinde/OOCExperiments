import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** Experimental eviction-time tile packing, intentionally outside the SystemDS runtime. */
public final class ReactivePackExperiment {
    private record Ref(int pack, int slot) {}

    private static final class Pack {
        final int[] keys;
        final int[] offsets;
        final int bytes;
        final long fileOffset;
        final int min;
        final int max;

        Pack(int[] keys, int[] offsets, int bytes, long fileOffset) {
            this.keys = keys;
            this.offsets = offsets;
            this.bytes = bytes;
            this.fileOffset = fileOffset;
            this.min = keys[0];
            this.max = keys[keys.length - 1];
        }
    }

    private static final class Cache implements AutoCloseable {
        final int rows;
        final int cols;
        final int tileRows;
        final int count;
        final int target;
        final int hotLimit;
        final int maxGap;
        final Path path;
        final Ref[] refs;
        final ArrayList<Pack> packs = new ArrayList<>();
        final LinkedHashMap<Integer, double[]> hot = new LinkedHashMap<>();
        final TreeMap<Integer, double[]> hotByKey = new TreeMap<>();
        final FileChannel file;
        int hotBytes;
        long fileBytes;
        int underfilled;
        int writeCalls;

        Cache(Path path, int rows, int cols, int tileRows, int target, int hotLimit, int maxGap)
            throws IOException {
            if((long) tileRows * cols * Double.BYTES > target)
                throw new IllegalArgumentException("A logical tile cannot exceed the physical pack cap");
            this.rows = rows;
            this.cols = cols;
            this.tileRows = tileRows;
            count = (rows + tileRows - 1) / tileRows;
            this.target = target;
            this.hotLimit = hotLimit;
            this.maxGap = maxGap;
            this.path = path;
            refs = new Ref[count];
            Files.createDirectories(path.getParent());
            file = FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.READ,
                StandardOpenOption.WRITE);
        }

        int rowsIn(int index) {
            return Math.min(tileRows, rows - index * tileRows);
        }

        void insert(int index) throws IOException {
            double[] values = new double[rowsIn(index) * cols];
            for(int r = 0; r < rowsIn(index); r++) {
                long row = (long) index * tileRows + r;
                for(int c = 0; c < cols; c++)
                    values[r * cols + c] = ((row * 13 + c * 7) % 101) / 101.0;
            }
            hot.put(index, values);
            hotByKey.put(index, values);
            hotBytes += values.length * Double.BYTES;
            while(hotBytes > hotLimit)
                flushOne();
        }

        void finish() throws IOException {
            while(!hot.isEmpty())
                flushOne();
            file.force(false);
        }

        void flushOne() throws IOException {
            int seed = hot.keySet().iterator().next();
            ArrayList<Integer> selected = new ArrayList<>();
            selected.add(seed);
            int bytes = hot.get(seed).length * Double.BYTES;
            int lo = seed, hi = seed;
            while(bytes < target) {
                Integer lower = hotByKey.lowerKey(lo);
                Integer upper = hotByKey.higherKey(hi);
                int lowerGap = lower == null ? Integer.MAX_VALUE : lo - lower;
                int upperGap = upper == null ? Integer.MAX_VALUE : upper - hi;
                Integer next = lowerGap <= upperGap ? lower : upper;
                int gap = Math.min(lowerGap, upperGap);
                if(next == null || gap > maxGap)
                    break;
                int candidateBytes = hotByKey.get(next).length * Double.BYTES;
                if(bytes + candidateBytes > target)
                    break;
                selected.add(next);
                bytes += candidateBytes;
                lo = Math.min(lo, next);
                hi = Math.max(hi, next);
            }
            Collections.sort(selected);
            int[] keys = new int[selected.size()];
            int[] offsets = new int[selected.size()];
            ByteBuffer buffer = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
            for(int slot = 0; slot < selected.size(); slot++) {
                int key = selected.get(slot);
                keys[slot] = key;
                offsets[slot] = buffer.position();
                double[] values = hot.remove(key);
                hotByKey.remove(key);
                hotBytes -= values.length * Double.BYTES;
                for(double value : values)
                    buffer.putDouble(value);
            }
            buffer.flip();
            long offset = fileBytes;
            while(buffer.hasRemaining()) {
                int written = file.write(buffer, fileBytes);
                if(written <= 0)
                    throw new IOException("No forward progress writing a pack");
                fileBytes += written;
            }
            writeCalls++;
            int packId = packs.size();
            Pack pack = new Pack(keys, offsets, bytes, offset);
            packs.add(pack);
            for(int slot = 0; slot < keys.length; slot++)
                refs[keys[slot]] = new Ref(packId, slot);
            if(bytes < target * 3 / 4)
                underfilled++;
        }

        byte[][] readAll() throws IOException {
            byte[][] data = new byte[packs.size()][];
            for(int p = 0; p < packs.size(); p++) {
                Pack pack = packs.get(p);
                byte[] bytes = new byte[pack.bytes];
                ByteBuffer buffer = ByteBuffer.wrap(bytes);
                long offset = pack.fileOffset;
                while(buffer.hasRemaining()) {
                    int got = file.read(buffer, offset);
                    if(got <= 0)
                        throw new IOException("Short pack read");
                    offset += got;
                }
                data[p] = bytes;
            }
            return data;
        }

        int candidatePacks(int low, int high) {
            int n = 0;
            for(Pack pack : packs)
                if(pack.min <= high && pack.max >= low)
                    n++;
            return n;
        }

        @Override
        public void close() throws IOException {
            file.close();
        }
    }

    private static double operate(String op, Cache x, byte[][] xd, Cache b, byte[][] bd, double[] weights,
        boolean batch, int threads) throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        ArrayList<Future<Double>> tasks = new ArrayList<>();
        try {
            if(batch) {
                for(int p = 0; p < x.packs.size(); p++) {
                    final int packId = p;
                    tasks.add(pool.submit(() -> {
                        Pack pack = x.packs.get(packId);
                        ByteBuffer data = ByteBuffer.wrap(xd[packId]).order(ByteOrder.LITTLE_ENDIAN);
                        double result = 0;
                        for(int slot = 0; slot < pack.keys.length; slot++)
                            result += tileOp(op, x, data, pack.keys[slot], pack.offsets[slot], b, bd, weights);
                        return result;
                    }));
                }
            }
            else {
                for(int i = 0; i < x.count; i++) {
                    final int tileId = i;
                    tasks.add(pool.submit(() -> {
                        Ref ref = x.refs[tileId];
                        Pack pack = x.packs.get(ref.pack());
                        ByteBuffer data = ByteBuffer.wrap(xd[ref.pack()]).order(ByteOrder.LITTLE_ENDIAN);
                        return tileOp(op, x, data, tileId, pack.offsets[ref.slot()], b, bd, weights);
                    }));
                }
            }
            double sum = 0;
            for(Future<Double> task : tasks)
                sum += task.get();
            return sum;
        }
        finally {
            pool.shutdown();
        }
    }

    private static double tileOp(String op, Cache x, ByteBuffer data, int tileId, int start,
        Cache b, byte[][] bd, double[] weights) {
        int nr = x.rowsIn(tileId);
        double result = 0;
        if(op.equals("scan")) {
            for(int i = 0; i < nr * x.cols; i++)
                result += data.getDouble(start + i * Double.BYTES);
        }
        else if(op.equals("broadcast_join")) {
            Ref ref = b.refs[tileId];
            ByteBuffer vector = ByteBuffer.wrap(bd[ref.pack()]).order(ByteOrder.LITTLE_ENDIAN);
            int vectorStart = b.packs.get(ref.pack()).offsets[ref.slot()];
            for(int r = 0; r < nr; r++) {
                double scale = vector.getDouble(vectorStart + r * Double.BYTES);
                for(int c = 0; c < x.cols; c++)
                    result += data.getDouble(start + (r * x.cols + c) * Double.BYTES) * scale;
            }
        }
        else if(op.equals("matmul")) {
            for(int r = 0; r < nr; r++)
                for(int k = 0; k < x.cols; k++) {
                    double value = data.getDouble(start + (r * x.cols + k) * Double.BYTES);
                    for(int c = 0; c < 8; c++)
                        result += value * weights[k * 8 + c];
                }
        }
        return result;
    }

    private static int[] order(int count, String kind, long seed) {
        int[] indexes = new int[count];
        for(int i = 0; i < count; i++)
            indexes[i] = i;
        if(kind.equals("global")) {
            Random random = new Random(seed);
            for(int i = count - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int tmp = indexes[i]; indexes[i] = indexes[j]; indexes[j] = tmp;
            }
        }
        else if(kind.equals("window")) {
            Random random = new Random(seed);
            for(int lo = 0; lo < count; lo += 64)
                for(int i = Math.min(lo + 64, count) - 1; i > lo; i--) {
                    int j = lo + random.nextInt(i - lo + 1);
                    int tmp = indexes[i]; indexes[i] = indexes[j]; indexes[j] = tmp;
                }
        }
        return indexes;
    }

    private static void readPass(Cache cache, int stride, int repetition, int residentLimit, String kind)
        throws IOException {
        Process drop = new ProcessBuilder("/opt/devcon/env/python/bin/python",
            "advanced_ooc_benchmarks/drop_caches.py", cache.path.toString()).redirectErrorStream(true).start();
        try {
            String result = new String(drop.getInputStream().readAllBytes());
            if(drop.waitFor() != 0)
                throw new IOException("Could not evict source page cache: " + result);
        }
        catch(InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new IOException(ex);
        }
        long physicalBefore = physicalReadBytes();
        LinkedHashMap<Integer, byte[]> resident = new LinkedHashMap<>(16, 0.75f, true);
        int residentBytes = 0;
        long readBytes = 0;
        int readCalls = 0;
        double checksum = 0;
        long start = System.nanoTime();
        for(int tile = 0; tile < cache.count; tile += stride) {
            Ref ref = cache.refs[tile];
            byte[] data = resident.get(ref.pack());
            if(data == null) {
                Pack pack = cache.packs.get(ref.pack());
                data = new byte[pack.bytes];
                ByteBuffer buffer = ByteBuffer.wrap(data);
                long position = pack.fileOffset;
                while(buffer.hasRemaining()) {
                    int got = cache.file.read(buffer, position);
                    if(got <= 0)
                        throw new IOException("Short pack read");
                    position += got;
                }
                readCalls++;
                readBytes += pack.bytes;
                resident.put(ref.pack(), data);
                residentBytes += pack.bytes;
                while(residentBytes > residentLimit) {
                    Map.Entry<Integer, byte[]> eldest = resident.entrySet().iterator().next();
                    residentBytes -= eldest.getValue().length;
                    resident.remove(eldest.getKey());
                }
            }
            checksum += ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
                .getDouble(cache.packs.get(ref.pack()).offsets[ref.slot()]);
        }
        System.out.printf("READ,%s,%d,%d,%d,%d,%d,%.6f,%.12g%n", kind, stride, repetition, readCalls, readBytes,
            physicalReadBytes() - physicalBefore, (System.nanoTime() - start) / 1e9, checksum);
    }

    private static long physicalReadBytes() throws IOException {
        String relative = "";
        for(String line : Files.readAllLines(Path.of("/proc/self/cgroup")))
            if(line.startsWith("0::"))
                relative = line.substring(3);
        Path stat = Path.of("/sys/fs/cgroup" + relative, "io.stat");
        long total = 0;
        for(String line : Files.readAllLines(stat))
            for(String field : line.split(" "))
                if(field.startsWith("rbytes="))
                    total += Long.parseLong(field.substring(7));
        return total;
    }

    public static void main(String[] args) throws Exception {
        if(args.length != 13)
            throw new IllegalArgumentException("dir rows cols tileRows targetKiB hotMiB gap order threads repetitions kind readRepetitions readOnly");
        Path dir = Path.of(args[0]);
        int rows = Integer.parseInt(args[1]);
        int cols = Integer.parseInt(args[2]);
        int tileRows = Integer.parseInt(args[3]);
        int target = Integer.parseInt(args[4]) * 1024;
        int hot = Integer.parseInt(args[5]) * 1024 * 1024;
        int gap = Integer.parseInt(args[6]);
        String arrival = args[7];
        int threads = Integer.parseInt(args[8]);
        int reps = Integer.parseInt(args[9]);
        String kind = args[10];
        int readReps = Integer.parseInt(args[11]);
        boolean readOnly = Boolean.parseBoolean(args[12]);
        if(rows <= 0 || cols <= 0 || tileRows <= 0 || target <= 0 || hot <= target ||
            !(arrival.equals("sequential") || arrival.equals("window") || arrival.equals("global")) ||
            !(kind.equals("source") || kind.equals("reactive")) ||
            (kind.equals("source") && !arrival.equals("sequential")))
            throw new IllegalArgumentException("Invalid geometry, capacity, or arrival order");
        Files.createDirectories(dir);
        int ingestLimit = kind.equals("source") ? target : hot;
        int ingestGap = kind.equals("source") ? 1 : gap;
        Cache x = new Cache(dir.resolve("x.pack"), rows, cols, tileRows, target, ingestLimit, ingestGap);
        Cache b = new Cache(dir.resolve("b.pack"), rows, 1, tileRows, target, ingestLimit, ingestGap);
        try(x; b) {
            int[] arrivalOrder = order(x.count, arrival, 17);
            long ingestStart = System.nanoTime();
            for(int index : arrivalOrder) {
                x.insert(index);
                b.insert(index);
            }
            x.finish();
            b.finish();
            double ingestSec = (System.nanoTime() - ingestStart) / 1e9;
            int candidates = 0;
            for(Pack pack : x.packs)
                candidates += b.candidatePacks(pack.min, pack.max);
            HashSet<Long> actual = new HashSet<>();
            for(int i = 0; i < x.count; i++) {
                Ref xr = x.refs[i], br = b.refs[i];
                if(x.packs.get(xr.pack()).keys[xr.slot()] != i || b.packs.get(br.pack()).keys[br.slot()] != i)
                    throw new IllegalStateException("Inexact logical tile lookup at " + i);
                actual.add(((long) xr.pack() << 32) | (br.pack() & 0xffffffffL));
            }
            if(candidates < actual.size())
                throw new IllegalStateException("Pack range hints missed a join candidate");
            System.out.printf("PREP,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%d,%d,%d,%d,%d%n",
                kind, arrival, rows, cols, tileRows, target, hot, gap, x.count, x.packs.size(), ingestSec,
                x.underfilled, b.packs.size(), b.underfilled, candidates, actual.size());
            for(int rep = 1; rep <= readReps; rep++) {
                readPass(x, 1, rep, hot, kind);
                readPass(x, 10, rep, hot, kind);
            }
            if(readOnly)
                return;
            byte[][] xd = x.readAll();
            byte[][] bd = b.readAll();
            double[] weights = new double[cols * 8];
            for(int i = 0; i < weights.length; i++)
                weights[i] = (i % 13 + 1) / 13.0;
            for(String op : new String[] {"scan", "broadcast_join", "matmul"}) {
                operate(op, x, xd, b, bd, weights, true, threads);
                for(boolean batch : new boolean[] {false, true}) {
                    for(int rep = 1; rep <= reps; rep++) {
                        long start = System.nanoTime();
                        double result = operate(op, x, xd, b, bd, weights, batch, threads);
                        System.out.printf("RUN,%s,%s,%s,%s,%d,%.6f,%.12g,%d%n", kind, arrival, op,
                            batch ? "pack" : "tile", rep, (System.nanoTime() - start) / 1e9, result,
                            batch ? x.packs.size() : x.count);
                    }
                }
            }
        }
    }
}
