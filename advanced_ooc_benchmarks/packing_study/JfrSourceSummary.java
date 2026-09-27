import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordedFrame;
import jdk.jfr.consumer.RecordingFile;

/** Streaming summary: avoids materializing hundreds of MB of JFR JSON. */
public class JfrSourceSummary {
    public static void main(String[] args) throws Exception {
        Map<String, Double> blocked = new HashMap<>();
        Map<String, Long> samples = new HashMap<>();
        Map<String, Long> nnzStacks = new HashMap<>();
        Map<String, Double> parks = new HashMap<>();
        double gc = 0;
        long monitors = 0;
        try (RecordingFile file = new RecordingFile(Path.of(args[0]))) {
            while (file.hasMoreEvents()) {
                RecordedEvent event = file.readEvent();
                String type = event.getEventType().getName();
                if (type.equals("jdk.GCPhasePause"))
                    gc += event.getDuration().toNanos() / 1e9;
                if (event.getStackTrace() == null || event.getStackTrace().getFrames().isEmpty())
                    continue;
                RecordedFrame top = event.getStackTrace().getFrames().get(0);
                String name = top.getMethod().getType().getName() + "." + top.getMethod().getName();
                if (type.equals("jdk.JavaMonitorEnter")) {
                    blocked.merge(name, event.getDuration().toNanos() / 1e9, Double::sum);
                    monitors++;
                }
                if (type.equals("jdk.ExecutionSample"))
                    samples.merge(name, 1L, Long::sum);
                if (type.equals("jdk.ExecutionSample") && name.endsWith(".computeNnz"))
                    nnzStacks.merge(event.getStackTrace().toString(), 1L, Long::sum);
                if (type.equals("jdk.ThreadPark"))
                    parks.merge(event.getStackTrace().toString(), event.getDuration().toNanos() / 1e9, Double::sum);
            }
        }
        System.out.printf("GC pause seconds: %.3f; monitor events: %d%n", gc, monitors);
        System.out.println("Monitor blocked thread-seconds (overlapping, not wall time):");
        blocked.entrySet().stream().sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(15).forEach(System.out::println);
        System.out.println("Execution samples by top frame:");
        samples.entrySet().stream().sorted(Map.Entry.<String, Long>comparingByValue().reversed())
            .limit(20).forEach(System.out::println);
        System.out.println("NNZ recount stacks:");
        nnzStacks.entrySet().stream().sorted(Map.Entry.<String, Long>comparingByValue().reversed())
            .limit(2).forEach(System.out::println);
        System.out.println("Parked thread-seconds by stack (includes idle executors):");
        parks.entrySet().stream().sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(5).forEach(System.out::println);
    }
}
