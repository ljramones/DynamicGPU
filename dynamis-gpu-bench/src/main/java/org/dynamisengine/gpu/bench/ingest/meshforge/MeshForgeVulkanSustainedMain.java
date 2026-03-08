package org.dynamisengine.gpu.bench.ingest.meshforge;

import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.List;
import org.dynamisengine.gpu.api.layout.IndexType;
import org.dynamisengine.gpu.api.layout.SubmeshRange;
import org.dynamisengine.gpu.api.layout.VertexAttribute;
import org.dynamisengine.gpu.api.layout.VertexFormat;
import org.dynamisengine.gpu.api.layout.VertexLayout;
import org.dynamisengine.gpu.api.resource.GpuMeshResource;
import org.dynamisengine.gpu.api.upload.GpuGeometryUploadPlan;
import org.dynamisengine.gpu.vulkan.upload.VulkanGpuUploadExecutor;
import org.dynamisengine.meshforge.api.Packers;
import org.dynamisengine.meshforge.gpu.cache.RuntimeGeometryLoader;
import org.dynamisengine.meshforge.loader.MeshLoaders;

/**
 * Phase 3 sustained-throughput benchmark harness for upload execution modes.
 */
public final class MeshForgeVulkanSustainedMain {
  private MeshForgeVulkanSustainedMain() {}

  public static void main(String[] args) throws Exception {
    boolean debug = containsArg(args, "--debug");
    boolean phase4Overlap = containsArg(args, "--phase4-4a1");
    int maxInflight = parseIntOption(args, "--max-inflight", 1);
    int arrivalJitterMs = parseIntOption(args, "--arrival-jitter-ms", 0);
    int microburstSize = parseIntOption(args, "--microburst-size", 10);
    ArrivalPattern arrivalPattern = parseArrivalPattern(args);
    if (maxInflight < 1) {
      throw new IllegalArgumentException("--max-inflight must be >= 1");
    }
    if (arrivalJitterMs < 0) {
      throw new IllegalArgumentException("--arrival-jitter-ms must be >= 0");
    }
    if (microburstSize < 1) {
      throw new IllegalArgumentException("--microburst-size must be >= 1");
    }
    Path fixtureRoot =
        args.length > 0 && !args[0].startsWith("--")
            ? Path.of(args[0])
            : Path.of("..", "MeshForge", "fixtures", "baseline");

    VulkanLoaderBootstrap.bootstrap(debug);
    RuntimeGeometryLoader loader =
        new RuntimeGeometryLoader(MeshLoaders.defaultsFast(), Packers.realtimeFast());

    GpuGeometryUploadPlan lucyPlan = loadPlan(loader, fixtureRoot.resolve("lucy.obj"));
    GpuGeometryUploadPlan dragonPlan = loadPlan(loader, fixtureRoot.resolve("xyzrgb_dragon.obj"));
    if (lucyPlan == null || dragonPlan == null) {
      System.out.println("status=SKIPPED reason=missing_required_fixture");
      return;
    }

    try (VulkanHarnessContext context = VulkanHarnessContext.create();
        VulkanGpuUploadExecutor blockingExecutor =
            new VulkanGpuUploadExecutor(
                context.device(),
                context.physicalDevice(),
                context.commandPool(),
                context.graphicsQueue(),
                VulkanGpuUploadExecutor.UploadPathMode.OPTIMIZED);
        VulkanGpuUploadExecutor deferredExecutor =
            new VulkanGpuUploadExecutor(
                context.device(),
                context.physicalDevice(),
                context.commandPool(),
                context.graphicsQueue(),
                VulkanGpuUploadExecutor.UploadPathMode.OPTIMIZED_DEFERRED)) {
      if (phase4Overlap) {
        List<VulkanGpuUploadExecutor> overlapExecutors = new ArrayList<>(maxInflight);
        try {
          for (int i = 0; i < maxInflight; i++) {
            overlapExecutors.add(
                new VulkanGpuUploadExecutor(
                    context.device(),
                    context.physicalDevice(),
                    context.commandPool(),
                    context.graphicsQueue(),
                    VulkanGpuUploadExecutor.UploadPathMode.OPTIMIZED_DEFERRED));
          }
          printResult(
              runOverlapScenario(
                  "dragon_batch_10_overlap",
                  overlapExecutors,
                  dragonPlan,
                  10,
                  100,
                  maxInflight,
                  arrivalPattern,
                  arrivalJitterMs,
                  microburstSize));
          printResult(
              runOverlapScenario(
                  "lucy_batch_100_overlap",
                  overlapExecutors,
                  lucyPlan,
                  100,
                  100,
                  maxInflight,
                  arrivalPattern,
                  arrivalJitterMs,
                  microburstSize));
          printResult(
              runOverlapScenario(
                  "synthetic_100mb_overlap",
                  overlapExecutors,
                  syntheticPlan(100 * 1024 * 1024),
                  1,
                  30,
                  maxInflight,
                  arrivalPattern,
                  arrivalJitterMs,
                  microburstSize));
        } finally {
          closeExecutors(overlapExecutors);
        }
        return;
      }

      printResult(runRepeatedScenario("dragon_repeat_10", blockingExecutor, dragonPlan, 10));
      printResult(runRepeatedScenario("dragon_repeat_25", blockingExecutor, dragonPlan, 25));
      printResult(runRepeatedScenario("dragon_repeat_50", blockingExecutor, dragonPlan, 50));
      printResult(runRepeatedScenario("dragon_repeat_100", blockingExecutor, dragonPlan, 100));
      printResult(runRepeatedScenario("lucy_repeat_100", blockingExecutor, lucyPlan, 100));
      printResult(runRepeatedScenario("lucy_repeat_250", blockingExecutor, lucyPlan, 250));
      printResult(runRepeatedScenario("lucy_repeat_500", blockingExecutor, lucyPlan, 500));
      printResult(runRepeatedScenario("lucy_repeat_1000", blockingExecutor, lucyPlan, 1000));
      printResult(runBatchScenario("dragon_batch_10_blocking", blockingExecutor, dragonPlan, 10, false));
      printResult(runBatchScenario("dragon_batch_2_blocking", blockingExecutor, dragonPlan, 2, false));
      printResult(runBatchScenario("dragon_batch_5_blocking", blockingExecutor, dragonPlan, 5, false));
      printResult(runBatchScenario("dragon_batch_10_deferred", deferredExecutor, dragonPlan, 10, true));
      printResult(runBatchScenario("dragon_batch_2_deferred", deferredExecutor, dragonPlan, 2, true));
      printResult(runBatchScenario("dragon_batch_5_deferred", deferredExecutor, dragonPlan, 5, true));
      printResult(runBatchScenario("lucy_batch_100_blocking", blockingExecutor, lucyPlan, 100, false));
      printResult(runBatchScenario("lucy_batch_10_blocking", blockingExecutor, lucyPlan, 10, false));
      printResult(runBatchScenario("lucy_batch_25_blocking", blockingExecutor, lucyPlan, 25, false));
      printResult(runBatchScenario("lucy_batch_50_blocking", blockingExecutor, lucyPlan, 50, false));
      printResult(runBatchScenario("lucy_batch_100_deferred", deferredExecutor, lucyPlan, 100, true));
      printResult(runBatchScenario("lucy_batch_10_deferred", deferredExecutor, lucyPlan, 10, true));
      printResult(runBatchScenario("lucy_batch_25_deferred", deferredExecutor, lucyPlan, 25, true));
      printResult(runBatchScenario("lucy_batch_50_deferred", deferredExecutor, lucyPlan, 50, true));

      printResult(runSyntheticScenario("synthetic_25mb_deferred", deferredExecutor, 25 * 1024 * 1024));
      printResult(runSyntheticScenario("synthetic_50mb_deferred", deferredExecutor, 50 * 1024 * 1024));
      printResult(runSyntheticScenario("synthetic_100mb_deferred", deferredExecutor, 100 * 1024 * 1024));
    } catch (Throwable t) {
      System.out.println("status=UPLOAD_BLOCKED detail=" + summarizeThrowable(t));
    }
  }

  private static ScenarioResult runRepeatedScenario(
      String name, VulkanGpuUploadExecutor executor, GpuGeometryUploadPlan template, int count)
      throws Exception {
    long totalBytes = totalPlanBytes(template) * count;
    long start = System.nanoTime();
    for (int i = 0; i < count; i++) {
      try (GpuMeshResource resource = executor.upload(template)) {
        // Resource scope closes each iteration by design.
      }
    }
    long end = System.nanoTime();
    long uploadNanos = end - start;
    return new ScenarioResult(
        name,
        executor.mode().name(),
        "BLOCKING",
        count,
        count,
        totalBytes,
        uploadNanos,
        0L,
        uploadNanos,
        toGbps(totalBytes, uploadNanos));
  }

  private static ScenarioResult runBatchScenario(
      String name,
      VulkanGpuUploadExecutor executor,
      GpuGeometryUploadPlan template,
      int batchCount,
      boolean deferredCompletion)
      throws Exception {
    List<GpuGeometryUploadPlan> plans = Collections.nCopies(batchCount, template);
    long totalBytes = totalPlanBytes(template) * batchCount;
    if (!deferredCompletion) {
      long start = System.nanoTime();
      List<GpuMeshResource> resources = executor.uploadBatch(plans);
      long end = System.nanoTime();
      closeAll(resources);
      long totalNanos = end - start;
      return new ScenarioResult(
          name,
          executor.mode().name(),
          "BLOCKING",
          1,
          batchCount,
          totalBytes,
          totalNanos,
          0L,
          totalNanos,
          toGbps(totalBytes, totalNanos));
    }

    long submitStart = System.nanoTime();
    VulkanGpuUploadExecutor.DeferredUploadBatch pending = executor.submitBatchDeferred(plans);
    long submitEnd = System.nanoTime();
    List<GpuMeshResource> resources = executor.completeDeferredBatch(pending, 5_000_000_000L);
    long completeEnd = System.nanoTime();
    closeAll(resources);
    long submitNanos = submitEnd - submitStart;
    long completionNanos = completeEnd - submitEnd;
    long totalNanos = completeEnd - submitStart;
    return new ScenarioResult(
        name,
        executor.mode().name(),
        "DEFERRED",
        1,
        batchCount,
        totalBytes,
        submitNanos,
        completionNanos,
        totalNanos,
        toGbps(totalBytes, totalNanos));
  }

  private static ScenarioResult runSyntheticScenario(
      String name, VulkanGpuUploadExecutor executor, int totalBytes)
      throws Exception {
    GpuGeometryUploadPlan synthetic = syntheticPlan(totalBytes);
    return runBatchScenario(name, executor, synthetic, 1, true);
  }

  private static ScenarioResult runOverlapScenario(
      String name,
      List<VulkanGpuUploadExecutor> executors,
      GpuGeometryUploadPlan template,
      int batchCount,
      int iterations,
      int maxInflight,
      ArrivalPattern arrivalPattern,
      int arrivalJitterMs,
      int microburstSize)
      throws Exception {
    if (executors.isEmpty()) {
      throw new IllegalArgumentException("executors must not be empty");
    }
    List<GpuGeometryUploadPlan> plans = Collections.nCopies(batchCount, template);
    long bytesPerSubmit = totalPlanBytes(template) * batchCount;
    long totalBytes = bytesPerSubmit * iterations;
    long totalSubmitNanos = 0L;
    long totalCompletionNanos = 0L;
    long totalLatencyNanos = 0L;
    int maxInflightSeen = 0;
    int completed = 0;
    ArrayDeque<PendingTicket> inflight = new ArrayDeque<>();
    int submitIndex = 0;
    String mode = executors.get(0).mode().name();

    long wallStart = System.nanoTime();
    for (int i = 0; i < iterations; i++) {
      applyArrivalDelay(i, arrivalPattern, arrivalJitterMs, microburstSize);
      VulkanGpuUploadExecutor executor = executors.get(submitIndex % executors.size());
      submitIndex++;
      long submitStart = System.nanoTime();
      VulkanGpuUploadExecutor.DeferredUploadBatch batch = executor.submitBatchDeferred(plans);
      long submitEnd = System.nanoTime();
      totalSubmitNanos += submitEnd - submitStart;
      inflight.addLast(new PendingTicket(executor, batch, submitEnd));
      maxInflightSeen = Math.max(maxInflightSeen, inflight.size());

      while (inflight.size() >= maxInflight) {
        totalCompletionNanos += completeOldestDeferred(inflight);
        PendingTicket completedBatch = inflight.removeFirst();
        long completionEnd = System.nanoTime();
        totalLatencyNanos += completionEnd - completedBatch.submitEndNanos();
        completed++;
      }
    }
    while (!inflight.isEmpty()) {
      totalCompletionNanos += completeOldestDeferred(inflight);
      PendingTicket completedBatch = inflight.removeFirst();
      long completionEnd = System.nanoTime();
      totalLatencyNanos += completionEnd - completedBatch.submitEndNanos();
      completed++;
    }
    long wallEnd = System.nanoTime();
    long totalNanos = wallEnd - wallStart;
    return new ScenarioResult(
        name + "_inflight" + maxInflight,
        mode,
        "DEFERRED_OVERLAP",
        iterations,
        iterations * batchCount,
        totalBytes,
        totalSubmitNanos,
        totalCompletionNanos,
        totalNanos,
        toGbps(totalBytes, totalNanos),
        maxInflightSeen,
        completed == 0 ? 0.0 : nanosToMillis(totalLatencyNanos / completed));
  }

  private static long completeOldestDeferred(ArrayDeque<PendingTicket> inflight) throws Exception {
    PendingTicket oldest = inflight.peekFirst();
    if (oldest == null) {
      return 0L;
    }
    long completionStart = System.nanoTime();
    List<GpuMeshResource> resources =
        oldest.executor().completeDeferredBatch(oldest.batch(), 5_000_000_000L);
    long completionEnd = System.nanoTime();
    closeAll(resources);
    return completionEnd - completionStart;
  }

  private static void printResult(ScenarioResult result) {
    System.out.println(
        "scenario="
            + result.name()
            + " mode="
            + result.mode()
            + " completion="
            + result.completion()
            + " submits="
            + result.submits()
            + " uploads="
            + result.uploads()
            + " uploadedBytes="
            + result.uploadedBytes()
            + " submitMs="
            + nanosToMillis(result.submitNanos())
            + " completionMs="
            + nanosToMillis(result.completionNanos())
            + " totalMs="
            + nanosToMillis(result.totalNanos())
            + " uploadGbps="
            + String.format(java.util.Locale.ROOT, "%.3f", result.gbps())
            + " maxInflightSeen="
            + result.maxInflightSeen()
            + " avgCompletionLatencyMs="
            + String.format(java.util.Locale.ROOT, "%.3f", result.avgCompletionLatencyMs()));
  }

  private static GpuGeometryUploadPlan loadPlan(RuntimeGeometryLoader loader, Path meshPath)
      throws Exception {
    if (!java.nio.file.Files.isRegularFile(meshPath)) {
      return null;
    }
    Path cache = java.nio.file.Files.createTempFile("dynamisgpu-sustained-", ".mfgc");
    cache.toFile().deleteOnExit();
    RuntimeGeometryLoader.Result loaded = loader.load(meshPath, cache, false);
    return MeshForgeRuntimePlanAdapter.toApiPlan(loaded.payload());
  }

  private static GpuGeometryUploadPlan syntheticPlan(int totalBytes) {
    int vertexBytes = Math.max(16, (totalBytes * 2) / 3);
    int indexBytes = Math.max(4, totalBytes - vertexBytes);
    vertexBytes = vertexBytes - (vertexBytes % 16);
    if (vertexBytes <= 0) {
      vertexBytes = 16;
    }
    indexBytes = indexBytes - (indexBytes % 4);
    if (indexBytes <= 0) {
      indexBytes = 4;
    }

    ByteBuffer vertices = ByteBuffer.allocateDirect(vertexBytes);
    ByteBuffer indices = ByteBuffer.allocateDirect(indexBytes);
    for (int i = 0; i < vertexBytes; i++) {
      vertices.put((byte) (i & 0x7F));
    }
    vertices.flip();
    int indexCount = indexBytes / 4;
    for (int i = 0; i < indexCount; i++) {
      indices.putInt(i);
    }
    indices.flip();
    VertexLayout layout = new VertexLayout(16, List.of(new VertexAttribute(0, 0, VertexFormat.FLOAT4)));
    return new GpuGeometryUploadPlan(
        vertices, indices, layout, IndexType.UINT32, List.of(new SubmeshRange(0, indexCount, 0)));
  }

  private static long totalPlanBytes(GpuGeometryUploadPlan plan) {
    long total = plan.vertexData().remaining();
    if (plan.indexData() != null) {
      total += plan.indexData().remaining();
    }
    return total;
  }

  private static void closeAll(List<GpuMeshResource> resources) {
    RuntimeException first = null;
    for (GpuMeshResource resource : resources) {
      try {
        resource.close();
      } catch (RuntimeException e) {
        if (first == null) {
          first = e;
        } else {
          first.addSuppressed(e);
        }
      }
    }
    if (first != null) {
      throw first;
    }
  }

  private static void closeExecutors(List<VulkanGpuUploadExecutor> executors) {
    RuntimeException first = null;
    for (VulkanGpuUploadExecutor executor : executors) {
      try {
        executor.close();
      } catch (RuntimeException e) {
        if (first == null) {
          first = e;
        } else {
          first.addSuppressed(e);
        }
      }
    }
    if (first != null) {
      throw first;
    }
  }

  private static double toGbps(long bytes, long nanos) {
    if (bytes <= 0L || nanos <= 0L) {
      return 0.0;
    }
    double seconds = nanos / 1_000_000_000.0;
    return (bytes / seconds) / 1_000_000_000.0;
  }

  private static double nanosToMillis(long nanos) {
    return nanos / 1_000_000.0;
  }

  private static String summarizeThrowable(Throwable throwable) {
    String type = throwable.getClass().getSimpleName();
    String message = throwable.getMessage();
    if (message == null || message.isBlank()) {
      return type;
    }
    return type + ": " + message;
  }

  private static boolean containsArg(String[] args, String target) {
    for (String arg : args) {
      if (target.equals(arg)) {
        return true;
      }
    }
    return false;
  }

  private static int parseIntOption(String[] args, String option, int defaultValue) {
    String prefix = option + "=";
    for (String arg : args) {
      if (arg.startsWith(prefix)) {
        return Integer.parseInt(arg.substring(prefix.length()));
      }
    }
    return defaultValue;
  }

  private static ArrivalPattern parseArrivalPattern(String[] args) {
    String pattern = parseStringOption(args, "--arrival-pattern", "burst");
    return switch (pattern) {
      case "burst" -> ArrivalPattern.BURST;
      case "staggered" -> ArrivalPattern.STAGGERED;
      case "microburst" -> ArrivalPattern.MICROBURST;
      default -> throw new IllegalArgumentException("--arrival-pattern must be burst|staggered|microburst");
    };
  }

  private static String parseStringOption(String[] args, String option, String defaultValue) {
    String prefix = option + "=";
    for (String arg : args) {
      if (arg.startsWith(prefix)) {
        return arg.substring(prefix.length());
      }
    }
    return defaultValue;
  }

  private static void applyArrivalDelay(
      int iteration, ArrivalPattern arrivalPattern, int arrivalJitterMs, int microburstSize)
      throws InterruptedException {
    if (arrivalJitterMs <= 0) {
      return;
    }
    switch (arrivalPattern) {
      case BURST -> {
        // No delay.
      }
      case STAGGERED -> Thread.sleep(arrivalJitterMs);
      case MICROBURST -> {
        if (iteration > 0 && (iteration % microburstSize) == 0) {
          Thread.sleep(arrivalJitterMs);
        }
      }
    }
  }

  private record ScenarioResult(
      String name,
      String mode,
      String completion,
      int submits,
      int uploads,
      long uploadedBytes,
      long submitNanos,
      long completionNanos,
      long totalNanos,
      double gbps,
      int maxInflightSeen,
      double avgCompletionLatencyMs) {
    private ScenarioResult(
        String name,
        String mode,
        String completion,
        int submits,
        int uploads,
        long uploadedBytes,
        long submitNanos,
        long completionNanos,
        long totalNanos,
        double gbps) {
      this(name, mode, completion, submits, uploads, uploadedBytes, submitNanos, completionNanos, totalNanos, gbps, 0, 0.0);
    }
  }

  private record PendingTicket(
      VulkanGpuUploadExecutor executor,
      VulkanGpuUploadExecutor.DeferredUploadBatch batch,
      long submitEndNanos) {}

  private enum ArrivalPattern {
    BURST,
    STAGGERED,
    MICROBURST
  }
}
