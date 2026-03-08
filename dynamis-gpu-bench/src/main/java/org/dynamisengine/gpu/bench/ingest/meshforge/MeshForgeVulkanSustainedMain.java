package org.dynamisengine.gpu.bench.ingest.meshforge;

import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
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

      printResult(runRepeatedScenario("dragon_repeat_100", blockingExecutor, dragonPlan, 100));
      printResult(runRepeatedScenario("lucy_repeat_1000", blockingExecutor, lucyPlan, 1000));
      printResult(runBatchScenario("dragon_batch_10_blocking", blockingExecutor, dragonPlan, 10, false));
      printResult(runBatchScenario("dragon_batch_10_deferred", deferredExecutor, dragonPlan, 10, true));
      printResult(runBatchScenario("lucy_batch_100_blocking", blockingExecutor, lucyPlan, 100, false));
      printResult(runBatchScenario("lucy_batch_100_deferred", deferredExecutor, lucyPlan, 100, true));

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
      try (GpuMeshResource resource = executor.upload(clonePlan(template))) {
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
    List<GpuGeometryUploadPlan> plans = clonePlanList(template, batchCount);
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
            + String.format(java.util.Locale.ROOT, "%.3f", result.gbps()));
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

  private static GpuGeometryUploadPlan clonePlan(GpuGeometryUploadPlan source) {
    ByteBuffer vertex = copyToDirect(source.vertexData());
    ByteBuffer index = source.indexData() == null ? null : copyToDirect(source.indexData());
    return new GpuGeometryUploadPlan(
        vertex, index, source.vertexLayout(), source.indexType(), source.submeshes());
  }

  private static List<GpuGeometryUploadPlan> clonePlanList(GpuGeometryUploadPlan source, int count) {
    ArrayList<GpuGeometryUploadPlan> plans = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      plans.add(clonePlan(source));
    }
    return List.copyOf(plans);
  }

  private static ByteBuffer copyToDirect(ByteBuffer source) {
    ByteBuffer duplicate = source.duplicate();
    ByteBuffer direct = ByteBuffer.allocateDirect(duplicate.remaining());
    direct.put(duplicate);
    direct.flip();
    return direct;
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
      double gbps) {}
}
