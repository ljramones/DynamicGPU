package org.dynamisengine.gpu.bench.ingest.meshforge;

import org.dynamisengine.gpu.vulkan.upload.VulkanGpuUploadExecutor;
import org.dynamisengine.gpu.bench.ingest.DefaultGeometryUploadValidation;
import org.dynamisengine.meshforge.api.Packers;
import org.dynamisengine.meshforge.gpu.cache.RuntimeGeometryLoader;
import org.dynamisengine.meshforge.loader.MeshLoaders;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Manual runner for baseline MeshForge -> Vulkan ingestion timings.
 */
public final class MeshForgeVulkanIngestionMain {
  private MeshForgeVulkanIngestionMain() {}

  public static void main(String[] args) throws Exception {
    int argOffset = 0;
    boolean debug = false;
    if (args.length > 0 && "--debug".equals(args[0])) {
      debug = true;
      argOffset = 1;
    }
    VulkanLoaderBootstrap.bootstrap(debug);

    Path fixtureRoot =
        args.length > argOffset
            ? Path.of(args[argOffset])
            : Path.of("..", "MeshForge", "fixtures", "baseline");

    List<String> fixtureFiles = List.of("RevitHouse.obj", "lucy.obj", "xyzrgb_dragon.obj");

    RuntimeGeometryLoader loader =
        new RuntimeGeometryLoader(MeshLoaders.defaultsFast(), Packers.realtimeFast());
    Path harnessCacheDirectory = Files.createTempDirectory("dynamisgpu-ingest-cache-");
    harnessCacheDirectory.toFile().deleteOnExit();

    MeshForgeIngestionHarnessRunner fullRunner = null;
    MeshForgeIngestionHarnessRunner preuploadRunner =
        new MeshForgeIngestionHarnessRunner(loader, plan -> {
          throw new UnsupportedOperationException("upload unavailable in preupload-only mode");
        }, new DefaultGeometryUploadValidation(), harnessCacheDirectory);

    String uploadBlockedReason = null;
    VulkanHarnessContext context = null;
    try {
      context = VulkanHarnessContext.create();
      VulkanGpuUploadExecutor executor =
          new VulkanGpuUploadExecutor(
              context.device(),
              context.physicalDevice(),
              context.commandPool(),
              context.graphicsQueue());
      fullRunner =
          new MeshForgeIngestionHarnessRunner(
              loader, executor, new DefaultGeometryUploadValidation(), harnessCacheDirectory);
    } catch (Throwable t) {
      uploadBlockedReason = summarizeThrowable(t);
      System.err.println("[MeshForgeVulkanIngestionMain] upload path blocked: " + uploadBlockedReason);
    }

    try {
      for (String fixture : fixtureFiles) {
        Path fixturePath = fixtureRoot.resolve(fixture);
        if (!Files.isRegularFile(fixturePath)) {
          System.out.println("fixture=" + fixture + " status=SKIPPED reason=missing_file");
          continue;
        }
        MeshForgeIngestionRunReport report;
        try {
          if (fullRunner != null) {
            report = fullRunner.run(fixture, fixturePath);
          } else {
            report = preuploadRunner.runPreuploadOnly(fixture, fixturePath, uploadBlockedReason);
          }
        } catch (Throwable fixtureFailure) {
          report = new MeshForgeIngestionRunReport(
              fixture,
              MeshForgeIngestionStatus.FAILURE,
              null,
              new MeshForgeIngestionTiming(0L, 0L, -1L, 0L),
              null,
              summarizeThrowable(fixtureFailure));
        }
        System.out.println(MeshForgeIngestionReportFormatter.toLine(report));
      }
    } finally {
      if (context != null) {
        context.close();
      }
    }
  }

  private static String summarizeThrowable(Throwable throwable) {
    String type = throwable.getClass().getSimpleName();
    String message = throwable.getMessage();
    if (message == null || message.isBlank()) {
      return type;
    }
    return type + ": " + message;
  }
}
