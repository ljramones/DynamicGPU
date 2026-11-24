package org.dynamisengine.gpu.shaders;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Rebuilding DynamisGPU's own Slang kernels gives the committed bytes: the build is deterministic and the committed
 * outputs are current. Needs slangc, so it is skipped where there is none (it runs on the Apple GPU runner).
 */
class ShaderBuildTest {
  private static final Path SRC = Path.of("../dynamis-gpu-api/src/main/slang");
  private static final Path COMMITTED = Path.of("../dynamis-gpu-api/src/main/resources/org/dynamisengine/gpu/api/compute/spi/kernels");

  @Test
  void rebuildsTheCommittedKernels(@TempDir Path out) throws Exception {
    var slangc = ShaderBuild.findSlangc();
    Assumptions.assumeTrue(slangc.isPresent(), "slangc not available");
    ShaderBuild.build(new ShaderBuild.Config(SRC, out, "kernels.properties", List.of(), List.of(SRC),
        ShaderBuild.readKernels(SRC.resolve("kernels.list")), slangc.get()));
    for (String f : List.of("fma_probe.spv", "fma_probe.metal", "kernels.properties")) {
      assertArrayEquals(Files.readAllBytes(COMMITTED.resolve(f)), Files.readAllBytes(out.resolve(f)), f);
    }
  }

  @Test
  void sourceHashMatchesTheManifest() throws Exception {
    String manifest = Files.readString(COMMITTED.resolve("kernels.properties"));
    assertTrue(manifest.contains("source.sha256=" + ShaderBuild.sourceHash(List.of(SRC)) + "\n"));
  }
}
