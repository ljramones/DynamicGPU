package org.dynamisengine.gpu.shaders;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Rebuilding DynamisGPU's Slang kernels (its own fma probe and the capability-tier conformance kernels) gives the
 * committed bytes: the build is deterministic and the committed outputs are current. Needs slangc, so the rebuild is
 * skipped where there is none (it runs on the Apple GPU runner).
 */
class ShaderBuildTest {
  record Library(Path src, Path committed) { }

  static Stream<Library> libraries() {
    return Stream.of(
        new Library(Path.of("../dynamis-gpu-api/src/main/slang"),
            Path.of("../dynamis-gpu-api/src/main/resources/org/dynamisengine/gpu/api/compute/spi/kernels")),
        new Library(Path.of("../dynamis-gpu-test/src/main/slang"),
            Path.of("../dynamis-gpu-test/src/main/resources/compute/tiers")));
  }

  @ParameterizedTest
  @MethodSource("libraries")
  void rebuildsTheCommittedKernels(Library lib, @TempDir Path out) throws Exception {
    var slangc = ShaderBuild.findSlangc();
    Assumptions.assumeTrue(slangc.isPresent(), "slangc not available");
    ShaderBuild.build(new ShaderBuild.Config(lib.src(), out, "kernels.properties", List.of(), List.of(lib.src()),
        ShaderBuild.readKernels(lib.src().resolve("kernels.list")), slangc.get()));
    List<Path> committed;
    try (Stream<Path> files = Files.list(lib.committed())) { committed = files.toList(); }
    for (Path f : committed) {
      assertArrayEquals(Files.readAllBytes(f), Files.readAllBytes(out.resolve(f.getFileName())), f.toString());
    }
    try (Stream<Path> built = Files.list(out)) {
      assertEquals(committed.size(), built.count(), "the build writes exactly the committed files");
    }
  }

  @ParameterizedTest
  @MethodSource("libraries")
  void sourceHashMatchesTheManifest(Library lib) throws Exception {
    String manifest = Files.readString(lib.committed().resolve("kernels.properties"));
    assertTrue(manifest.contains("source.sha256=" + ShaderBuild.sourceHash(List.of(lib.src())) + "\n"));
  }

  @Test
  void readsTargets(@TempDir Path dir) throws Exception {
    Path list = dir.resolve("kernels.list");
    Files.writeString(list, "a a.slang a\nb b.slang b spirv # comment\nc c.slang c metal,spirv\n");
    var k = ShaderBuild.readKernels(list);
    assertEquals(new ShaderBuild.Kernel("a", "a.slang", "a", true, true), k.get(0));
    assertEquals(new ShaderBuild.Kernel("b", "b.slang", "b", true, false), k.get(1));
    assertEquals(new ShaderBuild.Kernel("c", "c.slang", "c", true, true), k.get(2));
    Files.writeString(list, "d d.slang d glsl\n");
    assertThrows(IllegalArgumentException.class, () -> ShaderBuild.readKernels(list));
  }
}
