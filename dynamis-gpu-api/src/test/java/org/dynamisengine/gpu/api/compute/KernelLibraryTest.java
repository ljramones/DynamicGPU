package org.dynamisengine.gpu.api.compute;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.dynamisengine.gpu.api.compute.spi.ComputeDevice;
import org.junit.jupiter.api.Test;

/** KernelLibrary over DynamisGPU's own Slang kernels (fma_probe), and that they were rebuilt after the last edit. */
class KernelLibraryTest {
  private static final KernelLibrary LIB = KernelLibrary.load(ComputeDevice.class, "kernels", "kernels.properties", "dynamisgpu.");

  @Test
  void readsTheManifest() {
    assertEquals(Set.of("fma_probe"), LIB.kernels());
    assertEquals("fma_probe", LIB.entryPoint("fma_probe"));
    assertEquals(new Dimensions(1, 1, 1), LIB.localSize("fma_probe"));
    assertEquals(List.of("o"), LIB.buffers("fma_probe"));
    assertEquals(Map.of("x", 0, "c", 4), LIB.parameterOffsets("fma_probe"));
    assertEquals(Map.of("x", "float32", "c", "float32"), LIB.parameterTypes("fma_probe"));
    assertEquals(8, LIB.parameterBytes("fma_probe"));
  }

  @Test
  void buildsBundlesAndPipelineDescriptors() {
    var bundle = LIB.bundle("fma_probe");
    assertEquals("dynamisgpu.fma_probe", bundle.name());
    assertEquals(LIB.sourceHash().substring(0, 16), bundle.version());
    assertEquals(KernelBundle.Format.SPIRV, bundle.executable(Backend.VULKAN).format());
    assertTrue(new String(bundle.executable(Backend.METAL).bytes(), StandardCharsets.UTF_8)
        .startsWith("#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n"));
    var d = LIB.pipeline("fma_probe", List.of(Access.WRITE), Set.of("FP32"), Set.of());
    assertEquals(8, d.layout().parameterBytes());
    assertEquals(1, d.layout().buffers().size());
    assertThrows(IllegalArgumentException.class, () -> LIB.pipeline("fma_probe", List.of(), Set.of(), Set.of()));
    assertThrows(IllegalArgumentException.class, () -> LIB.entryPoint("missing"));
  }

  /** The committed SPIR-V and Metal are the build of the current src/main/slang (scripts/build-slang-kernels.sh). */
  @Test
  void compiledKernelsAreUpToDate() throws Exception {
    List<Path> files = new ArrayList<>();
    try (Stream<Path> s = Files.list(Path.of("src/main/slang"))) {
      files.addAll(s.filter(p -> p.toString().endsWith(".slang")).toList());
    }
    files.sort((a, b) -> a.toAbsolutePath().toString().compareTo(b.toAbsolutePath().toString()));
    MessageDigest h = MessageDigest.getInstance("SHA-256");
    for (Path f : files) {
      h.update(f.getFileName().toString().getBytes(StandardCharsets.UTF_8)); h.update((byte) 0);
      h.update(Files.readAllBytes(f)); h.update((byte) 0);
    }
    assertEquals(LIB.sourceHash(), HexFormat.of().formatHex(h.digest()),
        "src/main/slang changed since the kernels were built: run scripts/build-slang-kernels.sh");
  }
}
