package org.dynamisengine.gpu.shaders;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

class SpirvTest {
  private static int op(int opcode, int words) { return (words << 16) | opcode; }

  /** A tiny module: capability, GLSL.std.450 import, a float type, then FAdd (%10), FMul (%11), Fma (%12). */
  private static byte[] module() {
    List<Integer> w = new ArrayList<>(List.of(Spirv.MAGIC, 0x00010000, 0, 20, 0));
    w.addAll(List.of(op(17, 2), 1));                                         // OpCapability Shader
    byte[] name = "GLSL.std.450\0\0\0\0".getBytes(StandardCharsets.US_ASCII); // 16 bytes = 4 words
    ByteBuffer nb = ByteBuffer.wrap(name).order(ByteOrder.LITTLE_ENDIAN);
    w.addAll(List.of(op(11, 6), 1, nb.getInt(), nb.getInt(), nb.getInt(), nb.getInt())); // %1 = OpExtInstImport
    w.addAll(List.of(op(22, 3), 2, 32));                                     // %2 = OpTypeFloat 32
    w.addAll(List.of(op(129, 5), 2, 10, 3, 4));                              // %10 = OpFAdd
    w.addAll(List.of(op(133, 5), 2, 11, 3, 4));                              // %11 = OpFMul
    w.addAll(List.of(op(12, 8), 2, 12, 1, 50, 3, 4, 5));                     // %12 = OpExtInst Fma
    ByteBuffer b = ByteBuffer.allocate(w.size() * 4).order(ByteOrder.LITTLE_ENDIAN);
    w.forEach(b::putInt);
    return b.array();
  }

  private static Set<Integer> noContractionTargets(byte[] spv) {
    int[] words = new int[spv.length / 4];
    ByteBuffer.wrap(spv).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(words);
    Set<Integer> t = new HashSet<>();
    for (int i = 5; i < words.length; i += words[i] >>> 16) {
      if ((words[i] & 0xFFFF) == 71 && words[i + 2] == 42) t.add(words[i + 1]);
    }
    return t;
  }

  @Test
  void marksAddsSubtractsAndFmaButNotMultiplies() {
    assertEquals(Set.of(10, 12), noContractionTargets(Spirv.noContraction(module())));
  }

  @Test
  void isIdempotent() {
    byte[] once = Spirv.noContraction(module());
    assertArrayEquals(once, Spirv.noContraction(once));
  }

  @Test
  void rejectsWhatIsNotSpirv() {
    assertThrows(IllegalArgumentException.class, () -> Spirv.noContraction(new byte[20]));
    assertThrows(IllegalArgumentException.class, () -> Spirv.noContraction(new byte[7]));
  }
}
