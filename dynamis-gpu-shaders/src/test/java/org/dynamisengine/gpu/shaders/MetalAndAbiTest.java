package org.dynamisengine.gpu.shaders;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.junit.jupiter.api.Test;

class MetalAndAbiTest {
  private static final String REFLECTION = """
      {"parameters": [
        {"name": "x", "binding": {"kind": "descriptorTableSlot", "index": 0}, "type": {"kind": "resource"}},
        {"name": "y", "binding": {"kind": "descriptorTableSlot", "index": 1}, "type": {"kind": "resource"}},
        {"name": "p", "binding": {"kind": "pushConstantBuffer", "index": 0}, "type": {"kind": "constantBuffer",
          "elementType": {"kind": "struct", "fields": [
            {"name": "n", "type": {"kind": "scalar", "scalarType": "uint32"}, "binding": {"kind": "uniform", "offset": 0, "size": 4}},
            {"name": "alpha", "type": {"kind": "scalar", "scalarType": "float32"}, "binding": {"kind": "uniform", "offset": 4, "size": 4}}]}}}],
       "entryPoints": [{"name": "axpy", "stage": "compute", "threadGroupSize": [64, 1, 1]}]}
      """;

  @Test
  void readsTheAbiFromReflection() {
    var abi = KernelAbi.fromReflection(REFLECTION, "axpy");
    assertEquals(List.of(new KernelAbi.Buffer("x", 0), new KernelAbi.Buffer("y", 1)), abi.buffers());
    assertEquals("p", abi.parameterName());
    assertEquals(List.of(new KernelAbi.Parameter("n", 0, "uint32"), new KernelAbi.Parameter("alpha", 4, "float32")), abi.parameters());
    assertArrayEquals(new int[] {64, 1, 1}, abi.localSize());
    assertThrows(IllegalArgumentException.class, () -> KernelAbi.fromReflection(REFLECTION, "other"));
  }

  @Test
  void checksMetalBufferIndices() {
    var abi = KernelAbi.fromReflection(REFLECTION, "axpy");
    String ok = "[[kernel]] void axpy(uint3 id_0 [[thread_position_in_grid]], float device* x_1 [[buffer(0)]], "
        + "float device* y_1 [[buffer(1)]], Params_0 constant* p_1 [[buffer(2)]])\n{";
    Metal.checkBindings(ok, "axpy", abi);
    String swapped = ok.replace("[[buffer(2)]]", "[[buffer(9)]]").replace("x_1 [[buffer(0)]]", "x_1 [[buffer(2)]]")
        .replace("[[buffer(9)]]", "[[buffer(0)]]");
    assertThrows(IllegalStateException.class, () -> Metal.checkBindings(swapped, "axpy", abi));
    assertThrows(IllegalStateException.class, () -> Metal.checkBindings("no kernel here", "axpy", abi));
  }

  @Test
  void parsesJson() {
    assertEquals(List.of(1.0, "a\"b", true, false), Json.parse("[1, \"a\\\"b\", true, false]"));
    assertThrows(IllegalArgumentException.class, () -> Json.parse("[1] x"));
  }
}
