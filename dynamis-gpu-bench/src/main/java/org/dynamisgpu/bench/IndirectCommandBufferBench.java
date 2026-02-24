package org.dynamisgpu.bench;

import org.dynamisgpu.api.gpu.IndirectCommandBuffer;
import org.dynamisgpu.vulkan.buffer.VulkanIndirectDrawBuffer;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@State(Scope.Thread)
public class IndirectCommandBufferBench {
  private IndirectCommandBuffer buffer;

  @Setup
  public void setup() {
    buffer = new VulkanIndirectDrawBuffer(1024, new int[] {256, 256, 256, 256});
  }

  @Benchmark
  public void writeCommand() {
    buffer.writeCommand(0, 36, 1, 0, 0, 0);
  }
}
