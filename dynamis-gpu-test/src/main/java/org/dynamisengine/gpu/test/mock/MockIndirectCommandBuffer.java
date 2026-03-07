package org.dynamisengine.gpu.test.mock;

import java.util.ArrayList;
import java.util.List;
import org.dynamisengine.gpu.api.gpu.IndirectCommandBuffer;

public final class MockIndirectCommandBuffer implements IndirectCommandBuffer {
  private final long bufferHandle;
  private final long countBufferHandle;
  private final int[] variantOffsets;
  private final int[] variantCapacities;
  private final List<CommandWrite> writes = new ArrayList<>();
  private boolean destroyed;

  public MockIndirectCommandBuffer(
      long bufferHandle, long countBufferHandle, int[] variantOffsets, int[] variantCapacities) {
    this.bufferHandle = bufferHandle;
    this.countBufferHandle = countBufferHandle;
    this.variantOffsets = variantOffsets.clone();
    this.variantCapacities = variantCapacities.clone();
  }

  @Override
  public void writeCommand(
      int slot,
      int indexCount,
      int instanceCount,
      int firstIndex,
      int vertexOffset,
      int firstInstance) {
    writes.add(new CommandWrite(slot, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance));
  }

  @Override
  public long bufferHandle() {
    return bufferHandle;
  }

  @Override
  public long countBufferHandle() {
    return countBufferHandle;
  }

  @Override
  public int variantOffset(int variantIndex) {
    return variantOffsets[variantIndex];
  }

  @Override
  public int variantCapacity(int variantIndex) {
    return variantCapacities[variantIndex];
  }

  @Override
  public void destroy() {
    destroyed = true;
  }

  public List<CommandWrite> writes() {
    return List.copyOf(writes);
  }

  public boolean destroyed() {
    return destroyed;
  }

  public record CommandWrite(
      int slot,
      int indexCount,
      int instanceCount,
      int firstIndex,
      int vertexOffset,
      int firstInstance) {}
}
