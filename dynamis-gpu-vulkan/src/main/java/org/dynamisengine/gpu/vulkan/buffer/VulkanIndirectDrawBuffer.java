package org.dynamisengine.gpu.vulkan.buffer;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;
import org.dynamisengine.gpu.api.gpu.IndirectCommandBuffer;

/**
 * Vulkan implementation of an indirect draw command buffer with variant buckets.
 */
public final class VulkanIndirectDrawBuffer implements IndirectCommandBuffer {
  private static final AtomicLong NEXT_HANDLE = new AtomicLong(1000);

  private final int[] variantCapacities;
  private final int[] variantOffsets;
  private final int[][] commands;
  private final long bufferHandle;
  private final long countBufferHandle;
  private boolean destroyed;

  /**
   * Creates an indirect buffer and partitions it into variant capacity ranges.
   *
   * @param totalCapacity total command slots
   * @param variantCapacities per-variant slot capacities
   */
  public VulkanIndirectDrawBuffer(int totalCapacity, int[] variantCapacities) {
    if (totalCapacity <= 0) {
      throw new IllegalArgumentException("totalCapacity must be > 0");
    }
    if (variantCapacities == null || variantCapacities.length == 0) {
      throw new IllegalArgumentException("variantCapacities must not be empty");
    }

    this.variantCapacities = Arrays.copyOf(variantCapacities, variantCapacities.length);
    this.variantOffsets = new int[variantCapacities.length];
    int offset = 0;
    int capacitySum = 0;
    for (int i = 0; i < variantCapacities.length; i++) {
      if (variantCapacities[i] < 0) {
        throw new IllegalArgumentException("variant capacity must be >= 0");
      }
      variantOffsets[i] = offset;
      offset += variantCapacities[i];
      capacitySum += variantCapacities[i];
    }
    if (capacitySum > totalCapacity) {
      throw new IllegalArgumentException("sum of variant capacities exceeds totalCapacity");
    }

    this.commands = new int[totalCapacity][6];
    this.bufferHandle = NEXT_HANDLE.getAndIncrement();
    this.countBufferHandle = NEXT_HANDLE.getAndIncrement();
  }

  @Override
  public void writeCommand(
      int slot,
      int indexCount,
      int instanceCount,
      int firstIndex,
      int vertexOffset,
      int firstInstance) {
    ensureAlive();
    if (slot < 0 || slot >= commands.length) {
      throw new IndexOutOfBoundsException("slot out of range: " + slot);
    }
    commands[slot][0] = indexCount;
    commands[slot][1] = instanceCount;
    commands[slot][2] = firstIndex;
    commands[slot][3] = vertexOffset;
    commands[slot][4] = firstInstance;
    commands[slot][5] = slot;
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
    validateVariantIndex(variantIndex);
    return variantOffsets[variantIndex];
  }

  @Override
  public int variantCapacity(int variantIndex) {
    validateVariantIndex(variantIndex);
    return variantCapacities[variantIndex];
  }

  @Override
  public void destroy() {
    destroyed = true;
  }

  private void validateVariantIndex(int variantIndex) {
    if (variantIndex < 0 || variantIndex >= variantOffsets.length) {
      throw new IndexOutOfBoundsException("variantIndex out of range: " + variantIndex);
    }
  }

  private void ensureAlive() {
    if (destroyed) {
      throw new IllegalStateException("Buffer has been destroyed");
    }
  }
}
