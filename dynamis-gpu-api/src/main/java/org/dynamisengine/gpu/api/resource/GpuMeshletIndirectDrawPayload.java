package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

/**
 * Immutable indexed-indirect draw command payload for visible meshlets.
 *
 * <p>Layout v1: packed {@code VkDrawIndexedIndirectCommand} fields per command:
 * {@code indexCount, instanceCount, firstIndex, vertexOffset, firstInstance}.
 */
public final class GpuMeshletIndirectDrawPayload {
  public static final int COMMAND_COMPONENTS = 5;
  public static final int COMMAND_STRIDE_BYTES = COMMAND_COMPONENTS * Integer.BYTES;

  private final int sourceVisibleMeshletCount;
  private final int commandCount;
  private final ByteBuffer commandBytes;

  private GpuMeshletIndirectDrawPayload(
      int sourceVisibleMeshletCount, int commandCount, ByteBuffer commandBytes) {
    this.sourceVisibleMeshletCount = sourceVisibleMeshletCount;
    this.commandCount = commandCount;
    this.commandBytes = commandBytes;
  }

  public static GpuMeshletIndirectDrawPayload fromLittleEndianBytes(
      int sourceVisibleMeshletCount, int commandCount, ByteBuffer sourceBytes) {
    if (sourceVisibleMeshletCount < 0) {
      throw new IllegalArgumentException("sourceVisibleMeshletCount must be >= 0");
    }
    if (commandCount < 0) {
      throw new IllegalArgumentException("commandCount must be >= 0");
    }
    if (commandCount > sourceVisibleMeshletCount) {
      throw new IllegalArgumentException("commandCount must be <= sourceVisibleMeshletCount");
    }
    if (sourceBytes == null) {
      throw new NullPointerException("sourceBytes");
    }
    if ((sourceBytes.remaining() % Integer.BYTES) != 0) {
      throw new IllegalArgumentException("sourceBytes remaining must be int-aligned");
    }
    int expectedBytes = commandCount * COMMAND_STRIDE_BYTES;
    if (sourceBytes.remaining() != expectedBytes) {
      throw new IllegalArgumentException(
          "indirect command byte count mismatch: expected="
              + expectedBytes
              + " actual="
              + sourceBytes.remaining());
    }

    ByteBuffer src = sourceBytes.duplicate().order(sourceBytes.order());
    ByteBuffer copy = ByteBuffer.allocate(src.remaining()).order(ByteOrder.LITTLE_ENDIAN);
    if (src.order() == ByteOrder.LITTLE_ENDIAN) {
      copy.put(src);
    } else {
      IntBuffer ints = src.slice().order(ByteOrder.BIG_ENDIAN).asIntBuffer();
      while (ints.hasRemaining()) {
        copy.putInt(ints.get());
      }
    }
    copy.flip();
    return new GpuMeshletIndirectDrawPayload(
        sourceVisibleMeshletCount, commandCount, copy.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN));
  }

  public int sourceVisibleMeshletCount() {
    return sourceVisibleMeshletCount;
  }

  public int commandCount() {
    return commandCount;
  }

  public int commandByteSize() {
    return commandBytes.remaining();
  }

  public ByteBuffer commandBytes() {
    return commandBytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
  }
}

