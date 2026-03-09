package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

/**
 * Immutable compacted visible meshlet list payload.
 *
 * <p>Layout v1: little-endian {@code int32} meshlet indices.
 */
public final class GpuVisibleMeshletListPayload {
  private final int sourceMeshletCount;
  private final int visibleMeshletCount;
  private final ByteBuffer visibleIndicesBytes;

  private GpuVisibleMeshletListPayload(
      int sourceMeshletCount, int visibleMeshletCount, ByteBuffer visibleIndicesBytes) {
    this.sourceMeshletCount = sourceMeshletCount;
    this.visibleMeshletCount = visibleMeshletCount;
    this.visibleIndicesBytes = visibleIndicesBytes;
  }

  public static GpuVisibleMeshletListPayload fromLittleEndianBytes(
      int sourceMeshletCount, int visibleMeshletCount, ByteBuffer sourceBytes) {
    if (sourceMeshletCount < 0) {
      throw new IllegalArgumentException("sourceMeshletCount must be >= 0");
    }
    if (visibleMeshletCount < 0) {
      throw new IllegalArgumentException("visibleMeshletCount must be >= 0");
    }
    if (visibleMeshletCount > sourceMeshletCount) {
      throw new IllegalArgumentException("visibleMeshletCount must be <= sourceMeshletCount");
    }
    if (sourceBytes == null) {
      throw new NullPointerException("sourceBytes");
    }
    if ((sourceBytes.remaining() % Integer.BYTES) != 0) {
      throw new IllegalArgumentException("sourceBytes remaining must be int-aligned");
    }
    int expectedBytes = visibleMeshletCount * Integer.BYTES;
    if (sourceBytes.remaining() != expectedBytes) {
      throw new IllegalArgumentException(
          "visible index payload byte count mismatch: expected="
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
    return new GpuVisibleMeshletListPayload(
        sourceMeshletCount, visibleMeshletCount, copy.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN));
  }

  public int sourceMeshletCount() {
    return sourceMeshletCount;
  }

  public int visibleMeshletCount() {
    return visibleMeshletCount;
  }

  public int visibleIndicesByteSize() {
    return visibleIndicesBytes.remaining();
  }

  public ByteBuffer visibleIndicesBytes() {
    return visibleIndicesBytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
  }
}
