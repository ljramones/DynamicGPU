package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;

/**
 * Immutable output payload for per-meshlet visibility flags.
 *
 * <p>Layout v1: one byte per meshlet ({@code 1 = visible}, {@code 0 = culled}).
 */
public final class GpuMeshletVisibilityFlagsPayload {
  private final int meshletCount;
  private final ByteBuffer flagsBytes;

  private GpuMeshletVisibilityFlagsPayload(int meshletCount, ByteBuffer flagsBytes) {
    this.meshletCount = meshletCount;
    this.flagsBytes = flagsBytes;
  }

  public static GpuMeshletVisibilityFlagsPayload fromBytes(int meshletCount, ByteBuffer sourceBytes) {
    if (meshletCount < 0) {
      throw new IllegalArgumentException("meshletCount must be >= 0");
    }
    if (sourceBytes == null) {
      throw new NullPointerException("sourceBytes");
    }
    int expectedByteCount = meshletCount;
    int actualByteCount = sourceBytes.remaining();
    if (actualByteCount != expectedByteCount) {
      throw new IllegalArgumentException(
          "flags payload byte count mismatch: expected="
              + expectedByteCount
              + " actual="
              + actualByteCount);
    }

    ByteBuffer src = sourceBytes.duplicate();
    ByteBuffer copy = ByteBuffer.allocate(src.remaining());
    copy.put(src);
    copy.flip();
    return new GpuMeshletVisibilityFlagsPayload(meshletCount, copy.asReadOnlyBuffer());
  }

  public int meshletCount() {
    return meshletCount;
  }

  public int flagsByteSize() {
    return flagsBytes.remaining();
  }

  public ByteBuffer flagsBytes() {
    return flagsBytes.asReadOnlyBuffer();
  }
}

