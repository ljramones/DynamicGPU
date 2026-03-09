package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Immutable GPU upload payload contract for meshlet bounds.
 *
 * <p>Layout v1:
 * <ul>
 *   <li>per-meshlet element width: 6 floats</li>
 *   <li>float order: minX, minY, minZ, maxX, maxY, maxZ</li>
 *   <li>little-endian packed byte payload</li>
 * </ul>
 */
public final class GpuMeshletBoundsPayload {
  public static final int BOUNDS_COMPONENTS = 6;

  private final int meshletCount;
  private final int boundsOffsetFloats;
  private final int boundsStrideFloats;
  private final ByteBuffer boundsBytes;

  private GpuMeshletBoundsPayload(
      int meshletCount, int boundsOffsetFloats, int boundsStrideFloats, ByteBuffer boundsBytes) {
    this.meshletCount = meshletCount;
    this.boundsOffsetFloats = boundsOffsetFloats;
    this.boundsStrideFloats = boundsStrideFloats;
    this.boundsBytes = boundsBytes;
  }

  public static GpuMeshletBoundsPayload fromLittleEndianBytes(
      int meshletCount, int boundsOffsetFloats, int boundsStrideFloats, ByteBuffer sourceBytes) {
    if (meshletCount < 0) {
      throw new IllegalArgumentException("meshletCount must be >= 0");
    }
    if (boundsOffsetFloats < 0) {
      throw new IllegalArgumentException("boundsOffsetFloats must be >= 0");
    }
    if (boundsStrideFloats < BOUNDS_COMPONENTS) {
      throw new IllegalArgumentException("boundsStrideFloats must be >= " + BOUNDS_COMPONENTS);
    }
    if (sourceBytes == null) {
      throw new NullPointerException("sourceBytes");
    }
    if ((sourceBytes.remaining() % Float.BYTES) != 0) {
      throw new IllegalArgumentException("sourceBytes remaining must be float-aligned");
    }

    int floatCount = sourceBytes.remaining() / Float.BYTES;
    int expectedFloatCount = expectedBoundsFloatCount(meshletCount, boundsOffsetFloats, boundsStrideFloats);
    if (floatCount != expectedFloatCount) {
      throw new IllegalArgumentException(
          "bounds payload float count mismatch: expected="
              + expectedFloatCount
              + " actual="
              + floatCount);
    }

    ByteBuffer src = sourceBytes.duplicate();
    ByteBuffer copy = ByteBuffer.allocate(src.remaining()).order(ByteOrder.LITTLE_ENDIAN);
    if (src.order() == ByteOrder.LITTLE_ENDIAN) {
      copy.put(src);
    } else {
      ByteBuffer normalized = src.slice().order(ByteOrder.BIG_ENDIAN);
      while (normalized.remaining() >= Float.BYTES) {
        copy.putFloat(normalized.getFloat());
      }
    }
    copy.flip();
    return new GpuMeshletBoundsPayload(
        meshletCount, boundsOffsetFloats, boundsStrideFloats, copy.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN));
  }

  public int meshletCount() {
    return meshletCount;
  }

  public int boundsOffsetFloats() {
    return boundsOffsetFloats;
  }

  public int boundsStrideFloats() {
    return boundsStrideFloats;
  }

  public int boundsStrideBytes() {
    return boundsStrideFloats * Float.BYTES;
  }

  public int boundsFloatCount() {
    return boundsBytes.remaining() / Float.BYTES;
  }

  public int expectedBoundsFloatCount() {
    return expectedBoundsFloatCount(meshletCount, boundsOffsetFloats, boundsStrideFloats);
  }

  public int boundsByteSize() {
    return boundsBytes.remaining();
  }

  /** Returns a read-only little-endian byte buffer view. */
  public ByteBuffer boundsBytes() {
    return boundsBytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static int expectedBoundsFloatCount(
      int meshletCount, int boundsOffsetFloats, int boundsStrideFloats) {
    return meshletCount == 0 ? 0 : boundsOffsetFloats + (meshletCount * boundsStrideFloats);
  }
}

