package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Immutable GPU upload payload contract for RT geometry regions.
 *
 * <p>Layout v1:
 * <ul>
 *   <li>per-region element width: 5 int32 values</li>
 *   <li>int order: submeshIndex, firstIndex, indexCount, materialSlot, flags</li>
 *   <li>little-endian packed byte payload</li>
 * </ul>
 */
public final class GpuRayTracingGeometryPayload {
  public static final int REGION_COMPONENTS = 5;

  private final int regionCount;
  private final int regionsOffsetInts;
  private final int regionsStrideInts;
  private final ByteBuffer regionsBytes;

  private GpuRayTracingGeometryPayload(
      int regionCount, int regionsOffsetInts, int regionsStrideInts, ByteBuffer regionsBytes) {
    this.regionCount = regionCount;
    this.regionsOffsetInts = regionsOffsetInts;
    this.regionsStrideInts = regionsStrideInts;
    this.regionsBytes = regionsBytes;
  }

  public static GpuRayTracingGeometryPayload fromLittleEndianBytes(
      int regionCount, int regionsOffsetInts, int regionsStrideInts, ByteBuffer sourceBytes) {
    if (regionCount < 0) {
      throw new IllegalArgumentException("regionCount must be >= 0");
    }
    if (regionsOffsetInts < 0) {
      throw new IllegalArgumentException("regionsOffsetInts must be >= 0");
    }
    if (regionsStrideInts < REGION_COMPONENTS) {
      throw new IllegalArgumentException("regionsStrideInts must be >= " + REGION_COMPONENTS);
    }
    if (sourceBytes == null) {
      throw new NullPointerException("sourceBytes");
    }
    if ((sourceBytes.remaining() % Integer.BYTES) != 0) {
      throw new IllegalArgumentException("sourceBytes remaining must be int-aligned");
    }

    int intCount = sourceBytes.remaining() / Integer.BYTES;
    int expectedIntCount = expectedRegionsIntCount(regionCount, regionsOffsetInts, regionsStrideInts);
    if (intCount != expectedIntCount) {
      throw new IllegalArgumentException(
          "RT regions payload int count mismatch: expected="
              + expectedIntCount
              + " actual="
              + intCount);
    }

    ByteBuffer src = sourceBytes.duplicate().order(sourceBytes.order());
    ByteBuffer copy = ByteBuffer.allocate(src.remaining()).order(ByteOrder.LITTLE_ENDIAN);
    if (src.order() == ByteOrder.LITTLE_ENDIAN) {
      copy.put(src);
    } else {
      ByteBuffer normalized = src.slice().order(ByteOrder.BIG_ENDIAN);
      while (normalized.remaining() >= Integer.BYTES) {
        copy.putInt(normalized.getInt());
      }
    }
    copy.flip();
    return new GpuRayTracingGeometryPayload(
        regionCount,
        regionsOffsetInts,
        regionsStrideInts,
        copy.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN));
  }

  public int regionCount() {
    return regionCount;
  }

  public int regionsOffsetInts() {
    return regionsOffsetInts;
  }

  public int regionsStrideInts() {
    return regionsStrideInts;
  }

  public int regionsStrideBytes() {
    return regionsStrideInts * Integer.BYTES;
  }

  public int regionsIntCount() {
    return regionsBytes.remaining() / Integer.BYTES;
  }

  public int expectedRegionsIntCount() {
    return expectedRegionsIntCount(regionCount, regionsOffsetInts, regionsStrideInts);
  }

  public int regionsByteSize() {
    return regionsBytes.remaining();
  }

  /** Returns a read-only little-endian byte buffer view. */
  public ByteBuffer regionsBytes() {
    return regionsBytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static int expectedRegionsIntCount(int regionCount, int regionsOffsetInts, int regionsStrideInts) {
    return regionCount == 0 ? 0 : regionsOffsetInts + (regionCount * regionsStrideInts);
  }
}

