package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Immutable GPU upload payload contract for meshlet streaming units.
 *
 * <p>Layout v1:
 * <ul>
 *   <li>per-unit element width: 5 int32 values</li>
 *   <li>int order: streamUnitId, meshletStart, meshletCount, payloadByteOffset, payloadByteSize</li>
 *   <li>little-endian packed byte payload</li>
 * </ul>
 */
public final class GpuMeshletStreamingPayload {
  public static final int STREAM_UNIT_COMPONENTS = 5;

  private final int streamUnitCount;
  private final int streamUnitsOffsetInts;
  private final int streamUnitsStrideInts;
  private final ByteBuffer streamUnitsBytes;

  private GpuMeshletStreamingPayload(
      int streamUnitCount, int streamUnitsOffsetInts, int streamUnitsStrideInts, ByteBuffer streamUnitsBytes) {
    this.streamUnitCount = streamUnitCount;
    this.streamUnitsOffsetInts = streamUnitsOffsetInts;
    this.streamUnitsStrideInts = streamUnitsStrideInts;
    this.streamUnitsBytes = streamUnitsBytes;
  }

  public static GpuMeshletStreamingPayload fromLittleEndianBytes(
      int streamUnitCount, int streamUnitsOffsetInts, int streamUnitsStrideInts, ByteBuffer sourceBytes) {
    if (streamUnitCount < 0) {
      throw new IllegalArgumentException("streamUnitCount must be >= 0");
    }
    if (streamUnitsOffsetInts < 0) {
      throw new IllegalArgumentException("streamUnitsOffsetInts must be >= 0");
    }
    if (streamUnitsStrideInts < STREAM_UNIT_COMPONENTS) {
      throw new IllegalArgumentException("streamUnitsStrideInts must be >= " + STREAM_UNIT_COMPONENTS);
    }
    if (sourceBytes == null) {
      throw new NullPointerException("sourceBytes");
    }
    if ((sourceBytes.remaining() % Integer.BYTES) != 0) {
      throw new IllegalArgumentException("sourceBytes remaining must be int-aligned");
    }

    int intCount = sourceBytes.remaining() / Integer.BYTES;
    int expectedIntCount = expectedStreamUnitsIntCount(streamUnitCount, streamUnitsOffsetInts, streamUnitsStrideInts);
    if (intCount != expectedIntCount) {
      throw new IllegalArgumentException(
          "stream units payload int count mismatch: expected="
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
    return new GpuMeshletStreamingPayload(
        streamUnitCount,
        streamUnitsOffsetInts,
        streamUnitsStrideInts,
        copy.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN));
  }

  public int streamUnitCount() {
    return streamUnitCount;
  }

  public int streamUnitsOffsetInts() {
    return streamUnitsOffsetInts;
  }

  public int streamUnitsStrideInts() {
    return streamUnitsStrideInts;
  }

  public int streamUnitsStrideBytes() {
    return streamUnitsStrideInts * Integer.BYTES;
  }

  public int streamUnitsIntCount() {
    return streamUnitsBytes.remaining() / Integer.BYTES;
  }

  public int expectedStreamUnitsIntCount() {
    return expectedStreamUnitsIntCount(streamUnitCount, streamUnitsOffsetInts, streamUnitsStrideInts);
  }

  public int streamUnitsByteSize() {
    return streamUnitsBytes.remaining();
  }

  /** Returns a read-only little-endian byte buffer view. */
  public ByteBuffer streamUnitsBytes() {
    return streamUnitsBytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static int expectedStreamUnitsIntCount(
      int streamUnitCount, int streamUnitsOffsetInts, int streamUnitsStrideInts) {
    return streamUnitCount == 0 ? 0 : streamUnitsOffsetInts + (streamUnitCount * streamUnitsStrideInts);
  }
}
