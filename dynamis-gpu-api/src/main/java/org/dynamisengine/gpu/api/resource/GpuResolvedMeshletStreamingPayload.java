package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Immutable GPU output payload for one resolved meshlet stream unit.
 *
 * <p>Layout v1 (5 int32 values):
 * <ul>
 *   <li>streamUnitId</li>
 *   <li>meshletStart</li>
 *   <li>meshletCount</li>
 *   <li>payloadByteOffset</li>
 *   <li>payloadByteSize</li>
 * </ul>
 */
public final class GpuResolvedMeshletStreamingPayload {
  public static final int COMPONENTS = 5;
  public static final int BYTE_SIZE = COMPONENTS * Integer.BYTES;

  private final int streamUnitId;
  private final int meshletStart;
  private final int meshletCount;
  private final int payloadByteOffset;
  private final int payloadByteSize;
  private final ByteBuffer resolvedBytes;

  private GpuResolvedMeshletStreamingPayload(
      int streamUnitId,
      int meshletStart,
      int meshletCount,
      int payloadByteOffset,
      int payloadByteSize,
      ByteBuffer resolvedBytes) {
    this.streamUnitId = streamUnitId;
    this.meshletStart = meshletStart;
    this.meshletCount = meshletCount;
    this.payloadByteOffset = payloadByteOffset;
    this.payloadByteSize = payloadByteSize;
    this.resolvedBytes = resolvedBytes;
  }

  public static GpuResolvedMeshletStreamingPayload of(
      int streamUnitId, int meshletStart, int meshletCount, int payloadByteOffset, int payloadByteSize) {
    if (streamUnitId < 0) {
      throw new IllegalArgumentException("streamUnitId must be >= 0");
    }
    if (meshletStart < 0) {
      throw new IllegalArgumentException("meshletStart must be >= 0");
    }
    if (meshletCount <= 0) {
      throw new IllegalArgumentException("meshletCount must be > 0");
    }
    if (payloadByteOffset < 0) {
      throw new IllegalArgumentException("payloadByteOffset must be >= 0");
    }
    if (payloadByteSize <= 0) {
      throw new IllegalArgumentException("payloadByteSize must be > 0");
    }

    ByteBuffer bytes = ByteBuffer.allocate(BYTE_SIZE).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(streamUnitId);
    bytes.putInt(meshletStart);
    bytes.putInt(meshletCount);
    bytes.putInt(payloadByteOffset);
    bytes.putInt(payloadByteSize);
    bytes.flip();

    return new GpuResolvedMeshletStreamingPayload(
        streamUnitId,
        meshletStart,
        meshletCount,
        payloadByteOffset,
        payloadByteSize,
        bytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN));
  }

  public static GpuResolvedMeshletStreamingPayload fromLittleEndianBytes(ByteBuffer sourceBytes) {
    if (sourceBytes == null) {
      throw new NullPointerException("sourceBytes");
    }
    if (sourceBytes.remaining() != BYTE_SIZE) {
      throw new IllegalArgumentException(
          "resolved payload byte size mismatch: expected=" + BYTE_SIZE + " actual=" + sourceBytes.remaining());
    }

    ByteBuffer src = sourceBytes.duplicate().order(sourceBytes.order());
    ByteBuffer normalized = ByteBuffer.allocate(BYTE_SIZE).order(ByteOrder.LITTLE_ENDIAN);
    if (src.order() == ByteOrder.LITTLE_ENDIAN) {
      normalized.put(src);
    } else {
      ByteBuffer big = src.slice().order(ByteOrder.BIG_ENDIAN);
      for (int i = 0; i < COMPONENTS; i++) {
        normalized.putInt(big.getInt());
      }
    }
    normalized.flip();

    ByteBuffer read = normalized.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
    int streamUnitId = read.getInt();
    int meshletStart = read.getInt();
    int meshletCount = read.getInt();
    int payloadByteOffset = read.getInt();
    int payloadByteSize = read.getInt();
    return of(streamUnitId, meshletStart, meshletCount, payloadByteOffset, payloadByteSize);
  }

  public int streamUnitId() {
    return streamUnitId;
  }

  public int meshletStart() {
    return meshletStart;
  }

  public int meshletCount() {
    return meshletCount;
  }

  public int payloadByteOffset() {
    return payloadByteOffset;
  }

  public int payloadByteSize() {
    return payloadByteSize;
  }

  public int byteSize() {
    return BYTE_SIZE;
  }

  /** Returns a read-only little-endian byte buffer view. */
  public ByteBuffer resolvedBytes() {
    return resolvedBytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
  }
}

