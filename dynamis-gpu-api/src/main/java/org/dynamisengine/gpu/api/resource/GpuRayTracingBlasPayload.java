package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Immutable GPU payload describing BLAS build-preparation input metadata.
 *
 * <p>Layout v1 (4 int32 values):
 * <ul>
 *   <li>regionCount</li>
 *   <li>regionsStrideBytes</li>
 *   <li>regionsByteSize</li>
 *   <li>reservedFlags</li>
 * </ul>
 */
public final class GpuRayTracingBlasPayload {
  public static final int COMPONENTS = 4;
  public static final int BYTE_SIZE = COMPONENTS * Integer.BYTES;

  private final int regionCount;
  private final int regionsStrideBytes;
  private final int regionsByteSize;
  private final int reservedFlags;
  private final ByteBuffer buildPrepBytes;

  private GpuRayTracingBlasPayload(
      int regionCount,
      int regionsStrideBytes,
      int regionsByteSize,
      int reservedFlags,
      ByteBuffer buildPrepBytes) {
    this.regionCount = regionCount;
    this.regionsStrideBytes = regionsStrideBytes;
    this.regionsByteSize = regionsByteSize;
    this.reservedFlags = reservedFlags;
    this.buildPrepBytes = buildPrepBytes;
  }

  public static GpuRayTracingBlasPayload of(
      int regionCount, int regionsStrideBytes, int regionsByteSize, int reservedFlags) {
    if (regionCount <= 0) {
      throw new IllegalArgumentException("regionCount must be > 0");
    }
    if (regionsStrideBytes <= 0) {
      throw new IllegalArgumentException("regionsStrideBytes must be > 0");
    }
    if ((regionsStrideBytes % Integer.BYTES) != 0) {
      throw new IllegalArgumentException("regionsStrideBytes must be int-aligned");
    }
    if (regionsByteSize <= 0) {
      throw new IllegalArgumentException("regionsByteSize must be > 0");
    }
    if ((regionsByteSize % Integer.BYTES) != 0) {
      throw new IllegalArgumentException("regionsByteSize must be int-aligned");
    }
    if (reservedFlags < 0) {
      throw new IllegalArgumentException("reservedFlags must be >= 0");
    }

    ByteBuffer bytes = ByteBuffer.allocate(BYTE_SIZE).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(regionCount);
    bytes.putInt(regionsStrideBytes);
    bytes.putInt(regionsByteSize);
    bytes.putInt(reservedFlags);
    bytes.flip();
    return new GpuRayTracingBlasPayload(
        regionCount,
        regionsStrideBytes,
        regionsByteSize,
        reservedFlags,
        bytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN));
  }

  public static GpuRayTracingBlasPayload forGeometryResource(GpuRayTracingGeometryResource geometryResource) {
    if (geometryResource == null) {
      throw new NullPointerException("geometryResource");
    }
    return of(
        geometryResource.regionCount(),
        geometryResource.regionsStrideBytes(),
        geometryResource.regionsByteSize(),
        0);
  }

  public int regionCount() {
    return regionCount;
  }

  public int regionsStrideBytes() {
    return regionsStrideBytes;
  }

  public int regionsByteSize() {
    return regionsByteSize;
  }

  public int reservedFlags() {
    return reservedFlags;
  }

  public int byteSize() {
    return BYTE_SIZE;
  }

  /** Returns a read-only little-endian byte buffer view. */
  public ByteBuffer buildPrepBytes() {
    return buildPrepBytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
  }
}

