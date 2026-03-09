package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Immutable GPU output payload for one selected meshlet LOD level.
 *
 * <p>Layout v1 (4 int32 values):
 * <ul>
 *   <li>selectedLodLevel</li>
 *   <li>meshletStart</li>
 *   <li>meshletCount</li>
 *   <li>geometricErrorBits ({@code Float.floatToRawIntBits(geometricError)})</li>
 * </ul>
 */
public final class GpuSelectedMeshletLodPayload {
  public static final int COMPONENTS = 4;
  public static final int BYTE_SIZE = COMPONENTS * Integer.BYTES;

  private final int selectedLodLevel;
  private final int meshletStart;
  private final int meshletCount;
  private final int geometricErrorBits;
  private final ByteBuffer selectedBytes;

  private GpuSelectedMeshletLodPayload(
      int selectedLodLevel,
      int meshletStart,
      int meshletCount,
      int geometricErrorBits,
      ByteBuffer selectedBytes) {
    this.selectedLodLevel = selectedLodLevel;
    this.meshletStart = meshletStart;
    this.meshletCount = meshletCount;
    this.geometricErrorBits = geometricErrorBits;
    this.selectedBytes = selectedBytes;
  }

  public static GpuSelectedMeshletLodPayload of(
      int selectedLodLevel, int meshletStart, int meshletCount, int geometricErrorBits) {
    if (selectedLodLevel < 0) {
      throw new IllegalArgumentException("selectedLodLevel must be >= 0");
    }
    if (meshletStart < 0) {
      throw new IllegalArgumentException("meshletStart must be >= 0");
    }
    if (meshletCount <= 0) {
      throw new IllegalArgumentException("meshletCount must be > 0");
    }

    ByteBuffer bytes = ByteBuffer.allocate(BYTE_SIZE).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(selectedLodLevel);
    bytes.putInt(meshletStart);
    bytes.putInt(meshletCount);
    bytes.putInt(geometricErrorBits);
    bytes.flip();

    return new GpuSelectedMeshletLodPayload(
        selectedLodLevel,
        meshletStart,
        meshletCount,
        geometricErrorBits,
        bytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN));
  }

  public static GpuSelectedMeshletLodPayload fromLittleEndianBytes(ByteBuffer sourceBytes) {
    if (sourceBytes == null) {
      throw new NullPointerException("sourceBytes");
    }
    if (sourceBytes.remaining() != BYTE_SIZE) {
      throw new IllegalArgumentException(
          "selected payload byte size mismatch: expected=" + BYTE_SIZE + " actual=" + sourceBytes.remaining());
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
    int lodLevel = read.getInt();
    int meshletStart = read.getInt();
    int meshletCount = read.getInt();
    int geometricErrorBits = read.getInt();
    return of(lodLevel, meshletStart, meshletCount, geometricErrorBits);
  }

  public int selectedLodLevel() {
    return selectedLodLevel;
  }

  public int meshletStart() {
    return meshletStart;
  }

  public int meshletCount() {
    return meshletCount;
  }

  public int geometricErrorBits() {
    return geometricErrorBits;
  }

  public float geometricError() {
    return Float.intBitsToFloat(geometricErrorBits);
  }

  public int byteSize() {
    return BYTE_SIZE;
  }

  /** Returns a read-only little-endian byte buffer view. */
  public ByteBuffer selectedBytes() {
    return selectedBytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
  }
}
