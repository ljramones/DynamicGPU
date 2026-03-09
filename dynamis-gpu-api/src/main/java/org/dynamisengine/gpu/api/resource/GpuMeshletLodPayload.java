package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Immutable GPU upload payload contract for meshlet LOD levels.
 *
 * <p>Layout v1:
 * <ul>
 *   <li>per-level element width: 4 int32 values</li>
 *   <li>int order: lodLevel, meshletStart, meshletCount, geometricErrorBits</li>
 *   <li>little-endian packed byte payload</li>
 * </ul>
 *
 * <p>{@code geometricErrorBits} stores {@code Float.floatToRawIntBits(geometricError)}
 * produced by MeshForge preparation.
 */
public final class GpuMeshletLodPayload {
  public static final int LEVEL_COMPONENTS = 4;

  private final int levelCount;
  private final int levelsOffsetInts;
  private final int levelsStrideInts;
  private final ByteBuffer levelsBytes;

  private GpuMeshletLodPayload(
      int levelCount, int levelsOffsetInts, int levelsStrideInts, ByteBuffer levelsBytes) {
    this.levelCount = levelCount;
    this.levelsOffsetInts = levelsOffsetInts;
    this.levelsStrideInts = levelsStrideInts;
    this.levelsBytes = levelsBytes;
  }

  public static GpuMeshletLodPayload fromLittleEndianBytes(
      int levelCount, int levelsOffsetInts, int levelsStrideInts, ByteBuffer sourceBytes) {
    if (levelCount < 0) {
      throw new IllegalArgumentException("levelCount must be >= 0");
    }
    if (levelsOffsetInts < 0) {
      throw new IllegalArgumentException("levelsOffsetInts must be >= 0");
    }
    if (levelsStrideInts < LEVEL_COMPONENTS) {
      throw new IllegalArgumentException("levelsStrideInts must be >= " + LEVEL_COMPONENTS);
    }
    if (sourceBytes == null) {
      throw new NullPointerException("sourceBytes");
    }
    if ((sourceBytes.remaining() % Integer.BYTES) != 0) {
      throw new IllegalArgumentException("sourceBytes remaining must be int-aligned");
    }

    int intCount = sourceBytes.remaining() / Integer.BYTES;
    int expectedIntCount = expectedLevelsIntCount(levelCount, levelsOffsetInts, levelsStrideInts);
    if (intCount != expectedIntCount) {
      throw new IllegalArgumentException(
          "levels payload int count mismatch: expected="
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
    return new GpuMeshletLodPayload(
        levelCount, levelsOffsetInts, levelsStrideInts, copy.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN));
  }

  public int levelCount() {
    return levelCount;
  }

  public int levelsOffsetInts() {
    return levelsOffsetInts;
  }

  public int levelsStrideInts() {
    return levelsStrideInts;
  }

  public int levelsStrideBytes() {
    return levelsStrideInts * Integer.BYTES;
  }

  public int levelsIntCount() {
    return levelsBytes.remaining() / Integer.BYTES;
  }

  public int expectedLevelsIntCount() {
    return expectedLevelsIntCount(levelCount, levelsOffsetInts, levelsStrideInts);
  }

  public int levelsByteSize() {
    return levelsBytes.remaining();
  }

  /** Returns a read-only little-endian byte buffer view. */
  public ByteBuffer levelsBytes() {
    return levelsBytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static int expectedLevelsIntCount(int levelCount, int levelsOffsetInts, int levelsStrideInts) {
    return levelCount == 0 ? 0 : levelsOffsetInts + (levelCount * levelsStrideInts);
  }
}
