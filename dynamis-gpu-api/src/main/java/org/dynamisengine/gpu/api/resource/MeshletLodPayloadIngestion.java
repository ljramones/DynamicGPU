package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.util.Objects;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * Ingestion seam for finalized upstream meshlet-LOD payload contracts.
 */
public final class MeshletLodPayloadIngestion {
  private MeshletLodPayloadIngestion() {}

  public static GpuMeshletLodPayload ingest(
      int levelCount,
      int levelsOffsetInts,
      int levelsStrideInts,
      int levelsIntCount,
      int expectedLevelsIntCount,
      ByteBuffer levelsBytes) {
    if (levelsIntCount != expectedLevelsIntCount) {
      throw new IllegalArgumentException(
          "upstream levels int count mismatch: count="
              + levelsIntCount
              + " expected="
              + expectedLevelsIntCount);
    }

    GpuMeshletLodPayload payload =
        GpuMeshletLodPayload.fromLittleEndianBytes(
            levelCount, levelsOffsetInts, levelsStrideInts, levelsBytes);
    if (payload.levelsIntCount() != levelsIntCount) {
      throw new IllegalArgumentException(
          "payload int count mismatch after ingestion: expected="
              + levelsIntCount
              + " actual="
              + payload.levelsIntCount());
    }
    return payload;
  }

  public static GpuMeshletLodResource toResource(GpuBuffer buffer, GpuMeshletLodPayload payload) {
    Objects.requireNonNull(buffer, "buffer");
    Objects.requireNonNull(payload, "payload");
    if (buffer.sizeBytes() != payload.levelsByteSize()) {
      throw new IllegalArgumentException(
          "buffer size does not match payload bytes: buffer="
              + buffer.sizeBytes()
              + " payload="
              + payload.levelsByteSize());
    }
    return new GpuMeshletLodResource(buffer, payload);
  }

  public static GpuMeshletLodResource toResource(
      GpuBufferHandle bufferHandle, GpuMeshletLodPayload payload) {
    Objects.requireNonNull(bufferHandle, "bufferHandle");
    Objects.requireNonNull(payload, "payload");
    return new GpuMeshletLodResource(
        new HandleOnlyGpuBuffer(bufferHandle, payload.levelsByteSize()), payload);
  }

  private static final class HandleOnlyGpuBuffer implements GpuBuffer {
    private final GpuBufferHandle handle;
    private final long sizeBytes;

    private HandleOnlyGpuBuffer(GpuBufferHandle handle, long sizeBytes) {
      this.handle = handle;
      this.sizeBytes = sizeBytes;
    }

    @Override
    public GpuBufferHandle handle() {
      return handle;
    }

    @Override
    public long sizeBytes() {
      return sizeBytes;
    }

    @Override
    public org.dynamisengine.gpu.api.buffer.GpuBufferUsage usage() {
      return org.dynamisengine.gpu.api.buffer.GpuBufferUsage.TRANSFER_DST;
    }

    @Override
    public org.dynamisengine.gpu.api.buffer.GpuMemoryLocation memoryLocation() {
      return org.dynamisengine.gpu.api.buffer.GpuMemoryLocation.DEVICE_LOCAL;
    }

    @Override
    public void close() {
      // No-op for handle-only adapter.
    }
  }
}
