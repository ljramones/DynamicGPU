package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.util.Objects;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * Ingestion seam for finalized upstream meshlet-bounds payload contracts.
 *
 * <p>This utility consumes upstream metadata + bytes as authoritative and creates
 * DynamisGPU payload/resource models without redefining layout.
 */
public final class MeshletBoundsPayloadIngestion {
  private MeshletBoundsPayloadIngestion() {}

  public static GpuMeshletBoundsPayload ingest(
      int meshletCount,
      int boundsOffsetFloats,
      int boundsStrideFloats,
      int boundsFloatCount,
      int expectedBoundsFloatCount,
      ByteBuffer boundsBytes) {
    if (boundsFloatCount != expectedBoundsFloatCount) {
      throw new IllegalArgumentException(
          "upstream bounds float count mismatch: count="
              + boundsFloatCount
              + " expected="
              + expectedBoundsFloatCount);
    }
    GpuMeshletBoundsPayload payload =
        GpuMeshletBoundsPayload.fromLittleEndianBytes(
            meshletCount, boundsOffsetFloats, boundsStrideFloats, boundsBytes);
    if (payload.boundsFloatCount() != boundsFloatCount) {
      throw new IllegalArgumentException(
          "payload float count mismatch after ingestion: expected="
              + boundsFloatCount
              + " actual="
              + payload.boundsFloatCount());
    }
    return payload;
  }

  public static GpuMeshletBoundsResource toResource(
      GpuBuffer buffer, GpuMeshletBoundsPayload payload) {
    Objects.requireNonNull(buffer, "buffer");
    Objects.requireNonNull(payload, "payload");
    if (buffer.sizeBytes() != payload.boundsByteSize()) {
      throw new IllegalArgumentException(
          "buffer size does not match payload bytes: buffer="
              + buffer.sizeBytes()
              + " payload="
              + payload.boundsByteSize());
    }
    return new GpuMeshletBoundsResource(buffer, payload);
  }

  public static GpuMeshletBoundsResource toResource(
      GpuBufferHandle bufferHandle, GpuMeshletBoundsPayload payload) {
    Objects.requireNonNull(bufferHandle, "bufferHandle");
    Objects.requireNonNull(payload, "payload");
    return new GpuMeshletBoundsResource(new HandleOnlyGpuBuffer(bufferHandle, payload.boundsByteSize()), payload);
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
