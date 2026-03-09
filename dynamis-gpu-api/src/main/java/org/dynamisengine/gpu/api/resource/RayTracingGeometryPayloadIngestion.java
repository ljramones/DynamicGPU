package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.util.Objects;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * Ingestion seam for finalized upstream RT geometry payload contracts.
 */
public final class RayTracingGeometryPayloadIngestion {
  private RayTracingGeometryPayloadIngestion() {}

  public static GpuRayTracingGeometryPayload ingest(
      int regionCount,
      int regionsOffsetInts,
      int regionsStrideInts,
      int regionsIntCount,
      int expectedRegionsIntCount,
      ByteBuffer regionsBytes) {
    if (regionsIntCount != expectedRegionsIntCount) {
      throw new IllegalArgumentException(
          "upstream RT regions int count mismatch: count="
              + regionsIntCount
              + " expected="
              + expectedRegionsIntCount);
    }

    GpuRayTracingGeometryPayload payload =
        GpuRayTracingGeometryPayload.fromLittleEndianBytes(
            regionCount, regionsOffsetInts, regionsStrideInts, regionsBytes);
    if (payload.regionsIntCount() != regionsIntCount) {
      throw new IllegalArgumentException(
          "payload int count mismatch after ingestion: expected="
              + regionsIntCount
              + " actual="
              + payload.regionsIntCount());
    }
    return payload;
  }

  public static GpuRayTracingGeometryResource toResource(
      GpuBuffer buffer, GpuRayTracingGeometryPayload payload) {
    Objects.requireNonNull(buffer, "buffer");
    Objects.requireNonNull(payload, "payload");
    if (buffer.sizeBytes() != payload.regionsByteSize()) {
      throw new IllegalArgumentException(
          "buffer size does not match payload bytes: buffer="
              + buffer.sizeBytes()
              + " payload="
              + payload.regionsByteSize());
    }
    return new GpuRayTracingGeometryResource(buffer, payload);
  }

  public static GpuRayTracingGeometryResource toResource(
      GpuBufferHandle bufferHandle, GpuRayTracingGeometryPayload payload) {
    Objects.requireNonNull(bufferHandle, "bufferHandle");
    Objects.requireNonNull(payload, "payload");
    return new GpuRayTracingGeometryResource(
        new HandleOnlyGpuBuffer(bufferHandle, payload.regionsByteSize()), payload);
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

