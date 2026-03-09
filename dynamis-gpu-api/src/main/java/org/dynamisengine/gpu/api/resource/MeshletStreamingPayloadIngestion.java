package org.dynamisengine.gpu.api.resource;

import java.nio.ByteBuffer;
import java.util.Objects;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * Ingestion seam for finalized upstream meshlet-streaming payload contracts.
 */
public final class MeshletStreamingPayloadIngestion {
  private MeshletStreamingPayloadIngestion() {}

  public static GpuMeshletStreamingPayload ingest(
      int streamUnitCount,
      int streamUnitsOffsetInts,
      int streamUnitsStrideInts,
      int streamUnitsIntCount,
      int expectedStreamUnitsIntCount,
      ByteBuffer streamUnitsBytes) {
    if (streamUnitsIntCount != expectedStreamUnitsIntCount) {
      throw new IllegalArgumentException(
          "upstream stream units int count mismatch: count="
              + streamUnitsIntCount
              + " expected="
              + expectedStreamUnitsIntCount);
    }

    GpuMeshletStreamingPayload payload =
        GpuMeshletStreamingPayload.fromLittleEndianBytes(
            streamUnitCount, streamUnitsOffsetInts, streamUnitsStrideInts, streamUnitsBytes);
    if (payload.streamUnitsIntCount() != streamUnitsIntCount) {
      throw new IllegalArgumentException(
          "payload int count mismatch after ingestion: expected="
              + streamUnitsIntCount
              + " actual="
              + payload.streamUnitsIntCount());
    }
    return payload;
  }

  public static GpuMeshletStreamingResource toResource(GpuBuffer buffer, GpuMeshletStreamingPayload payload) {
    Objects.requireNonNull(buffer, "buffer");
    Objects.requireNonNull(payload, "payload");
    if (buffer.sizeBytes() != payload.streamUnitsByteSize()) {
      throw new IllegalArgumentException(
          "buffer size does not match payload bytes: buffer="
              + buffer.sizeBytes()
              + " payload="
              + payload.streamUnitsByteSize());
    }
    return new GpuMeshletStreamingResource(buffer, payload);
  }

  public static GpuMeshletStreamingResource toResource(
      GpuBufferHandle bufferHandle, GpuMeshletStreamingPayload payload) {
    Objects.requireNonNull(bufferHandle, "bufferHandle");
    Objects.requireNonNull(payload, "payload");
    return new GpuMeshletStreamingResource(
        new HandleOnlyGpuBuffer(bufferHandle, payload.streamUnitsByteSize()), payload);
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
