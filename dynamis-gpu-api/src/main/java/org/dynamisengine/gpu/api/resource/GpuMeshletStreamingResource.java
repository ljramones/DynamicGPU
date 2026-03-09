package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * DynamisGPU-side resource representation for meshlet streaming metadata input.
 */
public final class GpuMeshletStreamingResource implements AutoCloseable {
  private final GpuBuffer buffer;
  private final GpuMeshletStreamingPayload payload;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuMeshletStreamingResource(GpuBuffer buffer, GpuMeshletStreamingPayload payload) {
    this.buffer = Objects.requireNonNull(buffer, "buffer");
    this.payload = Objects.requireNonNull(payload, "payload");
  }

  public GpuBuffer buffer() {
    return buffer;
  }

  public GpuBufferHandle bufferHandle() {
    return buffer.handle();
  }

  public GpuMeshletStreamingPayload payload() {
    return payload;
  }

  public int streamUnitCount() {
    return payload.streamUnitCount();
  }

  public int streamUnitsStrideBytes() {
    return payload.streamUnitsStrideBytes();
  }

  public int streamUnitsByteSize() {
    return payload.streamUnitsByteSize();
  }

  public boolean isClosed() {
    return closed.get();
  }

  @Override
  public void close() {
    if (!closed.compareAndSet(false, true)) {
      return;
    }
    buffer.close();
  }
}
