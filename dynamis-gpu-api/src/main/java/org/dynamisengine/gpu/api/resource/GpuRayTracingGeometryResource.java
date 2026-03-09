package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * DynamisGPU-side resource representation for RT geometry metadata input.
 */
public final class GpuRayTracingGeometryResource implements AutoCloseable {
  private final GpuBuffer buffer;
  private final GpuRayTracingGeometryPayload payload;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuRayTracingGeometryResource(GpuBuffer buffer, GpuRayTracingGeometryPayload payload) {
    this.buffer = Objects.requireNonNull(buffer, "buffer");
    this.payload = Objects.requireNonNull(payload, "payload");
  }

  public GpuBuffer buffer() {
    return buffer;
  }

  public GpuBufferHandle bufferHandle() {
    return buffer.handle();
  }

  public GpuRayTracingGeometryPayload payload() {
    return payload;
  }

  public int regionCount() {
    return payload.regionCount();
  }

  public int regionsStrideBytes() {
    return payload.regionsStrideBytes();
  }

  public int regionsByteSize() {
    return payload.regionsByteSize();
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

