package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * DynamisGPU-side resource representation for tessellation metadata input.
 */
public final class GpuTessellationResource implements AutoCloseable {
  private final GpuBuffer buffer;
  private final GpuTessellationPayload payload;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuTessellationResource(GpuBuffer buffer, GpuTessellationPayload payload) {
    this.buffer = Objects.requireNonNull(buffer, "buffer");
    this.payload = Objects.requireNonNull(payload, "payload");
  }

  public GpuBuffer buffer() {
    return buffer;
  }

  public GpuBufferHandle bufferHandle() {
    return buffer.handle();
  }

  public GpuTessellationPayload payload() {
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
