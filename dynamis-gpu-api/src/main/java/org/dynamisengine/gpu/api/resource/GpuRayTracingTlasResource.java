package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * DynamisGPU-side resource representation of a built TLAS.
 */
public final class GpuRayTracingTlasResource implements AutoCloseable {
  private final GpuBuffer buffer;
  private final GpuRayTracingTlasPayload payload;
  private final long accelerationStructureHandle;
  private final Runnable closeAction;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuRayTracingTlasResource(
      GpuBuffer buffer,
      GpuRayTracingTlasPayload payload,
      long accelerationStructureHandle,
      Runnable closeAction) {
    this.buffer = Objects.requireNonNull(buffer, "buffer");
    this.payload = Objects.requireNonNull(payload, "payload");
    if (accelerationStructureHandle <= 0L) {
      throw new IllegalArgumentException("accelerationStructureHandle must be > 0");
    }
    this.accelerationStructureHandle = accelerationStructureHandle;
    this.closeAction = closeAction != null ? closeAction : buffer::close;
  }

  public GpuBuffer buffer() {
    return buffer;
  }

  public GpuBufferHandle bufferHandle() {
    return buffer.handle();
  }

  public GpuRayTracingTlasPayload payload() {
    return payload;
  }

  public int instanceCount() {
    return payload.instanceCount();
  }

  public int instanceByteSize() {
    return payload.instanceByteSize();
  }

  public long accelerationStructureHandle() {
    return accelerationStructureHandle;
  }

  public boolean isClosed() {
    return closed.get();
  }

  @Override
  public void close() {
    if (!closed.compareAndSet(false, true)) {
      return;
    }
    closeAction.run();
  }
}

