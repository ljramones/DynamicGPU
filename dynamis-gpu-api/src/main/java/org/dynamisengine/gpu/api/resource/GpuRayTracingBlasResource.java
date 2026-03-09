package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * DynamisGPU-side resource representation of a BLAS build-preparation output seam.
 */
public final class GpuRayTracingBlasResource implements AutoCloseable {
  private final GpuBuffer buffer;
  private final GpuRayTracingBlasPayload payload;
  private final GpuRayTracingGeometryResource sourceGeometryResource;
  private final long accelerationStructureHandle;
  private final Runnable closeAction;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuRayTracingBlasResource(
      GpuBuffer buffer,
      GpuRayTracingBlasPayload payload,
      GpuRayTracingGeometryResource sourceGeometryResource) {
    this(buffer, payload, sourceGeometryResource, 0L, null);
  }

  public GpuRayTracingBlasResource(
      GpuBuffer buffer,
      GpuRayTracingBlasPayload payload,
      GpuRayTracingGeometryResource sourceGeometryResource,
      long accelerationStructureHandle,
      Runnable closeAction) {
    this.buffer = Objects.requireNonNull(buffer, "buffer");
    this.payload = Objects.requireNonNull(payload, "payload");
    this.sourceGeometryResource = Objects.requireNonNull(sourceGeometryResource, "sourceGeometryResource");
    if (accelerationStructureHandle < 0L) {
      throw new IllegalArgumentException("accelerationStructureHandle must be >= 0");
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

  public GpuRayTracingBlasPayload payload() {
    return payload;
  }

  public GpuRayTracingGeometryResource sourceGeometryResource() {
    return sourceGeometryResource;
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

  public long accelerationStructureHandle() {
    return accelerationStructureHandle;
  }

  public boolean hasAccelerationStructure() {
    return accelerationStructureHandle > 0L;
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
