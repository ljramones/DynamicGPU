package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * DynamisGPU-side resource representation for meshlet bounds visibility input.
 *
 * <p>This model represents GPU-managed identity (buffer handle) plus the
 * authoritative payload metadata consumed from MeshForge contract output.
 */
public final class GpuMeshletBoundsResource implements AutoCloseable {
  private final GpuBuffer buffer;
  private final GpuMeshletBoundsPayload payload;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuMeshletBoundsResource(GpuBuffer buffer, GpuMeshletBoundsPayload payload) {
    this.buffer = Objects.requireNonNull(buffer, "buffer");
    this.payload = Objects.requireNonNull(payload, "payload");
  }

  public GpuBuffer buffer() {
    return buffer;
  }

  public GpuBufferHandle bufferHandle() {
    return buffer.handle();
  }

  public GpuMeshletBoundsPayload payload() {
    return payload;
  }

  public int meshletCount() {
    return payload.meshletCount();
  }

  public int boundsStrideBytes() {
    return payload.boundsStrideBytes();
  }

  public int boundsByteSize() {
    return payload.boundsByteSize();
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
