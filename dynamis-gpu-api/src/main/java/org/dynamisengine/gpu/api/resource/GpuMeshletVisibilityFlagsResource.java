package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * GPU-managed output resource for meshlet visibility flags.
 */
public final class GpuMeshletVisibilityFlagsResource implements AutoCloseable {
  private final GpuBuffer buffer;
  private final GpuMeshletVisibilityFlagsPayload payload;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuMeshletVisibilityFlagsResource(GpuBuffer buffer, GpuMeshletVisibilityFlagsPayload payload) {
    this.buffer = Objects.requireNonNull(buffer, "buffer");
    this.payload = Objects.requireNonNull(payload, "payload");
  }

  public GpuBuffer buffer() {
    return buffer;
  }

  public GpuBufferHandle bufferHandle() {
    return buffer.handle();
  }

  public GpuMeshletVisibilityFlagsPayload payload() {
    return payload;
  }

  public int meshletCount() {
    return payload.meshletCount();
  }

  public int flagsByteSize() {
    return payload.flagsByteSize();
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

