package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * GPU-managed resource for compact visible meshlet indices.
 */
public final class GpuVisibleMeshletListResource implements AutoCloseable {
  private final GpuBuffer buffer;
  private final GpuVisibleMeshletListPayload payload;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuVisibleMeshletListResource(GpuBuffer buffer, GpuVisibleMeshletListPayload payload) {
    this.buffer = Objects.requireNonNull(buffer, "buffer");
    this.payload = Objects.requireNonNull(payload, "payload");
  }

  public GpuBuffer buffer() {
    return buffer;
  }

  public GpuBufferHandle bufferHandle() {
    return buffer.handle();
  }

  public GpuVisibleMeshletListPayload payload() {
    return payload;
  }

  public int sourceMeshletCount() {
    return payload.sourceMeshletCount();
  }

  public int visibleMeshletCount() {
    return payload.visibleMeshletCount();
  }

  public int visibleIndicesByteSize() {
    return payload.visibleIndicesByteSize();
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

