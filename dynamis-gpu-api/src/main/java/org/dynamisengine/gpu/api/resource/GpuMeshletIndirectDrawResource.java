package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * GPU-managed resource for generated indexed-indirect draw commands.
 */
public final class GpuMeshletIndirectDrawResource implements AutoCloseable {
  private final GpuBuffer buffer;
  private final GpuMeshletIndirectDrawPayload payload;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuMeshletIndirectDrawResource(GpuBuffer buffer, GpuMeshletIndirectDrawPayload payload) {
    this.buffer = Objects.requireNonNull(buffer, "buffer");
    this.payload = Objects.requireNonNull(payload, "payload");
  }

  public GpuBuffer buffer() {
    return buffer;
  }

  public GpuBufferHandle bufferHandle() {
    return buffer.handle();
  }

  public GpuMeshletIndirectDrawPayload payload() {
    return payload;
  }

  public int commandCount() {
    return payload.commandCount();
  }

  public int commandByteSize() {
    return payload.commandByteSize();
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

