package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * DynamisGPU-side resource representation for meshlet LOD metadata input.
 */
public final class GpuMeshletLodResource implements AutoCloseable {
  private final GpuBuffer buffer;
  private final GpuMeshletLodPayload payload;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuMeshletLodResource(GpuBuffer buffer, GpuMeshletLodPayload payload) {
    this.buffer = Objects.requireNonNull(buffer, "buffer");
    this.payload = Objects.requireNonNull(payload, "payload");
  }

  public GpuBuffer buffer() {
    return buffer;
  }

  public GpuBufferHandle bufferHandle() {
    return buffer.handle();
  }

  public GpuMeshletLodPayload payload() {
    return payload;
  }

  public int levelCount() {
    return payload.levelCount();
  }

  public int levelsStrideBytes() {
    return payload.levelsStrideBytes();
  }

  public int levelsByteSize() {
    return payload.levelsByteSize();
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
