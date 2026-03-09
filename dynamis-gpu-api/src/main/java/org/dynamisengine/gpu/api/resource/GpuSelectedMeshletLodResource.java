package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * DynamisGPU-side resource representation of one selected meshlet LOD output.
 */
public final class GpuSelectedMeshletLodResource implements AutoCloseable {
  private final GpuBuffer buffer;
  private final GpuSelectedMeshletLodPayload payload;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuSelectedMeshletLodResource(GpuBuffer buffer, GpuSelectedMeshletLodPayload payload) {
    this.buffer = Objects.requireNonNull(buffer, "buffer");
    this.payload = Objects.requireNonNull(payload, "payload");
  }

  public GpuBuffer buffer() {
    return buffer;
  }

  public GpuBufferHandle bufferHandle() {
    return buffer.handle();
  }

  public GpuSelectedMeshletLodPayload payload() {
    return payload;
  }

  public int selectedLodLevel() {
    return payload.selectedLodLevel();
  }

  public int meshletStart() {
    return payload.meshletStart();
  }

  public int meshletCount() {
    return payload.meshletCount();
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
