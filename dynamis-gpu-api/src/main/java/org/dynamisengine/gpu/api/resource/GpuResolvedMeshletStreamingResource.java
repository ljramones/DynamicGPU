package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * DynamisGPU-side resource representation for one resolved meshlet streaming output.
 */
public final class GpuResolvedMeshletStreamingResource implements AutoCloseable {
  private final GpuBuffer buffer;
  private final GpuResolvedMeshletStreamingPayload payload;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuResolvedMeshletStreamingResource(GpuBuffer buffer, GpuResolvedMeshletStreamingPayload payload) {
    this.buffer = Objects.requireNonNull(buffer, "buffer");
    this.payload = Objects.requireNonNull(payload, "payload");
  }

  public GpuBuffer buffer() {
    return buffer;
  }

  public GpuBufferHandle bufferHandle() {
    return buffer.handle();
  }

  public GpuResolvedMeshletStreamingPayload payload() {
    return payload;
  }

  public int streamUnitId() {
    return payload.streamUnitId();
  }

  public int meshletStart() {
    return payload.meshletStart();
  }

  public int meshletCount() {
    return payload.meshletCount();
  }

  public int payloadByteOffset() {
    return payload.payloadByteOffset();
  }

  public int payloadByteSize() {
    return payload.payloadByteSize();
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

