package org.dynamisengine.gpu.api.resource;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * DynamisGPU-side resource representation of BLAS build-input-ready RT geometry.
 *
 * <p>This seam includes validated build input payload plus resolved backend buffer device
 * addresses used by Vulkan BLAS geometry setup.
 */
public final class GpuRayTracingBuildInputResource implements AutoCloseable {
  private final GpuRayTracingBuildInputPayload payload;
  private final long vertexBufferDeviceAddress;
  private final long indexBufferDeviceAddress;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuRayTracingBuildInputResource(
      GpuRayTracingBuildInputPayload payload,
      long vertexBufferDeviceAddress,
      long indexBufferDeviceAddress) {
    this.payload = Objects.requireNonNull(payload, "payload");
    if (vertexBufferDeviceAddress <= 0L) {
      throw new IllegalArgumentException("vertexBufferDeviceAddress must be > 0");
    }
    if (indexBufferDeviceAddress <= 0L) {
      throw new IllegalArgumentException("indexBufferDeviceAddress must be > 0");
    }
    this.vertexBufferDeviceAddress = vertexBufferDeviceAddress;
    this.indexBufferDeviceAddress = indexBufferDeviceAddress;
  }

  public GpuRayTracingBuildInputPayload payload() {
    return payload;
  }

  public long vertexBufferDeviceAddress() {
    return vertexBufferDeviceAddress;
  }

  public long indexBufferDeviceAddress() {
    return indexBufferDeviceAddress;
  }

  public int regionCount() {
    return payload.regionCount();
  }

  public int vertexStrideBytes() {
    return payload.vertexStrideBytes();
  }

  public int maxVertexIndex() {
    return payload.maxVertexIndex();
  }

  public long vertexDataOffsetBytes() {
    return payload.vertexDataOffsetBytes();
  }

  public long indexDataOffsetBytes() {
    return payload.indexDataOffsetBytes();
  }

  public boolean isClosed() {
    return closed.get();
  }

  @Override
  public void close() {
    closed.set(true);
  }
}

