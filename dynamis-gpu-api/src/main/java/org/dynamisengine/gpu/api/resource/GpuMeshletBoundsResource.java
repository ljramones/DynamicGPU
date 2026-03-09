package org.dynamisengine.gpu.api.resource;

import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;

/**
 * DynamisGPU-side resource representation for meshlet bounds visibility input.
 *
 * <p>This model represents GPU-managed identity (buffer handle) plus the
 * authoritative payload metadata consumed from MeshForge contract output.
 */
public record GpuMeshletBoundsResource(GpuBufferHandle bufferHandle, GpuMeshletBoundsPayload payload) {
  public GpuMeshletBoundsResource {
    if (bufferHandle == null) {
      throw new NullPointerException("bufferHandle");
    }
    if (payload == null) {
      throw new NullPointerException("payload");
    }
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
}

