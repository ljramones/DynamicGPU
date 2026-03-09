package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.junit.jupiter.api.Test;

class GpuMeshletBoundsResourceTest {
  @Test
  void exposesResourceMetadataFromPayload() {
    ByteBuffer bytes = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putFloat(1f).putFloat(2f).putFloat(3f).putFloat(4f).putFloat(5f).putFloat(6f).flip();
    GpuMeshletBoundsPayload payload = GpuMeshletBoundsPayload.fromLittleEndianBytes(1, 0, 6, bytes);

    GpuMeshletBoundsResource resource = new GpuMeshletBoundsResource(new GpuBufferHandle(42L), payload);

    assertEquals(1, resource.meshletCount());
    assertEquals(24, resource.boundsByteSize());
    assertEquals(24, resource.boundsStrideBytes());
    assertEquals(42L, resource.bufferHandle().value());
  }
}

