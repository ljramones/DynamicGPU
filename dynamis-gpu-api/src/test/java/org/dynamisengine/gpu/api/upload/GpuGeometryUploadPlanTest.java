package org.dynamisengine.gpu.api.upload;

import java.nio.ByteBuffer;
import java.util.List;
import org.dynamisengine.gpu.api.layout.IndexType;
import org.dynamisengine.gpu.api.layout.SubmeshRange;
import org.dynamisengine.gpu.api.layout.VertexAttribute;
import org.dynamisengine.gpu.api.layout.VertexFormat;
import org.dynamisengine.gpu.api.layout.VertexLayout;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GpuGeometryUploadPlanTest {

  @Test
  void enforcesIndexDataAndTypePairing() {
    ByteBuffer vertex = ByteBuffer.allocate(12);
    vertex.put(new byte[12]);
    vertex.flip();
    VertexLayout layout = new VertexLayout(12, List.of(new VertexAttribute(0, 0, VertexFormat.FLOAT3)));

    assertThrows(
        IllegalArgumentException.class,
        () -> new GpuGeometryUploadPlan(vertex, null, layout, IndexType.UINT32, List.of(new SubmeshRange(0, 3, 0))));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new GpuGeometryUploadPlan(
                vertex,
                byteBuffer(new byte[6]),
                layout,
                null,
                List.of(new SubmeshRange(0, 3, 0))));
  }

  @Test
  void keepsReadOnlyCopiesOfPayloadBuffers() {
    ByteBuffer vertex = ByteBuffer.allocate(12).put(new byte[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    vertex.flip();
    ByteBuffer index = ByteBuffer.allocate(6).put(new byte[] {0, 1, 2, 3, 4, 5});
    index.flip();

    VertexLayout layout = new VertexLayout(12, List.of(new VertexAttribute(0, 0, VertexFormat.FLOAT3)));
    GpuGeometryUploadPlan plan =
        new GpuGeometryUploadPlan(vertex, index, layout, IndexType.UINT16, List.of(new SubmeshRange(0, 3, 0)));

    assertEquals(12, plan.vertexData().remaining());
    assertEquals(6, plan.indexData().remaining());
    assertThrows(java.nio.ReadOnlyBufferException.class, () -> plan.vertexData().put((byte) 0));
    assertThrows(java.nio.ReadOnlyBufferException.class, () -> plan.indexData().put((byte) 0));
  }

  private static ByteBuffer byteBuffer(byte[] bytes) {
    ByteBuffer buffer = ByteBuffer.allocate(bytes.length);
    buffer.put(bytes);
    buffer.flip();
    return buffer;
  }
}
