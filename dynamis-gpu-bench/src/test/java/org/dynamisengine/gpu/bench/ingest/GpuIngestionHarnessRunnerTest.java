package org.dynamisengine.gpu.bench.ingest;

import java.nio.ByteBuffer;
import java.util.List;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.layout.IndexType;
import org.dynamisengine.gpu.api.layout.SubmeshRange;
import org.dynamisengine.gpu.api.layout.VertexAttribute;
import org.dynamisengine.gpu.api.layout.VertexFormat;
import org.dynamisengine.gpu.api.layout.VertexLayout;
import org.dynamisengine.gpu.api.resource.GpuMeshResource;
import org.dynamisengine.gpu.api.upload.GpuGeometryUploadPlan;
import org.dynamisengine.gpu.api.upload.GpuUploadExecutor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GpuIngestionHarnessRunnerTest {
  @Test
  void runsUploadWithTimingAndValidationSummary() throws GpuException {
    GpuIngestionHarnessRunner runner =
        new GpuIngestionHarnessRunner(new MirroringUploadExecutor(), new DefaultGeometryUploadValidation());

    IngestionRunReport report = runner.run("fixture-a", this::indexedPlan);
    String line = IngestionReportFormatter.toLine(report);

    assertEquals("fixture-a", report.fixtureName());
    assertEquals(2, report.validation().vertexCount());
    assertEquals(3, report.validation().indexCount());
    assertEquals(12, report.validation().strideBytes());
    assertEquals(IndexType.UINT16, report.validation().indexType());
    assertEquals(1, report.validation().submeshCount());
    assertTrue(report.timing().planIntakeNanos() >= 0L);
    assertTrue(report.timing().uploadExecutionNanos() >= 0L);
    assertTrue(report.timing().totalExecutorNanos() >= 0L);
    assertTrue(line.contains("fixture=fixture-a"));
    assertTrue(line.contains("vertexCount=2"));
  }

  @Test
  void rejectsInvalidSubmeshAgainstIndexCount() {
    GpuIngestionHarnessRunner runner =
        new GpuIngestionHarnessRunner(new MirroringUploadExecutor(), new DefaultGeometryUploadValidation());
    assertThrows(
        IllegalArgumentException.class,
        () -> runner.run("invalid", this::invalidIndexedPlan));
  }

  private GpuGeometryUploadPlan indexedPlan() {
    VertexLayout layout = new VertexLayout(12, List.of(new VertexAttribute(0, 0, VertexFormat.FLOAT3)));
    ByteBuffer vertices = ByteBuffer.allocate(24);
    vertices.put(new byte[24]);
    vertices.flip();
    ByteBuffer indices = ByteBuffer.allocate(6);
    indices.put(new byte[] {0, 0, 1, 0, 0, 0});
    indices.flip();
    return new GpuGeometryUploadPlan(
        vertices, indices, layout, IndexType.UINT16, List.of(new SubmeshRange(0, 3, 0)));
  }

  private GpuGeometryUploadPlan invalidIndexedPlan() {
    VertexLayout layout = new VertexLayout(12, List.of(new VertexAttribute(0, 0, VertexFormat.FLOAT3)));
    ByteBuffer vertices = ByteBuffer.allocate(24);
    vertices.put(new byte[24]);
    vertices.flip();
    ByteBuffer indices = ByteBuffer.allocate(6);
    indices.put(new byte[] {0, 0, 1, 0, 0, 0});
    indices.flip();
    return new GpuGeometryUploadPlan(
        vertices, indices, layout, IndexType.UINT16, List.of(new SubmeshRange(2, 3, 0)));
  }

  private static final class MirroringUploadExecutor implements GpuUploadExecutor {
    @Override
    public GpuMeshResource upload(GpuGeometryUploadPlan plan) {
      TestBuffer vertex = new TestBuffer(11L, plan.vertexData().remaining(), GpuBufferUsage.VERTEX);
      TestBuffer index =
          plan.indexData() == null
              ? null
              : new TestBuffer(12L, plan.indexData().remaining(), GpuBufferUsage.INDEX);
      return new GpuMeshResource(vertex, index, plan.vertexLayout(), plan.indexType(), plan.submeshes());
    }
  }

  private static final class TestBuffer implements GpuBuffer {
    private final GpuBufferHandle handle;
    private final long size;
    private final GpuBufferUsage usage;

    private TestBuffer(long handle, long size, GpuBufferUsage usage) {
      this.handle = new GpuBufferHandle(handle);
      this.size = size;
      this.usage = usage;
    }

    @Override
    public GpuBufferHandle handle() {
      return handle;
    }

    @Override
    public long sizeBytes() {
      return size;
    }

    @Override
    public GpuBufferUsage usage() {
      return usage;
    }

    @Override
    public GpuMemoryLocation memoryLocation() {
      return GpuMemoryLocation.DEVICE_LOCAL;
    }

    @Override
    public void close() {}
  }
}
