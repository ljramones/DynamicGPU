package org.dynamisengine.gpu.api.upload;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferHandle;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.layout.SubmeshRange;
import org.dynamisengine.gpu.api.layout.VertexAttribute;
import org.dynamisengine.gpu.api.layout.VertexFormat;
import org.dynamisengine.gpu.api.layout.VertexLayout;
import org.dynamisengine.gpu.api.resource.GpuMeshResource;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DefaultUploadManagerTest {

  @Test
  void enforcesBacklogBoundWhenInflightIsSaturated() throws Exception {
    BlockingUploadExecutor executor = new BlockingUploadExecutor();
    GpuGeometryUploadPlan plan = samplePlan();
    try (DefaultUploadManager manager = new DefaultUploadManager(executor, 1, 1)) {
      UploadTicket t1 = manager.submit(plan);
      executor.awaitFirstStart();

      UploadTicket t2 = manager.submit(plan);
      assertEquals(1, manager.inflightCount());
      assertEquals(1, manager.backlogSize());
      assertNotEquals(t1.id(), t2.id());

      assertThrows(IllegalStateException.class, () -> manager.submit(plan));

      executor.releaseAll();
      GpuMeshResource r1 = t1.await();
      GpuMeshResource r2 = t2.await();
      r1.close();
      r2.close();
      manager.drain();
      assertEquals(0, manager.inflightCount());
      assertEquals(0, manager.backlogSize());
    }
  }

  @Test
  void neverExceedsConfiguredInflightTarget() throws Exception {
    TrackingUploadExecutor executor = new TrackingUploadExecutor(35);
    GpuGeometryUploadPlan plan = samplePlan();
    List<UploadTicket> tickets = new ArrayList<>();

    try (DefaultUploadManager manager = new DefaultUploadManager(executor, 2, 16)) {
      for (int i = 0; i < 8; i++) {
        tickets.add(manager.submit(plan));
      }
      for (UploadTicket ticket : tickets) {
        GpuMeshResource resource = ticket.await();
        resource.close();
      }
      manager.drain();

      UploadTelemetry telemetry = manager.telemetry();
      assertEquals(8, telemetry.completedUploads());
      assertEquals(2, telemetry.maxInflightSubmissions());
      assertTrue(telemetry.throughputGbps() > 0.0);
      assertTrue(telemetry.averageTtfuMillis() >= 0.0);
    }

    assertTrue(executor.maxConcurrent() <= 2);
  }

  @Test
  void ticketReturnsSameResourceInstanceAfterCompletion() throws Exception {
    TrackingUploadExecutor executor = new TrackingUploadExecutor(0);
    GpuGeometryUploadPlan plan = samplePlan();
    try (DefaultUploadManager manager = new DefaultUploadManager(executor, 1, 4)) {
      UploadTicket ticket = manager.submit(plan);
      GpuMeshResource first = ticket.await();
      GpuMeshResource second = ticket.await();
      assertTrue(ticket.isComplete());
      assertSame(first, second);
      first.close();
    }
  }

  @Test
  void debugSnapshotContainsTriangleFields() throws Exception {
    TrackingUploadExecutor executor = new TrackingUploadExecutor(10);
    GpuGeometryUploadPlan plan = samplePlan();
    try (DefaultUploadManager manager = new DefaultUploadManager(executor, 2, 8)) {
      UploadTicket ticket = manager.submit(plan);
      GpuMeshResource resource = ticket.await();
      resource.close();
      manager.drain();

      String snapshot = manager.debugSnapshot();
      assertTrue(snapshot.contains("inflight="));
      assertTrue(snapshot.contains("backlog="));
      assertTrue(snapshot.contains("inflightBytes="));
      assertTrue(snapshot.contains("throughputGbps="));
      assertTrue(snapshot.contains("avgTtfuMs="));
      assertTrue(snapshot.contains("p95TtfuMs="));
      assertTrue(snapshot.contains("avgCompletionLatencyMs="));
    }
  }

  private static GpuGeometryUploadPlan samplePlan() {
    ByteBuffer vertex = ByteBuffer.allocate(12);
    vertex.put(new byte[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    vertex.flip();
    VertexLayout layout = new VertexLayout(12, List.of(new VertexAttribute(0, 0, VertexFormat.FLOAT3)));
    return new GpuGeometryUploadPlan(vertex, null, layout, null, List.of(new SubmeshRange(0, 3, 0)));
  }

  private static final class BlockingUploadExecutor implements GpuUploadExecutor {
    private final CountDownLatch firstStarted = new CountDownLatch(1);
    private final CountDownLatch releaseLatch = new CountDownLatch(1);
    private final AtomicLong handles = new AtomicLong(100);

    @Override
    public GpuMeshResource upload(GpuGeometryUploadPlan plan) throws GpuException {
      firstStarted.countDown();
      try {
        releaseLatch.await(2, TimeUnit.SECONDS);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
      return mesh(plan, handles.incrementAndGet());
    }

    void awaitFirstStart() throws InterruptedException {
      assertTrue(firstStarted.await(2, TimeUnit.SECONDS));
    }

    void releaseAll() {
      releaseLatch.countDown();
    }
  }

  private static final class TrackingUploadExecutor implements GpuUploadExecutor {
    private final AtomicInteger concurrent = new AtomicInteger();
    private final AtomicInteger maxConcurrent = new AtomicInteger();
    private final AtomicLong handles = new AtomicLong(1000);
    private final int sleepMillis;

    private TrackingUploadExecutor(int sleepMillis) {
      this.sleepMillis = sleepMillis;
    }

    @Override
    public GpuMeshResource upload(GpuGeometryUploadPlan plan) throws GpuException {
      int active = concurrent.incrementAndGet();
      maxConcurrent.accumulateAndGet(active, Math::max);
      try {
        if (sleepMillis > 0) {
          Thread.sleep(sleepMillis);
        }
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      } finally {
        concurrent.decrementAndGet();
      }
      return mesh(plan, handles.incrementAndGet());
    }

    int maxConcurrent() {
      return maxConcurrent.get();
    }
  }

  private static GpuMeshResource mesh(GpuGeometryUploadPlan plan, long handleBase) {
    FakeBuffer vertexBuffer =
        new FakeBuffer(
            new GpuBufferHandle(handleBase),
            plan.vertexData().remaining(),
            GpuBufferUsage.VERTEX,
            GpuMemoryLocation.DEVICE_LOCAL);
    return new GpuMeshResource(vertexBuffer, null, plan.vertexLayout(), null, plan.submeshes());
  }

  private static final class FakeBuffer implements GpuBuffer {
    private final GpuBufferHandle handle;
    private final long sizeBytes;
    private final GpuBufferUsage usage;
    private final GpuMemoryLocation memoryLocation;
    private boolean closed;

    private FakeBuffer(
        GpuBufferHandle handle, long sizeBytes, GpuBufferUsage usage, GpuMemoryLocation memoryLocation) {
      this.handle = handle;
      this.sizeBytes = sizeBytes;
      this.usage = usage;
      this.memoryLocation = memoryLocation;
    }

    @Override
    public GpuBufferHandle handle() {
      return handle;
    }

    @Override
    public long sizeBytes() {
      return sizeBytes;
    }

    @Override
    public GpuBufferUsage usage() {
      return usage;
    }

    @Override
    public GpuMemoryLocation memoryLocation() {
      return memoryLocation;
    }

    @Override
    public void close() {
      closed = true;
    }

    @SuppressWarnings("unused")
    boolean closed() {
      return closed;
    }
  }
}
