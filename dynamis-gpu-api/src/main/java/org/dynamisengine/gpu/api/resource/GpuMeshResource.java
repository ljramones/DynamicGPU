package org.dynamisengine.gpu.api.resource;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.layout.IndexType;
import org.dynamisengine.gpu.api.layout.SubmeshRange;
import org.dynamisengine.gpu.api.layout.VertexLayout;

/**
 * GPU-owned mesh buffers and draw-range metadata.
 */
public final class GpuMeshResource implements AutoCloseable {
  private final GpuBuffer vertexBuffer;
  private final GpuBuffer indexBuffer;
  private final VertexLayout vertexLayout;
  private final IndexType indexType;
  private final List<SubmeshRange> submeshes;
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public GpuMeshResource(
      GpuBuffer vertexBuffer,
      GpuBuffer indexBuffer,
      VertexLayout vertexLayout,
      IndexType indexType,
      List<SubmeshRange> submeshes) {
    this.vertexBuffer = Objects.requireNonNull(vertexBuffer, "vertexBuffer");
    this.vertexLayout = Objects.requireNonNull(vertexLayout, "vertexLayout");
    Objects.requireNonNull(submeshes, "submeshes");
    if (submeshes.isEmpty()) {
      throw new IllegalArgumentException("submeshes must not be empty");
    }
    if ((indexBuffer == null) != (indexType == null)) {
      throw new IllegalArgumentException("indexBuffer and indexType must both be present or both be absent");
    }
    this.indexBuffer = indexBuffer;
    this.indexType = indexType;
    this.submeshes = List.copyOf(submeshes);
  }

  public GpuBuffer vertexBuffer() {
    return vertexBuffer;
  }

  public GpuBuffer indexBuffer() {
    return indexBuffer;
  }

  public VertexLayout vertexLayout() {
    return vertexLayout;
  }

  public IndexType indexType() {
    return indexType;
  }

  public List<SubmeshRange> submeshes() {
    return submeshes;
  }

  public boolean isIndexed() {
    return indexBuffer != null;
  }

  public boolean isClosed() {
    return closed.get();
  }

  @Override
  public void close() {
    if (!closed.compareAndSet(false, true)) {
      return;
    }
    RuntimeException firstFailure = null;
    try {
      vertexBuffer.close();
    } catch (RuntimeException e) {
      firstFailure = e;
    }
    if (indexBuffer != null) {
      try {
        indexBuffer.close();
      } catch (RuntimeException e) {
        if (firstFailure == null) {
          firstFailure = e;
        } else {
          firstFailure.addSuppressed(e);
        }
      }
    }
    if (firstFailure != null) {
      throw firstFailure;
    }
  }
}
