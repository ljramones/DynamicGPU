package org.dynamisengine.gpu.api.gpu;

import java.util.Objects;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshletIndirectDrawResource;
import org.dynamisengine.gpu.api.resource.GpuMeshletVisibilityFlagsResource;
import org.dynamisengine.gpu.api.resource.GpuVisibleMeshletListResource;

/**
 * End-to-end validation harness for the GPU-driven cluster visibility chain.
 *
 * <p>This composes existing capabilities without re-implementing stage logic.
 */
public final class GpuDrivenClusterVisibilityValidationHarness implements AutoCloseable {
  private final MeshletVisibilityCapability visibilityCapability;
  private final MeshletVisibilityCompactionCapability compactionCapability;
  private final MeshletIndirectDrawGenerationCapability indirectCapability;
  private boolean closed;

  public GpuDrivenClusterVisibilityValidationHarness(
      MeshletVisibilityCapability visibilityCapability,
      MeshletVisibilityCompactionCapability compactionCapability,
      MeshletIndirectDrawGenerationCapability indirectCapability) {
    this.visibilityCapability = Objects.requireNonNull(visibilityCapability, "visibilityCapability");
    this.compactionCapability = Objects.requireNonNull(compactionCapability, "compactionCapability");
    this.indirectCapability = Objects.requireNonNull(indirectCapability, "indirectCapability");
  }

  public GpuDrivenClusterVisibilityValidationResult run(
      GpuDrivenClusterVisibilityValidationWork work) throws GpuException {
    ensureOpen();
    Objects.requireNonNull(work, "work");

    long totalStart = System.nanoTime();
    GpuMeshletVisibilityFlagsResource flags = null;
    GpuVisibleMeshletListResource visible = null;
    GpuMeshletIndirectDrawResource indirect = null;
    long visibilityNanos;
    long compactionNanos;
    long indirectNanos;
    try {
      long stageStart = System.nanoTime();
      flags =
          visibilityCapability.execute(
              new MeshletVisibilityWork(work.boundsResource(), work.meshletCount(), work.frustum()));
      visibilityNanos = System.nanoTime() - stageStart;

      stageStart = System.nanoTime();
      visible =
          compactionCapability.execute(
              new MeshletVisibilityCompactionWork(flags, work.meshletCount()));
      compactionNanos = System.nanoTime() - stageStart;

      stageStart = System.nanoTime();
      indirect =
          indirectCapability.execute(
              new MeshletIndirectDrawGenerationWork(
                  visible, work.drawMetadata(), visible.visibleMeshletCount()));
      indirectNanos = System.nanoTime() - stageStart;
    } catch (RuntimeException | GpuException e) {
      closeQuietly(indirect);
      closeQuietly(visible);
      closeQuietly(flags);
      throw e;
    }
    long totalNanos = System.nanoTime() - totalStart;
    return new GpuDrivenClusterVisibilityValidationResult(
        flags, visible, indirect, visibilityNanos, compactionNanos, indirectNanos, totalNanos);
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    RuntimeException firstFailure = null;
    try {
      indirectCapability.close();
    } catch (RuntimeException e) {
      firstFailure = e;
    }
    try {
      compactionCapability.close();
    } catch (RuntimeException e) {
      if (firstFailure == null) {
        firstFailure = e;
      } else {
        firstFailure.addSuppressed(e);
      }
    }
    try {
      visibilityCapability.close();
    } catch (RuntimeException e) {
      if (firstFailure == null) {
        firstFailure = e;
      } else {
        firstFailure.addSuppressed(e);
      }
    }
    if (firstFailure != null) {
      throw firstFailure;
    }
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("Harness has been closed");
    }
  }

  private static void closeQuietly(AutoCloseable closeable) {
    if (closeable == null) {
      return;
    }
    try {
      closeable.close();
    } catch (Exception ignored) {
      // Preserve primary failure from run.
    }
  }
}

