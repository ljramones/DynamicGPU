package org.dynamisengine.gpu.api.gpu;

import java.util.Objects;
import org.dynamisengine.gpu.api.resource.GpuMeshletIndirectDrawResource;
import org.dynamisengine.gpu.api.resource.GpuMeshletVisibilityFlagsResource;
import org.dynamisengine.gpu.api.resource.GpuVisibleMeshletListResource;

/**
 * End-to-end validation result for GPU-driven cluster visibility.
 */
public final class GpuDrivenClusterVisibilityValidationResult implements AutoCloseable {
  private final GpuMeshletVisibilityFlagsResource visibilityFlags;
  private final GpuVisibleMeshletListResource visibleMeshlets;
  private final GpuMeshletIndirectDrawResource indirectDraws;
  private final long visibilityNanos;
  private final long compactionNanos;
  private final long indirectGenerationNanos;
  private final long totalNanos;

  public GpuDrivenClusterVisibilityValidationResult(
      GpuMeshletVisibilityFlagsResource visibilityFlags,
      GpuVisibleMeshletListResource visibleMeshlets,
      GpuMeshletIndirectDrawResource indirectDraws,
      long visibilityNanos,
      long compactionNanos,
      long indirectGenerationNanos,
      long totalNanos) {
    this.visibilityFlags = Objects.requireNonNull(visibilityFlags, "visibilityFlags");
    this.visibleMeshlets = Objects.requireNonNull(visibleMeshlets, "visibleMeshlets");
    this.indirectDraws = Objects.requireNonNull(indirectDraws, "indirectDraws");
    if (visibilityNanos < 0 || compactionNanos < 0 || indirectGenerationNanos < 0 || totalNanos < 0) {
      throw new IllegalArgumentException("timings must be >= 0");
    }
    this.visibilityNanos = visibilityNanos;
    this.compactionNanos = compactionNanos;
    this.indirectGenerationNanos = indirectGenerationNanos;
    this.totalNanos = totalNanos;
  }

  public GpuMeshletVisibilityFlagsResource visibilityFlags() {
    return visibilityFlags;
  }

  public GpuVisibleMeshletListResource visibleMeshlets() {
    return visibleMeshlets;
  }

  public GpuMeshletIndirectDrawResource indirectDraws() {
    return indirectDraws;
  }

  public long visibilityNanos() {
    return visibilityNanos;
  }

  public long compactionNanos() {
    return compactionNanos;
  }

  public long indirectGenerationNanos() {
    return indirectGenerationNanos;
  }

  public long totalNanos() {
    return totalNanos;
  }

  @Override
  public void close() {
    RuntimeException firstFailure = null;
    try {
      indirectDraws.close();
    } catch (RuntimeException e) {
      firstFailure = e;
    }
    try {
      visibleMeshlets.close();
    } catch (RuntimeException e) {
      if (firstFailure == null) {
        firstFailure = e;
      } else {
        firstFailure.addSuppressed(e);
      }
    }
    try {
      visibilityFlags.close();
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
}

