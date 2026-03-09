package org.dynamisengine.gpu.api.resource;

import java.util.Arrays;
import java.util.Objects;

/**
 * TLAS instance metadata describing one BLAS instance reference and transform.
 */
public final class GpuRayTracingTlasInstanceMetadata {
  public static final int TRANSFORM_FLOATS = 12;

  private final GpuRayTracingBlasResource blasResource;
  private final float[] transform3x4RowMajor;
  private final int instanceCustomIndex;
  private final int mask;
  private final int shaderBindingTableRecordOffset;
  private final int flags;

  private GpuRayTracingTlasInstanceMetadata(
      GpuRayTracingBlasResource blasResource,
      float[] transform3x4RowMajor,
      int instanceCustomIndex,
      int mask,
      int shaderBindingTableRecordOffset,
      int flags) {
    this.blasResource = blasResource;
    this.transform3x4RowMajor = transform3x4RowMajor;
    this.instanceCustomIndex = instanceCustomIndex;
    this.mask = mask;
    this.shaderBindingTableRecordOffset = shaderBindingTableRecordOffset;
    this.flags = flags;
  }

  public static GpuRayTracingTlasInstanceMetadata of(
      GpuRayTracingBlasResource blasResource,
      float[] transform3x4RowMajor,
      int instanceCustomIndex,
      int mask,
      int shaderBindingTableRecordOffset,
      int flags) {
    Objects.requireNonNull(blasResource, "blasResource");
    Objects.requireNonNull(transform3x4RowMajor, "transform3x4RowMajor");
    if (!blasResource.hasAccelerationStructure()) {
      throw new IllegalArgumentException("blasResource must contain a built acceleration structure handle");
    }
    if (blasResource.isClosed()) {
      throw new IllegalArgumentException("blasResource is already closed");
    }
    if (transform3x4RowMajor.length != TRANSFORM_FLOATS) {
      throw new IllegalArgumentException(
          "transform3x4RowMajor length mismatch: expected="
              + TRANSFORM_FLOATS
              + " actual="
              + transform3x4RowMajor.length);
    }
    if (instanceCustomIndex < 0) {
      throw new IllegalArgumentException("instanceCustomIndex must be >= 0");
    }
    if (mask < 0 || mask > 0xFF) {
      throw new IllegalArgumentException("mask must be in [0, 255]");
    }
    if (shaderBindingTableRecordOffset < 0) {
      throw new IllegalArgumentException("shaderBindingTableRecordOffset must be >= 0");
    }
    if (flags < 0) {
      throw new IllegalArgumentException("flags must be >= 0");
    }
    return new GpuRayTracingTlasInstanceMetadata(
        blasResource,
        Arrays.copyOf(transform3x4RowMajor, transform3x4RowMajor.length),
        instanceCustomIndex,
        mask,
        shaderBindingTableRecordOffset,
        flags);
  }

  public GpuRayTracingBlasResource blasResource() {
    return blasResource;
  }

  public float[] transform3x4RowMajor() {
    return Arrays.copyOf(transform3x4RowMajor, transform3x4RowMajor.length);
  }

  public int instanceCustomIndex() {
    return instanceCustomIndex;
  }

  public int mask() {
    return mask;
  }

  public int shaderBindingTableRecordOffset() {
    return shaderBindingTableRecordOffset;
  }

  public int flags() {
    return flags;
  }
}

