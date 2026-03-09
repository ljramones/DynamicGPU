package org.dynamisengine.gpu.api.gpu;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import org.dynamisengine.gpu.api.resource.GpuRayTracingTlasInstanceMetadata;

/**
 * Describes one TLAS build work unit over a set of BLAS instances.
 */
public final class RayTracingTlasWork {
  private final List<GpuRayTracingTlasInstanceMetadata> instances;

  public RayTracingTlasWork(List<GpuRayTracingTlasInstanceMetadata> instances) {
    Objects.requireNonNull(instances, "instances");
    if (instances.isEmpty()) {
      throw new IllegalArgumentException("instances must not be empty");
    }
    this.instances = Collections.unmodifiableList(new ArrayList<>(instances));
  }

  public static RayTracingTlasWork fromInstances(List<GpuRayTracingTlasInstanceMetadata> instances) {
    return new RayTracingTlasWork(instances);
  }

  public List<GpuRayTracingTlasInstanceMetadata> instances() {
    return instances;
  }
}

