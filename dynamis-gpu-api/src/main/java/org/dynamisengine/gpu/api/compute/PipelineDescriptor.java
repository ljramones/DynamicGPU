package org.dynamisengine.gpu.api.compute;
import java.util.Map;
import java.util.Set;
public record PipelineDescriptor(KernelBundle kernel, String entryPoint, BindingLayout layout,
        Dimensions localSize, Map<Integer, Integer> specialization, Set<String> requiredDataTypes,
        Set<String> requiredFeatures) {
    public PipelineDescriptor {
        java.util.Objects.requireNonNull(kernel); java.util.Objects.requireNonNull(layout); java.util.Objects.requireNonNull(localSize);
        if (entryPoint == null || entryPoint.isBlank()) throw new IllegalArgumentException("Entry point required");
        specialization = Map.copyOf(specialization);
        if (specialization.keySet().stream().anyMatch(id -> id < 0)) throw new IllegalArgumentException("Specialization IDs must be nonnegative");
        requiredDataTypes = Set.copyOf(requiredDataTypes); requiredFeatures = Set.copyOf(requiredFeatures);
    }
    public String cacheIdentity(Backend backend, String deviceId) {
        return backend + ":" + deviceId + ":" + kernel.name() + ":" + kernel.version() + ":"
            + kernel.executable(backend).identity() + ":" + entryPoint + ":" + layout + ":" + localSize
            + ":" + new java.util.TreeMap<>(specialization) + ":" + new java.util.TreeSet<>(requiredDataTypes)
            + ":" + new java.util.TreeSet<>(requiredFeatures);
    }
}
