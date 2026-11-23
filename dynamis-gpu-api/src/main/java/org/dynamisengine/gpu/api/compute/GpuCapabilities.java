package org.dynamisengine.gpu.api.compute;
import java.util.Set;
/** Conservative limits for enabled features, not promises about numerical reproducibility. */
public record GpuCapabilities(long maxBufferBytes, long maxStorageRange, int storageOffsetAlignment,
        int maxBufferBindings, int maxParameterBytes, Dimensions maxGroupSize, int maxGroupThreads,
        Dimensions maxGroupCount, Set<String> dataTypes, Set<String> shaderFeatures, boolean timestamps) {
    public GpuCapabilities { dataTypes = Set.copyOf(dataTypes); shaderFeatures = Set.copyOf(shaderFeatures); }
    public void validateGroups(Dimensions local, Dimensions groups) {
        if (local.x() > maxGroupSize.x() || local.y() > maxGroupSize.y() || local.z() > maxGroupSize.z()
                || local.volume() > maxGroupThreads || groups.x() > maxGroupCount.x()
                || groups.y() > maxGroupCount.y() || groups.z() > maxGroupCount.z())
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, "Dispatch exceeds device workgroup limits: local=" + local + " groups=" + groups);
    }
}
