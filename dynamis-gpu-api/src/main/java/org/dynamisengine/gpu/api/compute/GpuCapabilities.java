package org.dynamisengine.gpu.api.compute;
import java.util.Set;
/** Conservative limits for enabled features, not promises about numerical reproducibility. */
public record GpuCapabilities(long maxBufferBytes, long maxStorageRange, int storageOffsetAlignment,
        int maxBufferBindings, int maxParameterBytes, Dimensions maxGroupSize, int maxGroupThreads,
        Dimensions maxGroupCount, Set<String> dataTypes, Set<String> shaderFeatures, boolean timestamps) {
    /**
     * Shader feature: {@code fma} is fused, computing {@code a*b + c} with one rounding. Vulkan does not require it
     * (its precision rules inherit fma from a multiply followed by an add), and some drivers, such as Mesa's
     * lavapipe, round twice. Kernels that must match a CPU {@code Math.fma} bit for bit need it. A device reports it
     * after measuring it ({@code GpuDevice.capabilities()}); adapters, which have no device yet, do not.
     */
    public static final String FUSED_FMA = "FUSED_FMA";

    public GpuCapabilities { dataTypes = Set.copyOf(dataTypes); shaderFeatures = Set.copyOf(shaderFeatures); }

    /** These capabilities with {@code features} added to {@link #shaderFeatures()}. */
    public GpuCapabilities withShaderFeatures(Set<String> features) {
        var all = new java.util.HashSet<>(shaderFeatures); all.addAll(features);
        return new GpuCapabilities(maxBufferBytes, maxStorageRange, storageOffsetAlignment, maxBufferBindings,
                maxParameterBytes, maxGroupSize, maxGroupThreads, maxGroupCount, dataTypes, all, timestamps);
    }
    public void validateGroups(Dimensions local, Dimensions groups) {
        if (local.x() > maxGroupSize.x() || local.y() > maxGroupSize.y() || local.z() > maxGroupSize.z()
                || local.volume() > maxGroupThreads || groups.x() > maxGroupCount.x()
                || groups.y() > maxGroupCount.y() || groups.z() > maxGroupCount.z())
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, "Dispatch exceeds device workgroup limits: local=" + local + " groups=" + groups);
    }
}
