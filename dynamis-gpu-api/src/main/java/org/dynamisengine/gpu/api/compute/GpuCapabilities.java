package org.dynamisengine.gpu.api.compute;
import java.util.Set;
/**
 * Conservative limits for enabled features, not promises about numerical reproducibility.
 * <p>
 * <b>Data types</b> ({@link #dataTypes()}): a listed type may be used for arithmetic and in storage buffers. The
 * backend enables what the device supports when it opens it. {@link #FP32}, {@link #INT32} and {@link #BOOL} are
 * always present; the others are optional. BF16 is not a data type here: kernels store it as 16-bit words and compute
 * in FP32.
 * <p>
 * <b>Shader features</b> ({@link #shaderFeatures()}): subgroup (Metal SIMD-group) operations, buffer atomics beyond
 * 32-bit integers, and measured numeric properties such as {@link #FUSED_FMA}.
 * <p>
 * A pipeline lists what it needs in {@link PipelineDescriptor#requiredDataTypes()} and
 * {@link PipelineDescriptor#requiredFeatures()}, and creating it fails with {@link ComputeException.Code#UNSUPPORTED}
 * where any is missing. On Vulkan, a SPIR-V module that declares a capability the device did not enable is rejected
 * the same way, declared or not.
 *
 * @param timestamps whether {@link GpuSubmission#gpuTime()} measures submissions on this device (1.3.0)
 * @param minSubgroupSize the smallest subgroup (SIMD-group) size a kernel can run with; 0 without subgroups
 * @param maxSubgroupSize the largest; equal to {@code minSubgroupSize} where the size is fixed
 */
public record GpuCapabilities(long maxBufferBytes, long maxStorageRange, int storageOffsetAlignment,
        int maxBufferBindings, int maxParameterBytes, Dimensions maxGroupSize, int maxGroupThreads,
        Dimensions maxGroupCount, Set<String> dataTypes, Set<String> shaderFeatures, boolean timestamps,
        int minSubgroupSize, int maxSubgroupSize) {
    /** Data type: 32-bit float, always present. */
    public static final String FP32 = "FP32";
    /** Data type: 32-bit integers, always present. */
    public static final String INT32 = "INT32";
    /** Data type: booleans, always present. */
    public static final String BOOL = "BOOL";
    /** Data type: IEEE binary16 arithmetic and storage (Vulkan shaderFloat16 and 16-bit storage; Metal half). */
    public static final String FP16 = "FP16";
    /** Data type: 16-bit integer arithmetic and storage. */
    public static final String INT16 = "INT16";
    /** Data type: 8-bit integer arithmetic and storage. */
    public static final String INT8 = "INT8";
    /** Data type: 64-bit integer arithmetic and storage. */
    public static final String INT64 = "INT64";
    /** Data type: 64-bit float arithmetic and storage. Metal has none. */
    public static final String FP64 = "FP64";

    /** Shader feature: subgroup size, lane index, elect and barrier (SPIR-V GroupNonUniform). */
    public static final String SUBGROUP_BASIC = "SUBGROUP_BASIC";
    /** Shader feature: subgroup all, any and all-equal votes. */
    public static final String SUBGROUP_VOTE = "SUBGROUP_VOTE";
    /** Shader feature: subgroup reductions and prefix scans (sum, product, min, max, and, or, xor). */
    public static final String SUBGROUP_ARITHMETIC = "SUBGROUP_ARITHMETIC";
    /** Shader feature: subgroup ballots, bit counts and broadcasts. */
    public static final String SUBGROUP_BALLOT = "SUBGROUP_BALLOT";
    /** Shader feature: reading another lane's value by lane index (shuffle and shuffle-xor). */
    public static final String SUBGROUP_SHUFFLE = "SUBGROUP_SHUFFLE";
    /**
     * Shader feature: subgroup matrix multiply-accumulate of FP16 inputs into FP32 accumulators in hardware: Metal
     * SIMD-group matrices ({@code simdgroup_matrix}, 8 x 8) on Apple7 and later GPUs. Kernels using it are written for
     * one backend; Vulkan cooperative matrices are not reported yet.
     *
     * @since 1.3.0
     */
    public static final String SUBGROUP_MATRIX_F16 = "SUBGROUP_MATRIX_F16";
    /** Shader feature: atomic add on 32-bit floats in storage buffers, the scatter-add of gradient kernels. */
    public static final String ATOMIC_FLOAT32_ADD = "ATOMIC_FLOAT32_ADD";
    /** Shader feature: atomic add, min, max, exchange and compare-exchange on 64-bit integers in storage buffers. */
    public static final String ATOMIC_INT64 = "ATOMIC_INT64";
    /**
     * Shader feature: {@code fma} is fused, computing {@code a*b + c} with one rounding. Vulkan does not require it
     * (its precision rules inherit fma from a multiply followed by an add), and some drivers, such as Mesa's
     * lavapipe, round twice. Kernels that must match a CPU {@code Math.fma} bit for bit need it. A device reports it
     * after measuring it ({@code GpuDevice.capabilities()}); adapters, which have no device yet, do not.
     */
    public static final String FUSED_FMA = "FUSED_FMA";

    public GpuCapabilities {
        dataTypes = Set.copyOf(dataTypes); shaderFeatures = Set.copyOf(shaderFeatures);
        if (minSubgroupSize < 0 || maxSubgroupSize < minSubgroupSize)
            throw new IllegalArgumentException("subgroup sizes " + minSubgroupSize + ".." + maxSubgroupSize);
    }

    /** Capabilities without subgroups (the 1.1 form). */
    public GpuCapabilities(long maxBufferBytes, long maxStorageRange, int storageOffsetAlignment,
            int maxBufferBindings, int maxParameterBytes, Dimensions maxGroupSize, int maxGroupThreads,
            Dimensions maxGroupCount, Set<String> dataTypes, Set<String> shaderFeatures, boolean timestamps) {
        this(maxBufferBytes, maxStorageRange, storageOffsetAlignment, maxBufferBindings, maxParameterBytes, maxGroupSize,
                maxGroupThreads, maxGroupCount, dataTypes, shaderFeatures, timestamps, 0, 0);
    }

    /** These capabilities with {@code features} added to {@link #shaderFeatures()}. */
    public GpuCapabilities withShaderFeatures(Set<String> features) {
        var all = new java.util.HashSet<>(shaderFeatures); all.addAll(features);
        return new GpuCapabilities(maxBufferBytes, maxStorageRange, storageOffsetAlignment, maxBufferBindings,
                maxParameterBytes, maxGroupSize, maxGroupThreads, maxGroupCount, dataTypes, all, timestamps,
                minSubgroupSize, maxSubgroupSize);
    }
    public void validateGroups(Dimensions local, Dimensions groups) {
        if (local.x() > maxGroupSize.x() || local.y() > maxGroupSize.y() || local.z() > maxGroupSize.z()
                || local.volume() > maxGroupThreads || groups.x() > maxGroupCount.x()
                || groups.y() > maxGroupCount.y() || groups.z() > maxGroupCount.z())
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, "Dispatch exceeds device workgroup limits: local=" + local + " groups=" + groups);
    }
}
