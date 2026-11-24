package org.dynamisengine.gpu.api.compute;

/**
 * What a compiled pipeline allows, for choosing tile shapes and group sizes.
 *
 * @param maxThreads the most threads a group of this pipeline may have. On Metal it reflects the kernel's register
 *        use ({@code maxTotalThreadsPerThreadgroup}); on Vulkan it is the device limit.
 * @param executionWidth the threads that execute together: Metal {@code threadExecutionWidth}, the Vulkan subgroup
 *        size; 1 where unknown
 * @param groupSharedBytes the group-shared memory the kernel declares, -1 where unknown. Metal reports
 *        {@code staticThreadgroupMemoryLength}; on Vulkan it is the sum of the SPIR-V module's Workgroup variables,
 *        at the default size of any specialization-constant-sized array.
 * @since 1.7.0
 */
public record PipelineLimits(int maxThreads, int executionWidth, long groupSharedBytes) {
    public PipelineLimits {
        if (maxThreads <= 0 || executionWidth <= 0 || groupSharedBytes < -1) {
            throw new IllegalArgumentException("pipeline limits " + maxThreads + ", " + executionWidth + ", " + groupSharedBytes);
        }
    }
}
