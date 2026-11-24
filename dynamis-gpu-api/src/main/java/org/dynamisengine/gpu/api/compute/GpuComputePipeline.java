package org.dynamisengine.gpu.api.compute;
public interface GpuComputePipeline extends AutoCloseable {
    GpuDevice device();
    PipelineDescriptor descriptor();
    /** What the compiled pipeline allows. @since 1.7.0 */
    default PipelineLimits limits() {
        var c = device().capabilities();
        return new PipelineLimits(c.maxGroupThreads(), Math.max(1, c.maxSubgroupSize()), -1);
    }
    @Override void close();
}
