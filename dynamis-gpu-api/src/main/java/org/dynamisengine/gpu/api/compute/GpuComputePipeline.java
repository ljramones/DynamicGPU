package org.dynamisengine.gpu.api.compute;
public interface GpuComputePipeline extends AutoCloseable {
    GpuDevice device();
    PipelineDescriptor descriptor();
    @Override void close();
}
