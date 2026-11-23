package org.dynamisengine.gpu.api.compute;
/** Device and child operations are confined to the creating thread. Close drains pending work. */
public interface GpuDevice extends AutoCloseable {
    GpuAdapter adapter();
    default GpuCapabilities capabilities() { return adapter().capabilities(); }
    GpuBuffer allocate(BufferDescriptor descriptor);
    GpuComputePipeline pipeline(PipelineDescriptor descriptor);
    GpuCommandEncoder encoder();
    GpuQueue queue();
    @Override void close();
}
