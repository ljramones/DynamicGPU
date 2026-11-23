package org.dynamisengine.gpu.api.compute;
/** Closing a handle leaves its slices and recorded/submitted uses alive independently. */
public interface GpuBuffer extends AutoCloseable {
    long bytes();
    long offset();
    GpuDevice device();
    BufferDescriptor descriptor();
    BufferSlice slice(long offset, long bytes);
    GpuMapping map(Access access);
    @Override void close();
}
