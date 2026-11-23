package org.dynamisengine.gpu.api.compute;
import java.util.Map;
/** Single-use encoder. All operations declare accesses; backends insert visibility barriers. */
public interface GpuCommandEncoder extends AutoCloseable {
    GpuCommandEncoder upload(GpuBuffer destination, byte[] data);
    GpuReadback download(GpuBuffer source);
    GpuCommandEncoder copy(GpuBuffer source, GpuBuffer destination);
    GpuCommandEncoder fill(GpuBuffer destination, int pattern);
    GpuCommandEncoder dispatch(GpuComputePipeline pipeline, Map<Integer, ? extends GpuBuffer> bindings,
            byte[] parameters, Dimensions groups);
    @Override void close();
}
