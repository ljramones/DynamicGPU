package org.dynamisengine.gpu.api.compute;
import java.nio.ByteBuffer;
/** The returned view may only be used before close, on the owner thread, without GPU uses. */
public interface GpuMapping extends AutoCloseable {
    ByteBuffer bytes();
    @Override void close();
}
