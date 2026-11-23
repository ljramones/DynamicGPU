package org.dynamisengine.gpu.api.compute;
public interface GpuAdapter {
    Backend backend();
    String id();
    String name();
    GpuCapabilities capabilities();
    GpuDevice open();
}
