package org.dynamisengine.gpu.api.compute;
import java.util.List;
/** Discovery is explicit: constructing an unused provider must not load native libraries. */
public interface GpuProvider {
    Backend backend();
    List<GpuAdapter> adapters();
}
