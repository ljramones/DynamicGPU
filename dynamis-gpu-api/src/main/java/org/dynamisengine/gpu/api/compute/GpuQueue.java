package org.dynamisengine.gpu.api.compute;
/** One ordered compute queue. A submission failure poisons later work and is reported on drain. */
public interface GpuQueue { GpuSubmission submit(GpuCommandEncoder encoder); }
