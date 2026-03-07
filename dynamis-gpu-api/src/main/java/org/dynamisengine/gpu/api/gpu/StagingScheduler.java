package org.dynamisengine.gpu.api.gpu;

public interface StagingScheduler {
  void markDirty(long bufferHandle, long offset, long size);

  void flush(VkCommandBuffer cmd);

  void reset();

  void destroy();
}
