package org.dynamisgpu.api.gpu;

/**
 * API-level command buffer handle abstraction to keep the API module free of native bindings.
 */
public interface VkCommandBuffer {
  long handle();
}
