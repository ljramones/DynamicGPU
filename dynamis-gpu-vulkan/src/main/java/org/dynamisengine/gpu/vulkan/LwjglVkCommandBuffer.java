package org.dynamisengine.gpu.vulkan;

import java.util.Objects;
import org.dynamisengine.gpu.api.gpu.VkCommandBuffer;

/**
 * Lightweight adapter from LWJGL's command buffer type to the API abstraction.
 */
public final class LwjglVkCommandBuffer implements VkCommandBuffer {
  private final org.lwjgl.vulkan.VkCommandBuffer delegate;

  /**
   * Wraps a LWJGL command buffer for API-level consumption.
   *
   * @param delegate LWJGL command buffer instance
   */
  public LwjglVkCommandBuffer(org.lwjgl.vulkan.VkCommandBuffer delegate) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
  }

  /**
   * Returns the wrapped LWJGL command buffer.
   *
   * @return delegate command buffer
   */
  public org.lwjgl.vulkan.VkCommandBuffer delegate() {
    return delegate;
  }

  @Override
  public long handle() {
    return delegate.address();
  }
}
