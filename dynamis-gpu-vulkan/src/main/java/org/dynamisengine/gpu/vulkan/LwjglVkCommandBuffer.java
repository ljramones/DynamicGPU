package org.dynamisengine.gpu.vulkan;

import java.util.Objects;
import org.dynamisengine.gpu.api.gpu.VkCommandBuffer;

/**
 * Lightweight adapter from LWJGL's command buffer type to the API abstraction.
 */
public final class LwjglVkCommandBuffer implements VkCommandBuffer {
  private final org.lwjgl.vulkan.VkCommandBuffer delegate;

  public LwjglVkCommandBuffer(org.lwjgl.vulkan.VkCommandBuffer delegate) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
  }

  public org.lwjgl.vulkan.VkCommandBuffer delegate() {
    return delegate;
  }

  @Override
  public long handle() {
    return delegate.address();
  }
}
