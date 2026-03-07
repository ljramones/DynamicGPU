package org.dynamisengine.gpu.test.mock;

import org.dynamisengine.gpu.api.gpu.VkCommandBuffer;

public record MockVkCommandBuffer(long handle) implements VkCommandBuffer {}
