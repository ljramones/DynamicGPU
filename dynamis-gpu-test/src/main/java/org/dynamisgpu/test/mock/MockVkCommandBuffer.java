package org.dynamisgpu.test.mock;

import org.dynamisgpu.api.gpu.VkCommandBuffer;

public record MockVkCommandBuffer(long handle) implements VkCommandBuffer {}
