package org.dynamisengine.gpu.api;

import java.nio.ByteBuffer;

/**
 * Immutable raw texture payload for Vulkan image upload flows.
 *
 * @param data pixel bytes in the format expected by the upload path
 * @param width texture width in pixels
 * @param height texture height in pixels
 */
public record VulkanTexturePixelData(ByteBuffer data, int width, int height) {
}
