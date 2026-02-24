package org.dynamisgpu.api;

import java.nio.ByteBuffer;

public record VulkanTexturePixelData(ByteBuffer data, int width, int height) {
}
