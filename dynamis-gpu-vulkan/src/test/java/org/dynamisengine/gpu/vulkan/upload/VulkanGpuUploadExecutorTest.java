package org.dynamisengine.gpu.vulkan.upload;

import java.nio.ByteBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

class VulkanGpuUploadExecutorTest {

  @Test
  void mapsBufferUsageToVulkanFlags() {
    assertEquals(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VulkanGpuUploadExecutor.toVkBufferUsage(GpuBufferUsage.VERTEX));
    assertEquals(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VulkanGpuUploadExecutor.toVkBufferUsage(GpuBufferUsage.INDEX));
    assertEquals(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanGpuUploadExecutor.toVkBufferUsage(GpuBufferUsage.TRANSFER_SRC));
    assertEquals(VK_BUFFER_USAGE_TRANSFER_DST_BIT, VulkanGpuUploadExecutor.toVkBufferUsage(GpuBufferUsage.TRANSFER_DST));
  }

  @Test
  void mapsMemoryLocationToVulkanFlags() {
    assertEquals(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VulkanGpuUploadExecutor.toVkMemoryProperties(GpuMemoryLocation.DEVICE_LOCAL));
    assertEquals(
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VulkanGpuUploadExecutor.toVkMemoryProperties(GpuMemoryLocation.HOST_VISIBLE));
  }

  @Test
  void createsDirectCopyForUpload() {
    ByteBuffer source = ByteBuffer.wrap(new byte[] {1, 2, 3, 4});
    ByteBuffer copied = VulkanGpuUploadExecutor.toDirectCopy(source);

    assertNotSame(source, copied);
    assertTrue(copied.isDirect());
    assertEquals(4, copied.remaining());
    assertEquals(1, copied.get(0));
    assertEquals(4, copied.get(3));
  }
}
