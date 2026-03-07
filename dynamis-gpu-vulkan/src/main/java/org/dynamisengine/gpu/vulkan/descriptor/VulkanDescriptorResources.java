package org.dynamisengine.gpu.vulkan.descriptor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.dynamisengine.gpu.api.gpu.DescriptorWriter;

public final class VulkanDescriptorResources implements DescriptorWriter {
  private final List<DescriptorWrite> writes = new ArrayList<>();

  @Override
  public void writeStorageBuffer(
      long descriptorSet, int binding, int arrayElement, long bufferHandle, long offset, long range) {
    writes.add(
        new DescriptorWrite(
            WriteType.STORAGE_BUFFER, descriptorSet, binding, arrayElement, bufferHandle, offset, range, 0L, 0L));
  }

  @Override
  public void writeUniformBuffer(
      long descriptorSet, int binding, int arrayElement, long bufferHandle, long offset, long range) {
    writes.add(
        new DescriptorWrite(
            WriteType.UNIFORM_BUFFER, descriptorSet, binding, arrayElement, bufferHandle, offset, range, 0L, 0L));
  }

  @Override
  public void writeSampledImage(
      long descriptorSet, int binding, int arrayElement, long imageView, long sampler) {
    writes.add(
        new DescriptorWrite(
            WriteType.SAMPLED_IMAGE, descriptorSet, binding, arrayElement, 0L, 0L, 0L, imageView, sampler));
  }

  public List<DescriptorWrite> writes() {
    return Collections.unmodifiableList(writes);
  }

  public enum WriteType {
    STORAGE_BUFFER,
    UNIFORM_BUFFER,
    SAMPLED_IMAGE
  }

  public record DescriptorWrite(
      WriteType type,
      long descriptorSet,
      int binding,
      int arrayElement,
      long bufferHandle,
      long offset,
      long range,
      long imageView,
      long sampler) {}
}
