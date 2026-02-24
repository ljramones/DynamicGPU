package org.dynamisgpu.api.gpu;

public interface DescriptorWriter {
  void writeStorageBuffer(
      long descriptorSet, int binding, int arrayElement, long bufferHandle, long offset, long range);

  void writeUniformBuffer(
      long descriptorSet, int binding, int arrayElement, long bufferHandle, long offset, long range);

  void writeSampledImage(long descriptorSet, int binding, int arrayElement, long imageView, long sampler);
}
