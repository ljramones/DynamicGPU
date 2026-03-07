package org.dynamisengine.gpu.api.gpu;

/**
 * Backend-agnostic descriptor update writer abstraction.
 */
public interface DescriptorWriter {
  /**
   * Writes a storage-buffer descriptor binding.
   *
   * @param descriptorSet destination descriptor set handle
   * @param binding descriptor binding index
   * @param arrayElement descriptor array element
   * @param bufferHandle storage buffer handle
   * @param offset byte offset
   * @param range byte range
   */
  void writeStorageBuffer(
      long descriptorSet, int binding, int arrayElement, long bufferHandle, long offset, long range);

  /**
   * Writes a uniform-buffer descriptor binding.
   *
   * @param descriptorSet destination descriptor set handle
   * @param binding descriptor binding index
   * @param arrayElement descriptor array element
   * @param bufferHandle uniform buffer handle
   * @param offset byte offset
   * @param range byte range
   */
  void writeUniformBuffer(
      long descriptorSet, int binding, int arrayElement, long bufferHandle, long offset, long range);

  /**
   * Writes a sampled-image descriptor binding.
   *
   * @param descriptorSet destination descriptor set handle
   * @param binding descriptor binding index
   * @param arrayElement descriptor array element
   * @param imageView image view handle
   * @param sampler sampler handle
   */
  void writeSampledImage(long descriptorSet, int binding, int arrayElement, long imageView, long sampler);
}
