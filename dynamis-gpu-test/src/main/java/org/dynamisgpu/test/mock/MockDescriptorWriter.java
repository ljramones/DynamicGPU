package org.dynamisgpu.test.mock;

import java.util.ArrayList;
import java.util.List;
import org.dynamisgpu.api.gpu.DescriptorWriter;

public final class MockDescriptorWriter implements DescriptorWriter {
  private final List<DescriptorCall> calls = new ArrayList<>();

  @Override
  public void writeStorageBuffer(
      long descriptorSet, int binding, int arrayElement, long bufferHandle, long offset, long range) {
    calls.add(
        new DescriptorCall(
            "storage", descriptorSet, binding, arrayElement, bufferHandle, offset, range, 0L, 0L));
  }

  @Override
  public void writeUniformBuffer(
      long descriptorSet, int binding, int arrayElement, long bufferHandle, long offset, long range) {
    calls.add(
        new DescriptorCall(
            "uniform", descriptorSet, binding, arrayElement, bufferHandle, offset, range, 0L, 0L));
  }

  @Override
  public void writeSampledImage(
      long descriptorSet, int binding, int arrayElement, long imageView, long sampler) {
    calls.add(
        new DescriptorCall(
            "image", descriptorSet, binding, arrayElement, 0L, 0L, 0L, imageView, sampler));
  }

  public List<DescriptorCall> calls() {
    return List.copyOf(calls);
  }

  public record DescriptorCall(
      String type,
      long descriptorSet,
      int binding,
      int arrayElement,
      long bufferHandle,
      long offset,
      long range,
      long imageView,
      long sampler) {}
}
