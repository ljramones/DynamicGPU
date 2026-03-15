package org.dynamisengine.gpu.vulkan.compute;

import java.util.Objects;
import java.util.function.LongUnaryOperator;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBuildInputPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBuildInputResource;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferOps;
import org.lwjgl.vulkan.VkDevice;

/**
 * Resolves backend-usable device-address linkage for RT BLAS build-input payloads.
 */
public final class VulkanRayTracingBuildInputResolver {
  private final LongUnaryOperator deviceAddressResolver;

  public VulkanRayTracingBuildInputResolver(VkDevice device) {
    Objects.requireNonNull(device, "device");
    this.deviceAddressResolver = bufferHandle -> VulkanBufferOps.getBufferDeviceAddress(device, bufferHandle);
  }

  VulkanRayTracingBuildInputResolver(LongUnaryOperator deviceAddressResolver) {
    this.deviceAddressResolver = Objects.requireNonNull(deviceAddressResolver, "deviceAddressResolver");
  }

  public GpuRayTracingBuildInputResource resolve(GpuRayTracingBuildInputPayload payload) throws GpuException {
    Objects.requireNonNull(payload, "payload");
    long vertexAddress = deviceAddressResolver.applyAsLong(payload.vertexBufferHandle().value());
    long indexAddress = deviceAddressResolver.applyAsLong(payload.indexBufferHandle().value());
    if (vertexAddress <= 0L) {
      throw new GpuException(
          GpuErrorCode.BACKEND_INIT_FAILED,
          "resolved vertex buffer device address is invalid: " + vertexAddress,
          false);
    }
    if (indexAddress <= 0L) {
      throw new GpuException(
          GpuErrorCode.BACKEND_INIT_FAILED,
          "resolved index buffer device address is invalid: " + indexAddress,
          false);
    }
    return new GpuRayTracingBuildInputResource(payload, vertexAddress, indexAddress);
  }
}

