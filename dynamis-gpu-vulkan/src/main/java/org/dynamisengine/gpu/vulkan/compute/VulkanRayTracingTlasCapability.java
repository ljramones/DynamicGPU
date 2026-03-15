package org.dynamisengine.gpu.vulkan.compute;

import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_GEOMETRY_TYPE_INSTANCES_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.vkCmdBuildAccelerationStructuresKHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.vkCreateAccelerationStructureKHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.vkDestroyAccelerationStructureKHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.vkGetAccelerationStructureBuildSizesKHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.vkGetAccelerationStructureDeviceAddressKHR;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_SUBMIT_INFO;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkAllocateCommandBuffers;
import static org.lwjgl.vulkan.VK10.vkBeginCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkDestroyBuffer;
import static org.lwjgl.vulkan.VK10.vkEndCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkFreeCommandBuffers;
import static org.lwjgl.vulkan.VK10.vkFreeMemory;
import static org.lwjgl.vulkan.VK10.vkGetDeviceProcAddr;
import static org.lwjgl.vulkan.VK10.vkQueueSubmit;
import static org.lwjgl.vulkan.VK10.vkQueueWaitIdle;
import static org.lwjgl.vulkan.VK12.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
import static org.lwjgl.vulkan.VK12.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.gpu.RayTracingTlasCapability;
import org.dynamisengine.gpu.api.gpu.RayTracingTlasWork;
import org.dynamisengine.gpu.api.resource.GpuRayTracingTlasInstanceMetadata;
import org.dynamisengine.gpu.api.resource.GpuRayTracingTlasPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingTlasResource;
import org.dynamisengine.gpu.vulkan.buffer.VulkanGpuBuffer;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferOps;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkAccelerationStructureBuildGeometryInfoKHR;
import org.lwjgl.vulkan.VkAccelerationStructureBuildRangeInfoKHR;
import org.lwjgl.vulkan.VkAccelerationStructureBuildSizesInfoKHR;
import org.lwjgl.vulkan.VkAccelerationStructureCreateInfoKHR;
import org.lwjgl.vulkan.VkAccelerationStructureDeviceAddressInfoKHR;
import org.lwjgl.vulkan.VkAccelerationStructureGeometryInstancesDataKHR;
import org.lwjgl.vulkan.VkAccelerationStructureGeometryKHR;
import org.lwjgl.vulkan.VkAccelerationStructureInstanceKHR;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkDeviceOrHostAddressConstKHR;
import org.lwjgl.vulkan.VkDeviceOrHostAddressKHR;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkSubmitInfo;
import org.lwjgl.vulkan.VkTransformMatrixKHR;

/**
 * Vulkan-side TLAS capability with real acceleration-structure build execution when supported.
 */
public final class VulkanRayTracingTlasCapability implements RayTracingTlasCapability {
  @FunctionalInterface
  interface TlasExecutor {
    GpuRayTracingTlasResource execute(RayTracingTlasWork work) throws GpuException;
  }

  private final TlasExecutor tlasExecutor;
  private boolean closed;

  public VulkanRayTracingTlasCapability(
      VkDevice device, VkPhysicalDevice physicalDevice, long commandPool, VkQueue graphicsQueue) {
    this.tlasExecutor =
        new VulkanTlasExecutor(
            Objects.requireNonNull(device, "device"),
            Objects.requireNonNull(physicalDevice, "physicalDevice"),
            commandPool,
            Objects.requireNonNull(graphicsQueue, "graphicsQueue"));
  }

  VulkanRayTracingTlasCapability(TlasExecutor tlasExecutor) {
    this.tlasExecutor = Objects.requireNonNull(tlasExecutor, "tlasExecutor");
  }

  @Override
  public GpuRayTracingTlasResource execute(RayTracingTlasWork work) throws GpuException {
    ensureOpen();
    Objects.requireNonNull(work, "work");
    return tlasExecutor.execute(work);
  }

  @Override
  public void close() {
    closed = true;
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("Capability has been closed");
    }
  }

  private static final class VulkanTlasExecutor implements TlasExecutor {
    private final VkDevice device;
    private final VkPhysicalDevice physicalDevice;
    private final long commandPool;
    private final VkQueue graphicsQueue;
    private final Function<String, Boolean> procAvailable;

    private VulkanTlasExecutor(
        VkDevice device, VkPhysicalDevice physicalDevice, long commandPool, VkQueue graphicsQueue) {
      this(device, physicalDevice, commandPool, graphicsQueue, name -> vkGetDeviceProcAddr(device, name) != 0L);
    }

    private VulkanTlasExecutor(
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        long commandPool,
        VkQueue graphicsQueue,
        Function<String, Boolean> procAvailable) {
      this.device = device;
      this.physicalDevice = physicalDevice;
      if (commandPool == VK_NULL_HANDLE) {
        throw new IllegalArgumentException("commandPool must not be VK_NULL_HANDLE");
      }
      this.commandPool = commandPool;
      this.graphicsQueue = graphicsQueue;
      this.procAvailable = procAvailable;
    }

    @Override
    public GpuRayTracingTlasResource execute(RayTracingTlasWork work) throws GpuException {
      ensureBackendSupport();
      List<GpuRayTracingTlasInstanceMetadata> instances = work.instances();
      try (MemoryStack stack = MemoryStack.stackPush()) {
        ByteBuffer instanceBytes = buildInstanceByteBuffer(instances, stack);
        VulkanBufferAlloc instanceAlloc =
            VulkanBufferOps.createDeviceAddressBufferWithStaging(
                device,
                physicalDevice,
                commandPool,
                graphicsQueue,
                stack,
                instanceBytes,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                (op, code) ->
                    new GpuException(
                        GpuErrorCode.BACKEND_INIT_FAILED, op + " failed with code " + code, false));
        long instanceBufferDeviceAddress =
            VulkanBufferOps.getBufferDeviceAddress(device, instanceAlloc.buffer());
        if (instanceBufferDeviceAddress == 0L) {
          vkDestroyBuffer(device, instanceAlloc.buffer(), null);
          vkFreeMemory(device, instanceAlloc.memory(), null);
          throw new GpuException(
              GpuErrorCode.BACKEND_INIT_FAILED,
              "instance buffer device address resolution failed",
              false);
        }

        VkAccelerationStructureGeometryInstancesDataKHR instancesData =
            VkAccelerationStructureGeometryInstancesDataKHR.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR)
                .arrayOfPointers(false)
                .data(VkDeviceOrHostAddressConstKHR.calloc(stack).deviceAddress(instanceBufferDeviceAddress));

        VkAccelerationStructureGeometryKHR.Buffer geometryBuffer =
            VkAccelerationStructureGeometryKHR.calloc(1, stack);
        geometryBuffer
            .get(0)
            .sType(org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR)
            .geometryType(VK_GEOMETRY_TYPE_INSTANCES_KHR);
        geometryBuffer.get(0).geometry().instances(instancesData);

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo =
            VkAccelerationStructureBuildGeometryInfoKHR.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR)
                .type(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR)
                .flags(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
                .mode(VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR)
                .srcAccelerationStructure(VK_NULL_HANDLE)
                .dstAccelerationStructure(VK_NULL_HANDLE)
                .geometryCount(1)
                .pGeometries(geometryBuffer);

        IntBuffer maxPrimitiveCounts = stack.ints(instances.size());
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo =
            VkAccelerationStructureBuildSizesInfoKHR.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR);
        vkGetAccelerationStructureBuildSizesKHR(
            device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            buildInfo,
            maxPrimitiveCounts,
            sizeInfo);

        long tlasSize = sizeInfo.accelerationStructureSize();
        long scratchSize = sizeInfo.buildScratchSize();
        if (tlasSize <= 0L || scratchSize <= 0L) {
          vkDestroyBuffer(device, instanceAlloc.buffer(), null);
          vkFreeMemory(device, instanceAlloc.memory(), null);
          throw new GpuException(
              GpuErrorCode.BACKEND_INIT_FAILED,
              "invalid TLAS size query result: tlasSize=" + tlasSize + " scratchSize=" + scratchSize,
              false);
        }
        if (tlasSize > Integer.MAX_VALUE || scratchSize > Integer.MAX_VALUE) {
          vkDestroyBuffer(device, instanceAlloc.buffer(), null);
          vkFreeMemory(device, instanceAlloc.memory(), null);
          throw new GpuException(
              GpuErrorCode.BACKEND_INIT_FAILED,
              "TLAS sizes exceed current int-based allocation seam: tlasSize="
                  + tlasSize
                  + " scratchSize="
                  + scratchSize,
              false);
        }

        VulkanBufferAlloc tlasStorageAlloc =
            VulkanBufferOps.createBuffer(
                device,
                physicalDevice,
                stack,
                (int) tlasSize,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
        VulkanBufferAlloc scratchAlloc =
            VulkanBufferOps.createBuffer(
                device,
                physicalDevice,
                stack,
                (int) scratchSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

        final long[] tlasHandleRef = new long[] {VK_NULL_HANDLE};
        boolean success = false;
        VulkanGpuBuffer tlasStorageBuffer =
            new VulkanGpuBuffer(
                device,
                tlasStorageAlloc.buffer(),
                tlasStorageAlloc.memory(),
                tlasSize,
                GpuBufferUsage.STORAGE,
                GpuMemoryLocation.DEVICE_LOCAL,
                () -> {
                  vkDestroyBuffer(device, tlasStorageAlloc.buffer(), null);
                  vkFreeMemory(device, tlasStorageAlloc.memory(), null);
                });
        try {
          VkAccelerationStructureCreateInfoKHR createInfo =
              VkAccelerationStructureCreateInfoKHR.calloc(stack)
                  .sType(VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR)
                  .type(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR)
                  .size(tlasSize)
                  .buffer(tlasStorageAlloc.buffer())
                  .offset(0);
          LongBuffer pTlas = stack.mallocLong(1);
          int createResult = vkCreateAccelerationStructureKHR(device, createInfo, null, pTlas);
          if (createResult != VK_SUCCESS || pTlas.get(0) == VK_NULL_HANDLE) {
            throw new GpuException(
                GpuErrorCode.BACKEND_INIT_FAILED,
                "vkCreateAccelerationStructureKHR (TLAS) failed: " + createResult,
                false);
          }
          tlasHandleRef[0] = pTlas.get(0);

          long scratchAddress = VulkanBufferOps.getBufferDeviceAddress(device, scratchAlloc.buffer());
          if (scratchAddress == 0L) {
            throw new GpuException(
                GpuErrorCode.BACKEND_INIT_FAILED,
                "scratch buffer device address resolution failed",
                false);
          }

          buildInfo.dstAccelerationStructure(tlasHandleRef[0]);
          buildInfo.scratchData(VkDeviceOrHostAddressKHR.calloc(stack).deviceAddress(scratchAddress));
          VkAccelerationStructureBuildRangeInfoKHR.Buffer rangeInfo =
              VkAccelerationStructureBuildRangeInfoKHR.calloc(1, stack);
          rangeInfo.get(0)
              .primitiveCount(instances.size())
              .primitiveOffset(0)
              .firstVertex(0)
              .transformOffset(0);

          recordAndSubmitBuild(buildInfo, rangeInfo, stack);

          GpuRayTracingTlasPayload payload =
              GpuRayTracingTlasPayload.of(instances.size(), instanceBytes.remaining());
          GpuRayTracingTlasResource result =
              new GpuRayTracingTlasResource(
                  tlasStorageBuffer,
                  payload,
                  tlasHandleRef[0],
                  () -> {
                    vkDestroyAccelerationStructureKHR(device, tlasHandleRef[0], null);
                    tlasStorageBuffer.close();
                  });
          success = true;
          return result;
        } finally {
          vkDestroyBuffer(device, scratchAlloc.buffer(), null);
          vkFreeMemory(device, scratchAlloc.memory(), null);
          vkDestroyBuffer(device, instanceAlloc.buffer(), null);
          vkFreeMemory(device, instanceAlloc.memory(), null);
          if (!success) {
            if (tlasHandleRef[0] != VK_NULL_HANDLE) {
              vkDestroyAccelerationStructureKHR(device, tlasHandleRef[0], null);
            }
            vkDestroyBuffer(device, tlasStorageAlloc.buffer(), null);
            vkFreeMemory(device, tlasStorageAlloc.memory(), null);
          }
        }
      }
    }

    private ByteBuffer buildInstanceByteBuffer(List<GpuRayTracingTlasInstanceMetadata> instances, MemoryStack stack)
        throws GpuException {
      VkAccelerationStructureInstanceKHR.Buffer vkInstances =
          VkAccelerationStructureInstanceKHR.calloc(instances.size(), stack);
      for (int i = 0; i < instances.size(); i++) {
        GpuRayTracingTlasInstanceMetadata instance = instances.get(i);
        long blasHandle = instance.blasResource().accelerationStructureHandle();
        if (blasHandle == VK_NULL_HANDLE) {
          throw new GpuException(
              GpuErrorCode.INVALID_ARGUMENT,
              "instance BLAS handle is missing: index=" + i,
              false);
        }
        if (instance.blasResource().isClosed()) {
          throw new GpuException(
              GpuErrorCode.INVALID_ARGUMENT,
              "instance BLAS resource is closed: index=" + i,
              false);
        }
        VkAccelerationStructureDeviceAddressInfoKHR addressInfo =
            VkAccelerationStructureDeviceAddressInfoKHR.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR)
                .accelerationStructure(blasHandle);
        long blasDeviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device, addressInfo);
        if (blasDeviceAddress == 0L) {
          throw new GpuException(
              GpuErrorCode.BACKEND_INIT_FAILED,
              "BLAS device address resolution failed: index=" + i,
              false);
        }

        VkTransformMatrixKHR transform = VkTransformMatrixKHR.calloc(stack);
        float[] matrix = instance.transform3x4RowMajor();
        for (int m = 0; m < matrix.length; m++) {
          transform.matrix(m, matrix[m]);
        }

        vkInstances
            .get(i)
            .transform(transform)
            .instanceCustomIndex(instance.instanceCustomIndex())
            .mask(instance.mask())
            .instanceShaderBindingTableRecordOffset(instance.shaderBindingTableRecordOffset())
            .flags(instance.flags())
            .accelerationStructureReference(blasDeviceAddress);
      }
      return MemoryUtil.memByteBuffer(vkInstances.address(), instances.size() * VkAccelerationStructureInstanceKHR.SIZEOF);
    }

    private void recordAndSubmitBuild(
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo,
        VkAccelerationStructureBuildRangeInfoKHR.Buffer rangeInfo,
        MemoryStack stack)
        throws GpuException {
      VkCommandBuffer cmd = beginSingleTimeCommands(stack);
      try {
        VkAccelerationStructureBuildGeometryInfoKHR.Buffer buildInfos =
            VkAccelerationStructureBuildGeometryInfoKHR.calloc(1, stack);
        buildInfos.put(0, buildInfo);
        PointerBuffer ppRangeInfos = stack.mallocPointer(1);
        ppRangeInfos.put(0, rangeInfo.address());
        vkCmdBuildAccelerationStructuresKHR(cmd, buildInfos, ppRangeInfos);
        endSingleTimeCommands(cmd, stack);
      } catch (GpuException e) {
        vkFreeCommandBuffers(device, commandPool, stack.pointers(cmd.address()));
        throw e;
      }
    }

    private VkCommandBuffer beginSingleTimeCommands(MemoryStack stack) throws GpuException {
      VkCommandBufferAllocateInfo allocInfo =
          VkCommandBufferAllocateInfo.calloc(stack)
              .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
              .commandPool(commandPool)
              .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
              .commandBufferCount(1);
      PointerBuffer pCmd = stack.mallocPointer(1);
      int allocResult = vkAllocateCommandBuffers(device, allocInfo, pCmd);
      if (allocResult != VK_SUCCESS) {
        throw new GpuException(
            GpuErrorCode.BACKEND_INIT_FAILED,
            "vkAllocateCommandBuffers (TLAS build) failed: " + allocResult,
            false);
      }
      VkCommandBuffer cmd = new VkCommandBuffer(pCmd.get(0), device);
      VkCommandBufferBeginInfo beginInfo =
          VkCommandBufferBeginInfo.calloc(stack)
              .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
              .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
      int beginResult = vkBeginCommandBuffer(cmd, beginInfo);
      if (beginResult != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, stack.pointers(cmd.address()));
        throw new GpuException(
            GpuErrorCode.BACKEND_INIT_FAILED,
            "vkBeginCommandBuffer (TLAS build) failed: " + beginResult,
            false);
      }
      return cmd;
    }

    private void endSingleTimeCommands(VkCommandBuffer cmd, MemoryStack stack) throws GpuException {
      int endResult = vkEndCommandBuffer(cmd);
      if (endResult != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, stack.pointers(cmd.address()));
        throw new GpuException(
            GpuErrorCode.BACKEND_INIT_FAILED,
            "vkEndCommandBuffer (TLAS build) failed: " + endResult,
            false);
      }
      VkSubmitInfo submitInfo =
          VkSubmitInfo.calloc(stack)
              .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
              .pCommandBuffers(stack.pointers(cmd.address()));
      int submitResult = vkQueueSubmit(graphicsQueue, submitInfo, VK_NULL_HANDLE);
      if (submitResult != VK_SUCCESS) {
        vkFreeCommandBuffers(device, commandPool, stack.pointers(cmd.address()));
        throw new GpuException(
            GpuErrorCode.BACKEND_INIT_FAILED,
            "vkQueueSubmit (TLAS build) failed: " + submitResult,
            false);
      }
      int waitResult = vkQueueWaitIdle(graphicsQueue);
      vkFreeCommandBuffers(device, commandPool, stack.pointers(cmd.address()));
      if (waitResult != VK_SUCCESS) {
        throw new GpuException(
            GpuErrorCode.BACKEND_INIT_FAILED,
            "vkQueueWaitIdle (TLAS build) failed: " + waitResult,
            false);
      }
    }

    private void ensureBackendSupport() throws GpuException {
      if (!procAvailable.apply("vkCreateAccelerationStructureKHR")
          || !procAvailable.apply("vkGetAccelerationStructureBuildSizesKHR")
          || !procAvailable.apply("vkCmdBuildAccelerationStructuresKHR")
          || !procAvailable.apply("vkGetAccelerationStructureDeviceAddressKHR")
          || !procAvailable.apply("vkGetBufferDeviceAddress")) {
        throw new GpuException(
            GpuErrorCode.BACKEND_INIT_FAILED,
            "required Vulkan RT procedures are not available on this device",
            false);
      }
    }
  }
}
