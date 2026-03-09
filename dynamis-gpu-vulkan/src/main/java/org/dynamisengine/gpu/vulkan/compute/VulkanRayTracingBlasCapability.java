package org.dynamisengine.gpu.vulkan.compute;

import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_GEOMETRY_OPAQUE_BIT_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_GEOMETRY_TYPE_TRIANGLES_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.vkCmdBuildAccelerationStructuresKHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.vkCreateAccelerationStructureKHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.vkDestroyAccelerationStructureKHR;
import static org.lwjgl.vulkan.KHRAccelerationStructure.vkGetAccelerationStructureBuildSizesKHR;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
import static org.lwjgl.vulkan.VK10.VK_FORMAT_R32G32B32_SFLOAT;
import static org.lwjgl.vulkan.VK10.VK_INDEX_TYPE_UINT32;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_SHARING_MODE_EXCLUSIVE;
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
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Objects;
import java.util.function.Function;
import org.dynamisengine.gpu.api.buffer.GpuBuffer;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.gpu.RayTracingBlasCapability;
import org.dynamisengine.gpu.api.gpu.RayTracingBlasWork;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBlasPayload;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBlasResource;
import org.dynamisengine.gpu.api.resource.GpuRayTracingBuildInputResource;
import org.dynamisengine.gpu.api.resource.GpuRayTracingGeometryPayload;
import org.dynamisengine.gpu.vulkan.buffer.VulkanGpuBuffer;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanMemoryOps;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkAccelerationStructureBuildGeometryInfoKHR;
import org.lwjgl.vulkan.VkAccelerationStructureBuildRangeInfoKHR;
import org.lwjgl.vulkan.VkAccelerationStructureBuildSizesInfoKHR;
import org.lwjgl.vulkan.VkAccelerationStructureCreateInfoKHR;
import org.lwjgl.vulkan.VkAccelerationStructureGeometryKHR;
import org.lwjgl.vulkan.VkAccelerationStructureGeometryTrianglesDataKHR;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkDeviceOrHostAddressConstKHR;
import org.lwjgl.vulkan.VkDeviceOrHostAddressKHR;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkSubmitInfo;

/**
 * Vulkan-side BLAS capability with real acceleration-structure build execution when supported.
 */
public final class VulkanRayTracingBlasCapability implements RayTracingBlasCapability {
  @FunctionalInterface
  interface BlasExecutor {
    GpuRayTracingBlasResource execute(RayTracingBlasWork work) throws GpuException;
  }

  private final BlasExecutor blasExecutor;
  private boolean closed;

  public VulkanRayTracingBlasCapability(
      VkDevice device, VkPhysicalDevice physicalDevice, long commandPool, VkQueue graphicsQueue) {
    this.blasExecutor =
        new VulkanBlasExecutor(
            Objects.requireNonNull(device, "device"),
            Objects.requireNonNull(physicalDevice, "physicalDevice"),
            commandPool,
            Objects.requireNonNull(graphicsQueue, "graphicsQueue"));
  }

  VulkanRayTracingBlasCapability(BlasExecutor blasExecutor) {
    this.blasExecutor = Objects.requireNonNull(blasExecutor, "blasExecutor");
  }

  @Override
  public GpuRayTracingBlasResource execute(RayTracingBlasWork work) throws GpuException {
    ensureOpen();
    Objects.requireNonNull(work, "work");
    return blasExecutor.execute(work);
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

  private static final class VulkanBlasExecutor implements BlasExecutor {
    private final VkDevice device;
    private final VkPhysicalDevice physicalDevice;
    private final long commandPool;
    private final VkQueue graphicsQueue;
    private final Function<String, Boolean> procAvailable;

    private VulkanBlasExecutor(
        VkDevice device, VkPhysicalDevice physicalDevice, long commandPool, VkQueue graphicsQueue) {
      this(device, physicalDevice, commandPool, graphicsQueue, name -> vkGetDeviceProcAddr(device, name) != 0L);
    }

    private VulkanBlasExecutor(
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
    public GpuRayTracingBlasResource execute(RayTracingBlasWork work) throws GpuException {
      ensureBackendSupport();
      GpuRayTracingBuildInputResource buildInput = work.buildInputResource();
      if (buildInput.isClosed()) {
        throw new IllegalStateException("buildInputResource is already closed");
      }

      GpuRayTracingBlasPayload payload =
          GpuRayTracingBlasPayload.forGeometryResource(buildInput.payload().geometryResource());

      try (MemoryStack stack = MemoryStack.stackPush()) {
        ParsedBuildRanges parsed = parseBuildRanges(buildInput.payload().geometryResource().payload(), stack);
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo =
            createBuildInfo(buildInput, parsed.geometryBuffer(), parsed.regionCount(), stack);

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo =
            VkAccelerationStructureBuildSizesInfoKHR.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR);
        vkGetAccelerationStructureBuildSizesKHR(
            device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            buildInfo,
            parsed.maxPrimitiveCounts(),
            sizeInfo);

        long accelerationStructureSize = sizeInfo.accelerationStructureSize();
        long scratchSize = sizeInfo.buildScratchSize();
        if (accelerationStructureSize <= 0L || scratchSize <= 0L) {
          throw new GpuException(
              GpuErrorCode.BACKEND_INIT_FAILED,
              "invalid BLAS size query result: asSize="
                  + accelerationStructureSize
                  + " scratchSize="
                  + scratchSize,
              false);
        }
        if (accelerationStructureSize > Integer.MAX_VALUE || scratchSize > Integer.MAX_VALUE) {
          throw new GpuException(
              GpuErrorCode.BACKEND_INIT_FAILED,
              "BLAS sizes exceed current int-based allocation seam: asSize="
                  + accelerationStructureSize
                  + " scratchSize="
                  + scratchSize,
              false);
        }

        VulkanBufferAlloc asStorageAlloc =
            VulkanMemoryOps.createBuffer(
                device,
                physicalDevice,
                stack,
                (int) accelerationStructureSize,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
        VulkanBufferAlloc scratchAlloc =
            VulkanMemoryOps.createBuffer(
                device,
                physicalDevice,
                stack,
                (int) scratchSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);

        boolean success = false;
        final long[] asHandleRef = new long[] {VK_NULL_HANDLE};
        VulkanGpuBuffer asStorageBuffer =
            new VulkanGpuBuffer(
                device,
                asStorageAlloc.buffer(),
                asStorageAlloc.memory(),
                accelerationStructureSize,
                GpuBufferUsage.STORAGE,
                GpuMemoryLocation.DEVICE_LOCAL,
                () -> {
                  vkDestroyBuffer(device, asStorageAlloc.buffer(), null);
                  vkFreeMemory(device, asStorageAlloc.memory(), null);
                });
        try {
          VkAccelerationStructureCreateInfoKHR createInfo =
              VkAccelerationStructureCreateInfoKHR.calloc(stack)
                  .sType(VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR)
                  .buffer(asStorageAlloc.buffer())
                  .size(accelerationStructureSize)
                  .type(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR)
                  .offset(0);
          java.nio.LongBuffer pAs = stack.mallocLong(1);
          int createResult = vkCreateAccelerationStructureKHR(device, createInfo, null, pAs);
          if (createResult != VK_SUCCESS || pAs.get(0) == VK_NULL_HANDLE) {
            throw new GpuException(
                GpuErrorCode.BACKEND_INIT_FAILED,
                "vkCreateAccelerationStructureKHR failed: " + createResult,
                false);
          }
          asHandleRef[0] = pAs.get(0);

          long scratchAddress = VulkanMemoryOps.getBufferDeviceAddress(device, scratchAlloc.buffer());
          if (scratchAddress == 0L) {
            throw new GpuException(
                GpuErrorCode.BACKEND_INIT_FAILED,
                "scratch buffer device address resolution failed",
                false);
          }

          buildInfo.dstAccelerationStructure(asHandleRef[0]);
          buildInfo.scratchData(VkDeviceOrHostAddressKHR.calloc(stack).deviceAddress(scratchAddress));
          recordAndSubmitBuild(buildInfo, parsed.rangeBuffer(), stack);

          GpuRayTracingBlasResource result =
              new GpuRayTracingBlasResource(
                  asStorageBuffer,
                  payload,
                  buildInput.payload().geometryResource(),
                  asHandleRef[0],
                  () -> {
                    vkDestroyAccelerationStructureKHR(device, asHandleRef[0], null);
                    asStorageBuffer.close();
                  });
          success = true;
          return result;
        } finally {
          vkDestroyBuffer(device, scratchAlloc.buffer(), null);
          vkFreeMemory(device, scratchAlloc.memory(), null);
          if (!success) {
            if (asHandleRef[0] != VK_NULL_HANDLE) {
              vkDestroyAccelerationStructureKHR(device, asHandleRef[0], null);
            }
            vkDestroyBuffer(device, asStorageAlloc.buffer(), null);
            vkFreeMemory(device, asStorageAlloc.memory(), null);
          }
        }
      }
    }

    private void ensureBackendSupport() throws GpuException {
      if (!procAvailable.apply("vkCreateAccelerationStructureKHR")
          || !procAvailable.apply("vkGetAccelerationStructureBuildSizesKHR")
          || !procAvailable.apply("vkCmdBuildAccelerationStructuresKHR")
          || !procAvailable.apply("vkGetBufferDeviceAddress")) {
        throw new GpuException(
            GpuErrorCode.BACKEND_INIT_FAILED,
            "required Vulkan RT procedures are not available on this device",
            false);
      }
    }

    private ParsedBuildRanges parseBuildRanges(GpuRayTracingGeometryPayload payload, MemoryStack stack) {
      ByteBuffer regions = payload.regionsBytes().order(ByteOrder.LITTLE_ENDIAN);
      int regionCount = payload.regionCount();
      int offsetInts = payload.regionsOffsetInts();
      int strideInts = payload.regionsStrideInts();

      VkAccelerationStructureBuildRangeInfoKHR.Buffer rangeBuffer =
          VkAccelerationStructureBuildRangeInfoKHR.calloc(regionCount, stack);
      IntBuffer maxPrimitiveCounts = stack.mallocInt(regionCount);
      VkAccelerationStructureGeometryKHR.Buffer geometryBuffer =
          VkAccelerationStructureGeometryKHR.calloc(regionCount, stack);

      for (int i = 0; i < regionCount; i++) {
        int baseInt = offsetInts + (i * strideInts);
        int baseByte = baseInt * Integer.BYTES;
        int firstIndex = regions.getInt(baseByte + Integer.BYTES);
        int indexCount = regions.getInt(baseByte + (2 * Integer.BYTES));
        if ((indexCount % 3) != 0) {
          throw new IllegalArgumentException("indexCount must be triangle-aligned (multiple of 3): region=" + i);
        }
        int primitiveCount = indexCount / 3;
        maxPrimitiveCounts.put(i, primitiveCount);
        rangeBuffer.get(i)
            .primitiveCount(primitiveCount)
            .primitiveOffset(firstIndex * Integer.BYTES)
            .firstVertex(0)
            .transformOffset(0);
      }
      maxPrimitiveCounts.flip();
      return new ParsedBuildRanges(regionCount, geometryBuffer, rangeBuffer, maxPrimitiveCounts);
    }

    private VkAccelerationStructureBuildGeometryInfoKHR createBuildInfo(
        GpuRayTracingBuildInputResource buildInput,
        VkAccelerationStructureGeometryKHR.Buffer geometryBuffer,
        int geometryCount,
        MemoryStack stack) {
      for (int i = 0; i < geometryCount; i++) {
        VkAccelerationStructureGeometryTrianglesDataKHR triangles =
            VkAccelerationStructureGeometryTrianglesDataKHR.calloc(stack)
                .sType(org.lwjgl.vulkan.KHRAccelerationStructure.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR)
                .vertexFormat(VK_FORMAT_R32G32B32_SFLOAT)
                .vertexStride(buildInput.vertexStrideBytes())
                .maxVertex(buildInput.maxVertexIndex())
                .indexType(VK_INDEX_TYPE_UINT32)
                .vertexData(
                    VkDeviceOrHostAddressConstKHR.calloc(stack)
                        .deviceAddress(
                            buildInput.vertexBufferDeviceAddress()
                                + buildInput.vertexDataOffsetBytes()))
                .indexData(
                    VkDeviceOrHostAddressConstKHR.calloc(stack)
                        .deviceAddress(
                            buildInput.indexBufferDeviceAddress()
                                + buildInput.indexDataOffsetBytes()));

        geometryBuffer
            .get(i)
            .sType(VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR)
            .geometryType(VK_GEOMETRY_TYPE_TRIANGLES_KHR)
            .flags(VK_GEOMETRY_OPAQUE_BIT_KHR);
        geometryBuffer.get(i).geometry().triangles(triangles);
      }

      return VkAccelerationStructureBuildGeometryInfoKHR.calloc(stack)
          .sType(VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR)
          .type(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR)
          .flags(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
          .mode(VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR)
          .srcAccelerationStructure(VK_NULL_HANDLE)
          .dstAccelerationStructure(VK_NULL_HANDLE)
          .geometryCount(geometryCount)
          .pGeometries(geometryBuffer);
    }

    private void recordAndSubmitBuild(
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo,
        VkAccelerationStructureBuildRangeInfoKHR.Buffer rangeBuffer,
        MemoryStack stack)
        throws GpuException {
      VkCommandBuffer cmd = beginSingleTimeCommands(stack);
      try {
        VkAccelerationStructureBuildGeometryInfoKHR.Buffer buildInfos =
            VkAccelerationStructureBuildGeometryInfoKHR.calloc(1, stack);
        buildInfos.put(0, buildInfo);
        PointerBuffer ppRangeInfos = stack.mallocPointer(1);
        ppRangeInfos.put(0, rangeBuffer.address());
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
            "vkAllocateCommandBuffers (BLAS build) failed: " + allocResult,
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
            "vkBeginCommandBuffer (BLAS build) failed: " + beginResult,
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
            "vkEndCommandBuffer (BLAS build) failed: " + endResult,
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
            "vkQueueSubmit (BLAS build) failed: " + submitResult,
            false);
      }
      int waitResult = vkQueueWaitIdle(graphicsQueue);
      vkFreeCommandBuffers(device, commandPool, stack.pointers(cmd.address()));
      if (waitResult != VK_SUCCESS) {
        throw new GpuException(
            GpuErrorCode.BACKEND_INIT_FAILED,
            "vkQueueWaitIdle (BLAS build) failed: " + waitResult,
            false);
      }
    }
  }

  private record ParsedBuildRanges(
      int regionCount,
      VkAccelerationStructureGeometryKHR.Buffer geometryBuffer,
      VkAccelerationStructureBuildRangeInfoKHR.Buffer rangeBuffer,
      IntBuffer maxPrimitiveCounts) {}
}
