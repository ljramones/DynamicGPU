package org.dynamisengine.gpu.vulkan.upload;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import org.dynamisengine.gpu.api.buffer.GpuBufferUsage;
import org.dynamisengine.gpu.api.buffer.GpuMemoryLocation;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshResource;
import org.dynamisengine.gpu.api.upload.GpuGeometryUploadPlan;
import org.dynamisengine.gpu.api.upload.GpuUploadExecutor;
import org.dynamisengine.gpu.vulkan.buffer.VulkanGpuBuffer;
import org.dynamisengine.gpu.vulkan.memory.VulkanBufferAlloc;
import org.dynamisengine.gpu.vulkan.memory.VulkanMemoryOps;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkBufferCopy;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkSubmitInfo;

import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_SUBMIT_INFO;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkAllocateCommandBuffers;
import static org.lwjgl.vulkan.VK10.vkBeginCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkCmdCopyBuffer;
import static org.lwjgl.vulkan.VK10.vkEndCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkFreeCommandBuffers;
import static org.lwjgl.vulkan.VK10.vkQueueSubmit;
import static org.lwjgl.vulkan.VK10.vkQueueWaitIdle;

/**
 * Vulkan implementation of {@link GpuUploadExecutor} for runtime geometry payloads.
 */
public final class VulkanGpuUploadExecutor implements GpuUploadExecutor, AutoCloseable {
  public enum UploadPathMode {
    SIMPLE,
    OPTIMIZED
  }

  private final VkDevice device;
  private final VkPhysicalDevice physicalDevice;
  private final long commandPool;
  private final VkQueue graphicsQueue;
  private final UploadPathMode mode;
  private final VulkanUploadArena uploadArena;
  private final VulkanDeviceLocalBufferPool deviceBufferPool;
  private final BiFunction<String, Integer, GpuException> vkFailure;

  public VulkanGpuUploadExecutor(
      VkDevice device,
      VkPhysicalDevice physicalDevice,
      long commandPool,
      VkQueue graphicsQueue) {
    this(device, physicalDevice, commandPool, graphicsQueue, UploadPathMode.OPTIMIZED);
  }

  public VulkanGpuUploadExecutor(
      VkDevice device,
      VkPhysicalDevice physicalDevice,
      long commandPool,
      VkQueue graphicsQueue,
      UploadPathMode mode) {
    this.device = Objects.requireNonNull(device, "device");
    this.physicalDevice = Objects.requireNonNull(physicalDevice, "physicalDevice");
    if (commandPool == 0L) {
      throw new IllegalArgumentException("commandPool must not be VK_NULL_HANDLE");
    }
    this.commandPool = commandPool;
    this.graphicsQueue = Objects.requireNonNull(graphicsQueue, "graphicsQueue");
    this.mode = Objects.requireNonNull(mode, "mode");
    this.vkFailure =
        (op, code) ->
            new GpuException(
                GpuErrorCode.BACKEND_INIT_FAILED, op + " failed with code " + code, false);
    this.uploadArena = mode == UploadPathMode.OPTIMIZED ? createUploadArena(device, physicalDevice) : null;
    this.deviceBufferPool =
        mode == UploadPathMode.OPTIMIZED
            ? new VulkanDeviceLocalBufferPool(device, physicalDevice)
            : null;
  }

  @Override
  public GpuMeshResource upload(GpuGeometryUploadPlan plan) throws GpuException {
    if (mode == UploadPathMode.SIMPLE) {
      return uploadSimple(plan);
    }
    return uploadBatch(List.of(plan)).get(0);
  }

  public List<GpuMeshResource> uploadBatch(List<GpuGeometryUploadPlan> plans) throws GpuException {
    if (mode == UploadPathMode.SIMPLE) {
      ArrayList<GpuMeshResource> resources = new ArrayList<>(plans.size());
      for (GpuGeometryUploadPlan plan : plans) {
        resources.add(uploadSimple(plan));
      }
      return List.copyOf(resources);
    }
    return uploadBatchOptimized(plans);
  }

  @Override
  public void close() {
    if (deviceBufferPool != null) {
      deviceBufferPool.close();
    }
    if (uploadArena != null) {
      uploadArena.close();
    }
  }

  private List<GpuMeshResource> uploadBatchOptimized(List<GpuGeometryUploadPlan> plans)
      throws GpuException {
    Objects.requireNonNull(plans, "plans");
    if (plans.isEmpty()) {
      return List.of();
    }
    ArrayList<PreparedPlan> preparedPlans = new ArrayList<>(plans.size());
    try (MemoryStack stack = MemoryStack.stackPush()) {
      uploadArena.reset();
      for (GpuGeometryUploadPlan plan : plans) {
        preparedPlans.add(preparePlanForBatch(stack, plan));
      }
      submitCopies(stack, preparedPlans);
      ArrayList<GpuMeshResource> resources = new ArrayList<>(preparedPlans.size());
      for (PreparedPlan prepared : preparedPlans) {
        GpuMeshResource resource =
            new GpuMeshResource(
                prepared.vertexBuffer,
                prepared.indexBuffer,
                prepared.plan.vertexLayout(),
                prepared.plan.indexType(),
                prepared.plan.submeshes());
        resources.add(resource);
      }
      return List.copyOf(resources);
    } catch (Throwable t) {
      for (PreparedPlan prepared : preparedPlans) {
        closeQuietly(prepared.indexBuffer);
        closeQuietly(prepared.vertexBuffer);
      }
      if (t instanceof GpuException gpuException) {
        throw gpuException;
      }
      throw new GpuException(
          GpuErrorCode.BACKEND_INIT_FAILED,
          "Optimized upload batch failed: " + t.getMessage(),
          t,
          false);
    }
  }

  private GpuMeshResource uploadSimple(GpuGeometryUploadPlan plan) throws GpuException {
    Objects.requireNonNull(plan, "plan");

    VulkanGpuBuffer vertexBuffer = null;
    VulkanGpuBuffer indexBuffer = null;
    try (MemoryStack stack = MemoryStack.stackPush()) {
      ByteBuffer vertexBytes = toDirectCopy(plan.vertexData());
      VulkanBufferAlloc vertexAlloc =
          VulkanMemoryOps.createDeviceLocalBufferWithStaging(
              device,
              physicalDevice,
              commandPool,
              graphicsQueue,
              stack,
              vertexBytes,
              toVkBufferUsage(GpuBufferUsage.VERTEX),
              vkFailure);
      vertexBuffer =
          new VulkanGpuBuffer(
              device,
              vertexAlloc.buffer(),
              vertexAlloc.memory(),
              vertexBytes.remaining(),
              GpuBufferUsage.VERTEX,
              GpuMemoryLocation.DEVICE_LOCAL);

      if (plan.indexData() != null) {
        ByteBuffer indexBytes = toDirectCopy(plan.indexData());
        VulkanBufferAlloc indexAlloc =
            VulkanMemoryOps.createDeviceLocalBufferWithStaging(
                device,
                physicalDevice,
                commandPool,
                graphicsQueue,
                stack,
                indexBytes,
                toVkBufferUsage(GpuBufferUsage.INDEX),
                vkFailure);
        indexBuffer =
            new VulkanGpuBuffer(
                device,
                indexAlloc.buffer(),
                indexAlloc.memory(),
                indexBytes.remaining(),
                GpuBufferUsage.INDEX,
                GpuMemoryLocation.DEVICE_LOCAL);
      }

      return new GpuMeshResource(
          vertexBuffer, indexBuffer, plan.vertexLayout(), plan.indexType(), plan.submeshes());
    } catch (Throwable t) {
      closeQuietly(indexBuffer);
      closeQuietly(vertexBuffer);
      if (t instanceof GpuException gpuException) {
        throw gpuException;
      }
      throw new GpuException(
          GpuErrorCode.BACKEND_INIT_FAILED, "Geometry upload failed: " + t.getMessage(), t, false);
    }
  }

  private PreparedPlan preparePlanForBatch(MemoryStack stack, GpuGeometryUploadPlan plan)
      throws GpuException {
    Objects.requireNonNull(plan, "plan");
    ByteBuffer vertexBytes = toDirectCopy(plan.vertexData());
    int vertexUsage = toVkBufferUsage(GpuBufferUsage.VERTEX) | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VulkanDeviceLocalBufferPool.Lease vertexLease =
        deviceBufferPool.acquire(stack, vertexBytes.remaining(), vertexUsage);
    VulkanUploadArena.Slice vertexSlice = uploadArena.stage(stack, vertexBytes, 16);
    VulkanGpuBuffer vertexBuffer =
        new VulkanGpuBuffer(
            device,
            vertexLease.bufferHandle(),
            vertexLease.memoryHandle(),
            vertexLease.requestedSizeBytes(),
            GpuBufferUsage.VERTEX,
            GpuMemoryLocation.DEVICE_LOCAL,
            vertexLease.releaseAction());

    VulkanGpuBuffer indexBuffer = null;
    CopyOp indexCopy = null;
    if (plan.indexData() != null) {
      ByteBuffer indexBytes = toDirectCopy(plan.indexData());
      int indexUsage = toVkBufferUsage(GpuBufferUsage.INDEX) | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      VulkanDeviceLocalBufferPool.Lease indexLease =
          deviceBufferPool.acquire(stack, indexBytes.remaining(), indexUsage);
      VulkanUploadArena.Slice indexSlice = uploadArena.stage(stack, indexBytes, 16);
      indexBuffer =
          new VulkanGpuBuffer(
              device,
              indexLease.bufferHandle(),
              indexLease.memoryHandle(),
              indexLease.requestedSizeBytes(),
              GpuBufferUsage.INDEX,
              GpuMemoryLocation.DEVICE_LOCAL,
              indexLease.releaseAction());
      indexCopy = new CopyOp(indexSlice.stagingBuffer(), indexSlice.offsetBytes(), indexLease.bufferHandle(), 0, indexSlice.sizeBytes());
    }

    return new PreparedPlan(
        plan,
        vertexBuffer,
        indexBuffer,
        new CopyOp(
            vertexSlice.stagingBuffer(),
            vertexSlice.offsetBytes(),
            vertexLease.bufferHandle(),
            0,
            vertexSlice.sizeBytes()),
        indexCopy);
  }

  private void submitCopies(MemoryStack stack, List<PreparedPlan> preparedPlans) throws GpuException {
    VkCommandBufferAllocateInfo allocInfo =
        VkCommandBufferAllocateInfo.calloc(stack)
            .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
            .commandPool(commandPool)
            .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
            .commandBufferCount(1);
    var pCommandBuffer = stack.mallocPointer(1);
    int allocResult = vkAllocateCommandBuffers(device, allocInfo, pCommandBuffer);
    if (allocResult != VK_SUCCESS) {
      throw new GpuException(
          GpuErrorCode.BACKEND_INIT_FAILED,
          "vkAllocateCommandBuffers(upload-batch) failed: " + allocResult,
          false);
    }
    VkCommandBuffer commandBuffer = new VkCommandBuffer(pCommandBuffer.get(0), device);
    try {
      VkCommandBufferBeginInfo beginInfo =
          VkCommandBufferBeginInfo.calloc(stack)
              .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
              .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
      int beginResult = vkBeginCommandBuffer(commandBuffer, beginInfo);
      if (beginResult != VK_SUCCESS) {
        throw vkFailure.apply("vkBeginCommandBuffer(upload-batch)", beginResult);
      }
      for (PreparedPlan preparedPlan : preparedPlans) {
        recordCopy(stack, commandBuffer, preparedPlan.vertexCopy);
        if (preparedPlan.indexCopy != null) {
          recordCopy(stack, commandBuffer, preparedPlan.indexCopy);
        }
      }
      int endResult = vkEndCommandBuffer(commandBuffer);
      if (endResult != VK_SUCCESS) {
        throw vkFailure.apply("vkEndCommandBuffer(upload-batch)", endResult);
      }
      VkSubmitInfo submitInfo =
          VkSubmitInfo.calloc(stack)
              .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
              .pCommandBuffers(stack.pointers(commandBuffer.address()));
      int submitResult = vkQueueSubmit(graphicsQueue, submitInfo, VK_NULL_HANDLE);
      if (submitResult != VK_SUCCESS) {
        throw vkFailure.apply("vkQueueSubmit(upload-batch)", submitResult);
      }
      int waitResult = vkQueueWaitIdle(graphicsQueue);
      if (waitResult != VK_SUCCESS) {
        throw vkFailure.apply("vkQueueWaitIdle(upload-batch)", waitResult);
      }
    } finally {
      vkFreeCommandBuffers(device, commandPool, stack.pointers(commandBuffer.address()));
    }
  }

  private static void recordCopy(MemoryStack stack, VkCommandBuffer commandBuffer, CopyOp copy) {
    VkBufferCopy.Buffer region = VkBufferCopy.calloc(1, stack);
    region.get(0).srcOffset(copy.srcOffsetBytes).dstOffset(copy.dstOffsetBytes).size(copy.sizeBytes);
    vkCmdCopyBuffer(commandBuffer, copy.srcBuffer, copy.dstBuffer, region);
  }

  static int toVkBufferUsage(GpuBufferUsage usage) {
    return switch (usage) {
      case VERTEX -> VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
      case INDEX -> VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
      case TRANSFER_SRC -> VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      case TRANSFER_DST -> VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    };
  }

  static int toVkMemoryProperties(GpuMemoryLocation location) {
    return switch (location) {
      case DEVICE_LOCAL -> VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      case HOST_VISIBLE -> VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    };
  }

  static ByteBuffer toDirectCopy(ByteBuffer source) {
    ByteBuffer duplicate = source.duplicate();
    ByteBuffer direct = ByteBuffer.allocateDirect(duplicate.remaining());
    direct.put(duplicate);
    direct.flip();
    return direct;
  }

  private static void closeQuietly(VulkanGpuBuffer buffer) {
    if (buffer == null) {
      return;
    }
    try {
      buffer.close();
    } catch (RuntimeException ignored) {
      // Preserve original failure in upload path.
    }
  }

  private static VulkanUploadArena createUploadArena(VkDevice device, VkPhysicalDevice physicalDevice) {
    try {
      return new VulkanUploadArena(device, physicalDevice, 8 * 1024 * 1024);
    } catch (GpuException e) {
      throw new IllegalStateException("Failed to initialize Vulkan upload arena", e);
    }
  }

  private record PreparedPlan(
      GpuGeometryUploadPlan plan,
      VulkanGpuBuffer vertexBuffer,
      VulkanGpuBuffer indexBuffer,
      CopyOp vertexCopy,
      CopyOp indexCopy) {}

  private record CopyOp(
      long srcBuffer, long srcOffsetBytes, long dstBuffer, long dstOffsetBytes, long sizeBytes) {}
}
