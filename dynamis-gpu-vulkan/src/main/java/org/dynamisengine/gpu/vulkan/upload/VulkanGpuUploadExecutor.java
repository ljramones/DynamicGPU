package org.dynamisengine.gpu.vulkan.upload;

import java.nio.ByteBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
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
import org.lwjgl.vulkan.VkFenceCreateInfo;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkSubmitInfo;

import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
import static org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
import static org.lwjgl.vulkan.VK10.VK_FENCE_CREATE_SIGNALED_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
import static org.lwjgl.vulkan.VK10.VK_NOT_READY;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_SUBMIT_INFO;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkAllocateCommandBuffers;
import static org.lwjgl.vulkan.VK10.vkBeginCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkCmdCopyBuffer;
import static org.lwjgl.vulkan.VK10.vkCreateFence;
import static org.lwjgl.vulkan.VK10.vkDestroyFence;
import static org.lwjgl.vulkan.VK10.vkEndCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkFreeCommandBuffers;
import static org.lwjgl.vulkan.VK10.vkGetFenceStatus;
import static org.lwjgl.vulkan.VK10.vkQueueSubmit;
import static org.lwjgl.vulkan.VK10.vkQueueWaitIdle;
import static org.lwjgl.vulkan.VK10.vkWaitForFences;

/**
 * Vulkan implementation of {@link GpuUploadExecutor} for runtime geometry payloads.
 */
public final class VulkanGpuUploadExecutor implements GpuUploadExecutor, AutoCloseable {
  public enum UploadPathMode {
    SIMPLE,
    OPTIMIZED,
    OPTIMIZED_DEFERRED
  }

  private static final long DEFAULT_FENCE_WAIT_NANOS = 5_000_000_000L;
  private static final boolean DEBUG_UPLOAD =
      Boolean.parseBoolean(System.getProperty("dynamisgpu.upload.debug", "false"))
          || Boolean.parseBoolean(System.getenv().getOrDefault("DYNAMISGPU_UPLOAD_DEBUG", "false"));

  private final VkDevice device;
  private final VkPhysicalDevice physicalDevice;
  private final long commandPool;
  private final VkQueue graphicsQueue;
  private final UploadPathMode mode;
  private final VulkanUploadArena uploadArena;
  private final VulkanDeviceLocalBufferPool deviceBufferPool;
  private final BiFunction<String, Integer, GpuException> vkFailure;
  private final ArrayDeque<PendingSubmission> pendingSubmissions = new ArrayDeque<>();
  private final AtomicLong nextSubmissionId = new AtomicLong(1L);
  private ByteBuffer reusableDirectCopyBuffer;
  private final Map<Long, List<Range>> srcRangesByBufferScratch = new HashMap<>();
  private final Map<Long, List<Range>> dstRangesByBufferScratch = new HashMap<>();

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
    boolean optimizedMode = mode != UploadPathMode.SIMPLE;
    this.uploadArena = optimizedMode ? createUploadArena(device, physicalDevice) : null;
    this.deviceBufferPool =
        optimizedMode
            ? new VulkanDeviceLocalBufferPool(device, physicalDevice)
            : null;
  }

  @Override
  public synchronized GpuMeshResource upload(GpuGeometryUploadPlan plan) throws GpuException {
    if (mode == UploadPathMode.SIMPLE) {
      return uploadSimple(plan);
    }
    return uploadBatch(List.of(plan)).get(0);
  }

  public synchronized List<GpuMeshResource> uploadBatch(List<GpuGeometryUploadPlan> plans)
      throws GpuException {
    if (mode == UploadPathMode.SIMPLE) {
      ArrayList<GpuMeshResource> resources = new ArrayList<>(plans.size());
      for (GpuGeometryUploadPlan plan : plans) {
        resources.add(uploadSimple(plan));
      }
      return List.copyOf(resources);
    }
    DeferredUploadBatch pending = submitBatchDeferred(plans);
    return completeDeferredBatch(pending, DEFAULT_FENCE_WAIT_NANOS);
  }

  public synchronized DeferredUploadBatch submitBatchDeferred(List<GpuGeometryUploadPlan> plans)
      throws GpuException {
    Objects.requireNonNull(plans, "plans");
    if (plans.isEmpty()) {
      throw new IllegalArgumentException("plans must not be empty");
    }
    if (mode == UploadPathMode.SIMPLE) {
      throw new IllegalStateException("Deferred submit is unavailable in SIMPLE mode");
    }
    if (!pendingSubmissions.isEmpty()) {
      throw new IllegalStateException(
          "Deferred upload already in flight; call completeDeferredBatch or drainDeferredSubmissions first");
    }

    ArrayList<PreparedPlan> preparedPlans = new ArrayList<>(plans.size());
    ArrayList<PlanSizing> planSizings = new ArrayList<>(plans.size());
    long totalBytes = 0L;
    long estimatedBytes = 0L;
    for (GpuGeometryUploadPlan plan : plans) {
      PlanSizing sizing = PlanSizing.from(plan);
      planSizings.add(sizing);
      totalBytes += sizing.totalBytes();
      estimatedBytes += sizing.estimatedStagingBytes();
    }
    long submissionId = nextSubmissionId.getAndIncrement();
    try (MemoryStack stack = MemoryStack.stackPush()) {
      if (estimatedBytes > Integer.MAX_VALUE) {
        throw new GpuException(
            GpuErrorCode.BACKEND_INIT_FAILED,
            "Deferred upload batch too large for staging arena capacity tracking: " + estimatedBytes,
            false);
      }
      uploadArena.reserveForBatch(stack, (int) estimatedBytes);
      uploadArena.reset();
      for (PlanSizing sizing : planSizings) {
        preparedPlans.add(preparePlanForBatch(stack, sizing));
      }
      validateCopyOps(preparedPlans, submissionId, srcRangesByBufferScratch, dstRangesByBufferScratch);
      SubmissionHandles handles = submitCopies(stack, preparedPlans, false);
      uploadArena.markSubmissionInFlight(submissionId);
      PendingSubmission pending =
          new PendingSubmission(
              submissionId,
              handles.commandBufferHandle(),
              handles.fenceHandle(),
              preparedPlans,
              totalBytes);
      pendingSubmissions.addLast(pending);
      return new DeferredUploadBatch(pending.submissionId, pending.planCount(), pending.totalBytes);
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
          "Deferred upload submit failed: " + t.getMessage(),
          t,
          false);
    }
  }

  public synchronized List<GpuMeshResource> completeDeferredBatch(
      DeferredUploadBatch batch, long timeoutNanos) throws GpuException {
    Objects.requireNonNull(batch, "batch");
    if (pendingSubmissions.isEmpty()) {
      throw new IllegalStateException("No deferred upload submission is pending");
    }
    PendingSubmission pending = pendingSubmissions.peekFirst();
    if (pending.submissionId != batch.submissionId()) {
      throw new IllegalArgumentException(
          "batch submissionId="
              + batch.submissionId()
              + " does not match pending submissionId="
              + pending.submissionId);
    }
    pendingSubmissions.removeFirst();
    return retireSubmission(pending, true, timeoutNanos);
  }

  public synchronized List<GpuMeshResource> tryCollectDeferredBatch() throws GpuException {
    if (pendingSubmissions.isEmpty()) {
      return List.of();
    }
    PendingSubmission pending = pendingSubmissions.peekFirst();
    int status = fenceStatus(pending.fenceHandle);
    if (status == VK_NOT_READY) {
      return List.of();
    }
    if (status != VK_SUCCESS) {
      throw vkFailure.apply("vkGetFenceStatus(upload-batch)", status);
    }
    pendingSubmissions.removeFirst();
    return retireSubmission(pending, false, 0L);
  }

  public synchronized List<GpuMeshResource> drainDeferredSubmissions() throws GpuException {
    ArrayList<GpuMeshResource> resources = new ArrayList<>();
    while (!pendingSubmissions.isEmpty()) {
      PendingSubmission pending = pendingSubmissions.removeFirst();
      resources.addAll(retireSubmission(pending, true, DEFAULT_FENCE_WAIT_NANOS));
    }
    return List.copyOf(resources);
  }

  public synchronized int pendingSubmissionCount() {
    return pendingSubmissions.size();
  }

  public UploadPathMode mode() {
    return mode;
  }

  @Override
  public synchronized void close() {
    RuntimeException closeFailure = null;
    while (!pendingSubmissions.isEmpty()) {
      PendingSubmission pending = pendingSubmissions.removeFirst();
      try {
        waitFenceOrThrow(pending.fenceHandle, DEFAULT_FENCE_WAIT_NANOS);
      } catch (GpuException e) {
        if (closeFailure == null) {
          closeFailure = new RuntimeException(e);
        }
      }
      destroyFenceQuietly(pending.fenceHandle);
      freeCommandBufferQuietly(pending.commandBufferHandle);
      for (PreparedPlan preparedPlan : pending.preparedPlans) {
        closeQuietly(preparedPlan.indexBuffer);
        closeQuietly(preparedPlan.vertexBuffer);
      }
      uploadArena.retireSubmission(pending.submissionId);
    }
    if (deviceBufferPool != null) {
      deviceBufferPool.close();
    }
    if (uploadArena != null) {
      uploadArena.close();
    }
    if (closeFailure != null) {
      throw closeFailure;
    }
  }

  private List<GpuMeshResource> retireSubmission(
      PendingSubmission pending, boolean waitForCompletion, long timeoutNanos) throws GpuException {
    try {
      if (waitForCompletion) {
        waitFenceOrThrow(pending.fenceHandle, timeoutNanos);
      }
      ArrayList<GpuMeshResource> resources = new ArrayList<>(pending.preparedPlans.size());
      for (PreparedPlan prepared : pending.preparedPlans) {
        resources.add(
            new GpuMeshResource(
                prepared.vertexBuffer,
                prepared.indexBuffer,
                prepared.plan.vertexLayout(),
                prepared.plan.indexType(),
                prepared.plan.submeshes()));
      }
      return List.copyOf(resources);
    } finally {
      destroyFenceQuietly(pending.fenceHandle);
      freeCommandBufferQuietly(pending.commandBufferHandle);
      uploadArena.retireSubmission(pending.submissionId);
    }
  }

  private void waitFenceOrThrow(long fenceHandle, long timeoutNanos) throws GpuException {
    try (MemoryStack stack = MemoryStack.stackPush()) {
      if (DEBUG_UPLOAD) {
        int status = fenceStatus(fenceHandle);
        System.out.println("[VulkanGpuUploadExecutor] fenceStatusBeforeWait=" + status + " fence=" + fenceHandle);
      }
      int waitResult =
          vkWaitForFences(device, stack.longs(fenceHandle), true, Math.max(1L, timeoutNanos));
      if (waitResult != VK_SUCCESS) {
        throw vkFailure.apply("vkWaitForFences(upload-batch)", waitResult);
      }
      if (DEBUG_UPLOAD) {
        int status = fenceStatus(fenceHandle);
        System.out.println("[VulkanGpuUploadExecutor] fenceStatusAfterWait=" + status + " fence=" + fenceHandle);
      }
    }
  }

  private int fenceStatus(long fenceHandle) {
    return vkGetFenceStatus(device, fenceHandle);
  }

  private void destroyFenceQuietly(long fenceHandle) {
    if (fenceHandle != VK_NULL_HANDLE) {
      vkDestroyFence(device, fenceHandle, null);
    }
  }

  private void freeCommandBufferQuietly(long commandBufferHandle) {
    if (commandBufferHandle == VK_NULL_HANDLE) {
      return;
    }
    try (MemoryStack stack = MemoryStack.stackPush()) {
      vkFreeCommandBuffers(device, commandPool, stack.pointers(commandBufferHandle));
    }
  }

  private GpuMeshResource uploadSimple(GpuGeometryUploadPlan plan) throws GpuException {
    Objects.requireNonNull(plan, "plan");

    VulkanGpuBuffer vertexBuffer = null;
    VulkanGpuBuffer indexBuffer = null;
    try (MemoryStack stack = MemoryStack.stackPush()) {
      ByteBuffer vertexBytes = toReusableDirectCopy(plan.vertexData());
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
        ByteBuffer indexBytes = toReusableDirectCopy(plan.indexData());
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

  private PreparedPlan preparePlanForBatch(MemoryStack stack, PlanSizing sizing)
      throws GpuException {
    GpuGeometryUploadPlan plan = sizing.plan;
    ByteBuffer vertexBytes = toReusableDirectCopy(plan.vertexData());
    int vertexUsage = toVkBufferUsage(GpuBufferUsage.VERTEX) | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VulkanDeviceLocalBufferPool.Lease vertexLease =
        deviceBufferPool.acquire(stack, sizing.vertexBytes, vertexUsage);
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
    if (sizing.hasIndexData) {
      ByteBuffer indexBytes = toReusableDirectCopy(plan.indexData());
      int indexUsage = toVkBufferUsage(GpuBufferUsage.INDEX) | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      VulkanDeviceLocalBufferPool.Lease indexLease =
          deviceBufferPool.acquire(stack, sizing.indexBytes, indexUsage);
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
      indexCopy =
          new CopyOp(
              indexSlice.stagingBuffer(),
              indexSlice.offsetBytes(),
              indexLease.bufferHandle(),
              0,
              indexSlice.sizeBytes());
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

  private SubmissionHandles submitCopies(
      MemoryStack stack, List<PreparedPlan> preparedPlans, boolean waitForCompletion)
      throws GpuException {
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
    long commandBufferHandle = pCommandBuffer.get(0);
    VkCommandBuffer commandBuffer = new VkCommandBuffer(commandBufferHandle, device);

    VkCommandBufferBeginInfo beginInfo =
        VkCommandBufferBeginInfo.calloc(stack)
            .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
            .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    int beginResult = vkBeginCommandBuffer(commandBuffer, beginInfo);
    if (beginResult != VK_SUCCESS) {
      freeCommandBufferQuietly(commandBufferHandle);
      throw vkFailure.apply("vkBeginCommandBuffer(upload-batch)", beginResult);
    }

    VkBufferCopy.Buffer copyRegion = VkBufferCopy.calloc(1, stack);
    for (PreparedPlan preparedPlan : preparedPlans) {
      recordCopy(commandBuffer, copyRegion, preparedPlan.vertexCopy);
      if (preparedPlan.indexCopy != null) {
        recordCopy(commandBuffer, copyRegion, preparedPlan.indexCopy);
      }
    }
    int endResult = vkEndCommandBuffer(commandBuffer);
    if (endResult != VK_SUCCESS) {
      freeCommandBufferQuietly(commandBufferHandle);
      throw vkFailure.apply("vkEndCommandBuffer(upload-batch)", endResult);
    }

    long fenceHandle = createFence(stack, false);
    VkSubmitInfo submitInfo =
        VkSubmitInfo.calloc(stack)
            .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
            .pCommandBuffers(stack.pointers(commandBuffer.address()));
    int submitResult = vkQueueSubmit(graphicsQueue, submitInfo, fenceHandle);
    if (submitResult != VK_SUCCESS) {
      destroyFenceQuietly(fenceHandle);
      freeCommandBufferQuietly(commandBufferHandle);
      throw vkFailure.apply("vkQueueSubmit(upload-batch)", submitResult);
    }

    if (waitForCompletion) {
      waitFenceOrThrow(fenceHandle, DEFAULT_FENCE_WAIT_NANOS);
      destroyFenceQuietly(fenceHandle);
      freeCommandBufferQuietly(commandBufferHandle);
      return new SubmissionHandles(VK_NULL_HANDLE, VK_NULL_HANDLE);
    }
    return new SubmissionHandles(commandBufferHandle, fenceHandle);
  }

  private long createFence(MemoryStack stack, boolean signaled) throws GpuException {
    VkFenceCreateInfo fenceInfo =
        VkFenceCreateInfo.calloc(stack)
            .sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
            .flags(signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0);
    var pFence = stack.longs(VK_NULL_HANDLE);
    int createResult = vkCreateFence(device, fenceInfo, null, pFence);
    if (createResult != VK_SUCCESS || pFence.get(0) == VK_NULL_HANDLE) {
      throw vkFailure.apply("vkCreateFence(upload-batch)", createResult);
    }
    return pFence.get(0);
  }

  private static void recordCopy(
      VkCommandBuffer commandBuffer, VkBufferCopy.Buffer copyRegion, CopyOp copy) {
    copyRegion
        .get(0)
        .srcOffset(copy.srcOffsetBytes)
        .dstOffset(copy.dstOffsetBytes)
        .size(copy.sizeBytes);
    vkCmdCopyBuffer(commandBuffer, copy.srcBuffer, copy.dstBuffer, copyRegion);
  }

  private static void validateCopyOps(
      List<PreparedPlan> preparedPlans,
      long submissionId,
      Map<Long, List<Range>> srcRangesByBuffer,
      Map<Long, List<Range>> dstRangesByBuffer) {
    clearRanges(srcRangesByBuffer);
    clearRanges(dstRangesByBuffer);
    for (PreparedPlan prepared : preparedPlans) {
      validateCopy(prepared.vertexCopy, srcRangesByBuffer, dstRangesByBuffer, submissionId);
      if (prepared.indexCopy != null) {
        validateCopy(prepared.indexCopy, srcRangesByBuffer, dstRangesByBuffer, submissionId);
      }
    }
  }

  private static void clearRanges(Map<Long, List<Range>> rangesByBuffer) {
    for (List<Range> ranges : rangesByBuffer.values()) {
      ranges.clear();
    }
    rangesByBuffer.clear();
  }

  private static void validateCopy(
      CopyOp copy,
      Map<Long, List<Range>> srcRangesByBuffer,
      Map<Long, List<Range>> dstRangesByBuffer,
      long submissionId) {
    if (copy.sizeBytes <= 0L) {
      throw new IllegalStateException("Invalid copy sizeBytes=" + copy.sizeBytes + " submissionId=" + submissionId);
    }
    if (copy.srcBuffer == copy.dstBuffer) {
      throw new IllegalStateException(
          "Source and destination buffers are equal in copy submissionId=" + submissionId + " buffer=" + copy.srcBuffer);
    }
    Range srcRange = new Range(copy.srcOffsetBytes, copy.srcOffsetBytes + copy.sizeBytes);
    Range dstRange = new Range(copy.dstOffsetBytes, copy.dstOffsetBytes + copy.sizeBytes);
    ensureNoOverlap(srcRangesByBuffer.computeIfAbsent(copy.srcBuffer, unused -> new ArrayList<>()), srcRange, "src", submissionId, copy.srcBuffer);
    ensureNoOverlap(dstRangesByBuffer.computeIfAbsent(copy.dstBuffer, unused -> new ArrayList<>()), dstRange, "dst", submissionId, copy.dstBuffer);
    srcRangesByBuffer.get(copy.srcBuffer).add(srcRange);
    dstRangesByBuffer.get(copy.dstBuffer).add(dstRange);
    if (DEBUG_UPLOAD) {
      System.out.println(
          "[VulkanGpuUploadExecutor] copy submissionId="
              + submissionId
              + " srcBuffer="
              + copy.srcBuffer
              + " dstBuffer="
              + copy.dstBuffer
              + " srcOffset="
              + copy.srcOffsetBytes
              + " dstOffset="
              + copy.dstOffsetBytes
              + " size="
              + copy.sizeBytes);
    }
  }

  private static void ensureNoOverlap(
      List<Range> existing, Range candidate, String kind, long submissionId, long bufferHandle) {
    for (Range range : existing) {
      if (candidate.overlaps(range)) {
        throw new IllegalStateException(
            "Detected "
                + kind
                + " overlap for submissionId="
                + submissionId
                + " buffer="
                + bufferHandle
                + " existing="
                + range
                + " candidate="
                + candidate);
      }
    }
  }

  static int toVkBufferUsage(GpuBufferUsage usage) {
    return switch (usage) {
      case VERTEX -> VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
      case INDEX -> VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
      case STORAGE -> VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
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

  private ByteBuffer toReusableDirectCopy(ByteBuffer source) {
    ByteBuffer duplicate = source.duplicate();
    int requiredBytes = duplicate.remaining();
    ensureReusableDirectCopyCapacity(requiredBytes);
    ByteBuffer reusable = reusableDirectCopyBuffer;
    reusable.clear();
    reusable.limit(requiredBytes);
    reusable.put(duplicate);
    reusable.flip();
    return reusable;
  }

  private void ensureReusableDirectCopyCapacity(int requiredBytes) {
    if (requiredBytes < 0) {
      throw new IllegalArgumentException("requiredBytes must be >= 0");
    }
    if (reusableDirectCopyBuffer == null || reusableDirectCopyBuffer.capacity() < requiredBytes) {
      reusableDirectCopyBuffer = ByteBuffer.allocateDirect(requiredBytes);
    }
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

  public record DeferredUploadBatch(long submissionId, int planCount, long totalBytes) {}

  private record PreparedPlan(
      GpuGeometryUploadPlan plan,
      VulkanGpuBuffer vertexBuffer,
      VulkanGpuBuffer indexBuffer,
      CopyOp vertexCopy,
      CopyOp indexCopy) {}

  private record CopyOp(
      long srcBuffer, long srcOffsetBytes, long dstBuffer, long dstOffsetBytes, long sizeBytes) {}

  private record PendingSubmission(
      long submissionId,
      long commandBufferHandle,
      long fenceHandle,
      List<PreparedPlan> preparedPlans,
      long totalBytes) {
    int planCount() {
      return preparedPlans.size();
    }
  }

  private record SubmissionHandles(long commandBufferHandle, long fenceHandle) {}

  private record Range(long startInclusive, long endExclusive) {
    private boolean overlaps(Range other) {
      return this.startInclusive < other.endExclusive && other.startInclusive < this.endExclusive;
    }
  }

  private record PlanSizing(
      GpuGeometryUploadPlan plan,
      int vertexBytes,
      int indexBytes,
      boolean hasIndexData,
      long estimatedStagingBytes) {
    private static PlanSizing from(GpuGeometryUploadPlan plan) {
      int vertexBytes = plan.vertexData().remaining();
      ByteBuffer indexData = plan.indexData();
      boolean hasIndexData = indexData != null;
      int indexBytes = hasIndexData ? indexData.remaining() : 0;
      long chunkPadding = hasIndexData ? 32L : 16L;
      return new PlanSizing(plan, vertexBytes, indexBytes, hasIndexData, vertexBytes + indexBytes + chunkPadding);
    }

    private long totalBytes() {
      return (long) vertexBytes + indexBytes;
    }
  }
}
