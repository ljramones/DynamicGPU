package org.dynamisengine.gpu.vulkan.compute;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.spi.ComputeDriver;
import org.lwjgl.system.*;
import org.lwjgl.vulkan.*;
import java.nio.*;
import java.util.*;
import static org.lwjgl.vulkan.VK10.*;
import static org.dynamisengine.gpu.vulkan.compute.VulkanProvider.check;

/** Submission-scoped headless compute. A single queue family avoids ownership transfers. */
final class VulkanComputeDriver implements ComputeDriver {
    private record Allocation(long buffer, long memory, long bytes, long allocationBytes, boolean coherent) { }
    private record Pipeline(long pipeline, long layout, long setLayout) { }
    private final VkInstance instance;
    private final VkPhysicalDevice physical;
    private final VkDevice device;
    private final VkQueue queue;
    private final long pool;
    private final VulkanFeatures features;
    /** Valid bits of the compute queue's timestamps (0: none) and nanoseconds per tick. */
    private final int timestampBits;
    private final double timestampPeriod;

    VulkanComputeDriver(int index) {
        instance = VulkanProvider.instance(); VkDevice created = null; long createdPool = 0;
        try (var stack = MemoryStack.stackPush()) {
            var devices = VulkanProvider.physicals(instance);
            if (index < 0 || index >= devices.size()) throw new ComputeException(ComputeException.Code.UNAVAILABLE, "Selected Vulkan adapter disappeared");
            physical = devices.get(index); int family = VulkanProvider.family(physical);
            if (family < 0) throw new ComputeException(ComputeException.Code.UNAVAILABLE, "No compute-capable Vulkan queue");
            var qi = VkDeviceQueueCreateInfo.calloc(1, stack); qi.get(0).sType$Default().queueFamilyIndex(family).pQueuePriorities(stack.floats(1));
            features = VulkanFeatures.query(physical);
            var limits = VkPhysicalDeviceProperties.calloc(stack); vkGetPhysicalDeviceProperties(physical, limits);
            timestampPeriod = limits.limits().timestampPeriod();
            timestampBits = timestampPeriod > 0 ? VulkanProvider.timestampBits(physical, family) : 0;
            var info = VkDeviceCreateInfo.calloc(stack).sType$Default().pQueueCreateInfos(qi).pNext(features.enableChain(stack));
            List<String> extensions = new ArrayList<>(features.extensions());
            if (VulkanProvider.extension(physical, "VK_KHR_portability_subset")) extensions.add("VK_KHR_portability_subset");
            if (!extensions.isEmpty()) {
                var names = stack.mallocPointer(extensions.size()); for (var name : extensions) names.put(stack.UTF8(name)); names.flip();
                info.ppEnabledExtensionNames(names);
            }
            var out = stack.mallocPointer(1); check(vkCreateDevice(physical, info, null, out), "create compute device");
            device = created = new VkDevice(out.get(0), physical, info);
            vkGetDeviceQueue(device, family, 0, out); queue = new VkQueue(out.get(0), device);
            var poolInfo = VkCommandPoolCreateInfo.calloc(stack).sType$Default().flags(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT).queueFamilyIndex(family);
            var handle = stack.mallocLong(1); check(vkCreateCommandPool(device, poolInfo, null, handle), "create compute command pool"); pool = createdPool = handle.get(0);
        } catch (RuntimeException | Error ex) {
            if (createdPool != 0) vkDestroyCommandPool(created, createdPool, null);
            if (created != null) vkDestroyDevice(created, null); vkDestroyInstance(instance, null); throw ex;
        }
    }
    @Override public Object allocate(BufferDescriptor d) {
        long buffer = 0, memory = 0;
        try (var stack = MemoryStack.stackPush()) {
            int usages = 0;
            if (d.usages().contains(BufferUsage.STORAGE)) usages |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            if (d.usages().contains(BufferUsage.UNIFORM)) usages |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            if (d.usages().contains(BufferUsage.TRANSFER_SOURCE)) usages |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            if (d.usages().contains(BufferUsage.TRANSFER_DESTINATION)) usages |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            if (usages == 0) throw new ComputeException(ComputeException.Code.INVALID_ARGUMENT, "Vulkan buffers require at least one GPU usage");
            var info = VkBufferCreateInfo.calloc(stack).sType$Default().size(d.bytes()).usage(usages).sharingMode(VK_SHARING_MODE_EXCLUSIVE);
            var out = stack.mallocLong(1); check(vkCreateBuffer(device, info, null, out), "create buffer"); buffer = out.get(0);
            var requirements = VkMemoryRequirements.calloc(stack); vkGetBufferMemoryRequirements(device, buffer, requirements);
            var properties = VkPhysicalDeviceMemoryProperties.calloc(stack); vkGetPhysicalDeviceMemoryProperties(physical, properties);
            int required = d.placement() == Placement.SHARED ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            int selected = -1;
            for (int i = 0; i < properties.memoryTypeCount(); i++) {
                int flags = properties.memoryTypes(i).propertyFlags();
                if ((requirements.memoryTypeBits() & (1 << i)) != 0 && (flags & required) == required) {
                    selected = i; if (d.placement() == Placement.PRIVATE || (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) break;
                }
            }
            if (selected < 0) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "No memory type for Vulkan " + d.placement());
            var allocation = VkMemoryAllocateInfo.calloc(stack).sType$Default().allocationSize(requirements.size()).memoryTypeIndex(selected);
            check(vkAllocateMemory(device, allocation, null, out), "allocate buffer memory"); memory = out.get(0);
            check(vkBindBufferMemory(device, buffer, memory, 0), "bind buffer memory");
            return new Allocation(buffer, memory, d.bytes(), requirements.size(), (properties.memoryTypes(selected).propertyFlags() & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0);
        } catch (RuntimeException | Error ex) {
            if (buffer != 0) vkDestroyBuffer(device, buffer, null); if (memory != 0) vkFreeMemory(device, memory, null); throw ex;
        }
    }
    @Override public void releaseBuffer(Object object) { var a = (Allocation) object; vkDestroyBuffer(device, a.buffer(), null); vkFreeMemory(device, a.memory(), null); }
    @Override public ByteBuffer map(Object object, long offset, long bytes, Access access) {
        var a = (Allocation) object;
        try (var stack = MemoryStack.stackPush()) {
            var pointer = stack.mallocPointer(1); check(vkMapMemory(device, a.memory(), 0, VK_WHOLE_SIZE, 0, pointer), "map buffer memory");
            try {
                if (!a.coherent() && access != Access.WRITE) {
                    var range = VkMappedMemoryRange.calloc(1, stack); range.get(0).sType$Default().memory(a.memory()).offset(0).size(VK_WHOLE_SIZE);
                    check(vkInvalidateMappedMemoryRanges(device, range), "invalidate noncoherent mapped memory");
                }
                return MemoryUtil.memByteBuffer(pointer.get(0) + offset, Math.toIntExact(bytes)).order(ByteOrder.nativeOrder());
            } catch (RuntimeException ex) { vkUnmapMemory(device, a.memory()); throw ex; }
        }
    }
    @Override public void unmap(Object object, Access access) {
        var a = (Allocation) object;
        try (var stack = MemoryStack.stackPush()) {
            if (!a.coherent() && access != Access.READ) {
                var range = VkMappedMemoryRange.calloc(1, stack); range.get(0).sType$Default().memory(a.memory()).offset(0).size(VK_WHOLE_SIZE);
                check(vkFlushMappedMemoryRanges(device, range), "flush noncoherent mapped memory");
            }
        } finally { vkUnmapMemory(device, a.memory()); }
    }
    @Override public Object pipeline(PipelineDescriptor d) {
        var executable = d.kernel().executable(Backend.VULKAN);
        if (executable.format() != KernelBundle.Format.SPIRV) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "Vulkan requires SPIR-V");
        byte[] bytes = executable.bytes(); SpirvWorkgroup.validate(bytes, d); features.checkModule(bytes, d.entryPoint());
        long shader = 0, setLayout = 0, layout = 0, pipeline = 0;
        try (var stack = MemoryStack.stackPush()) {
            var code = MemoryUtil.memAlloc(bytes.length);
            try {
                code.put(bytes).flip(); var info = VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(code);
                var out = stack.mallocLong(1); check(vkCreateShaderModule(device, info, null, out), "create compute shader"); shader = out.get(0);
            } finally { MemoryUtil.memFree(code); }
            var bindings = VkDescriptorSetLayoutBinding.calloc(d.layout().buffers().size(), stack);
            for (int i = 0; i < bindings.capacity(); i++) bindings.get(i).binding(d.layout().buffers().get(i).slot()).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            var setInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(bindings);
            var out = stack.mallocLong(1); check(vkCreateDescriptorSetLayout(device, setInfo, null, out), "create compute binding layout"); setLayout = out.get(0);
            var layoutInfo = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default().pSetLayouts(stack.longs(setLayout));
            if (d.layout().parameterBytes() > 0) {
                var push = VkPushConstantRange.calloc(1, stack); push.get(0).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(d.layout().parameterBytes()); layoutInfo.pPushConstantRanges(push);
            }
            check(vkCreatePipelineLayout(device, layoutInfo, null, out), "create compute pipeline layout"); layout = out.get(0);
            var stage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default().stage(VK_SHADER_STAGE_COMPUTE_BIT).module(shader).pName(stack.UTF8(d.entryPoint()));
            if (!d.specialization().isEmpty()) {
                var entries = VkSpecializationMapEntry.calloc(d.specialization().size(), stack); var data = stack.malloc(d.specialization().size() * 4).order(ByteOrder.nativeOrder());
                int i = 0; for (var e : new TreeMap<>(d.specialization()).entrySet()) { entries.get(i).constantID(e.getKey()).offset(i * 4).size(4); data.putInt(e.getValue()); i++; } data.flip();
                stage.pSpecializationInfo(VkSpecializationInfo.calloc(stack).pMapEntries(entries).pData(data));
            }
            var info = VkComputePipelineCreateInfo.calloc(1, stack); info.get(0).sType$Default().stage(stage).layout(layout);
            check(vkCreateComputePipelines(device, VK_NULL_HANDLE, info, null, out), "create compute pipeline"); pipeline = out.get(0);
            return new Pipeline(pipeline, layout, setLayout);
        } catch (RuntimeException | Error ex) {
            if (pipeline != 0) vkDestroyPipeline(device, pipeline, null);
            if (layout != 0) vkDestroyPipelineLayout(device, layout, null);
            if (setLayout != 0) vkDestroyDescriptorSetLayout(device, setLayout, null); throw ex;
        } finally { if (shader != 0) vkDestroyShaderModule(device, shader, null); }
    }
    @Override public void releasePipeline(Object object) {
        var p = (Pipeline) object; vkDestroyPipeline(device, p.pipeline(), null); vkDestroyPipelineLayout(device, p.layout(), null); vkDestroyDescriptorSetLayout(device, p.setLayout(), null);
    }
    private static long buffer(Region r) { return ((Allocation) r.allocation()).buffer(); }
    private static void barrier(VkCommandBuffer cmd, boolean host, MemoryStack stack) {
        var memory = VkMemoryBarrier.calloc(1, stack); memory.get(0).sType$Default().srcAccessMask(VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT)
                .dstAccessMask(host ? VK_ACCESS_HOST_READ_BIT : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                host ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, memory, null, null);
    }
    private static void copy(VkCommandBuffer cmd, long src, long srcOffset, long dst, long dstOffset, long bytes, MemoryStack stack) {
        var region = VkBufferCopy.calloc(1, stack); region.get(0).srcOffset(srcOffset).dstOffset(dstOffset).size(bytes); vkCmdCopyBuffer(cmd, src, dst, region);
    }
    @Override public Job submit(List<Command> commands) {
        VkCommandBuffer cmd = null; long descriptorPool = 0, fence = 0, queries = 0;
        List<Allocation> stages = new ArrayList<>(); Map<GpuReadback, Allocation> reads = new LinkedHashMap<>();
        try (var stack = MemoryStack.stackPush()) {
            int dispatches = 0, descriptors = 0;
            for (var command : commands) if (command instanceof Dispatch d) { dispatches++; descriptors += d.bindings().size(); }
            if (dispatches > 0) {
                var sizes = VkDescriptorPoolSize.calloc(1, stack); sizes.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(Math.max(1, descriptors));
                var info = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().maxSets(dispatches).pPoolSizes(sizes);
                var out = stack.mallocLong(1); check(vkCreateDescriptorPool(device, info, null, out), "create submission descriptor pool"); descriptorPool = out.get(0);
            }
            var allocation = VkCommandBufferAllocateInfo.calloc(stack).sType$Default().commandPool(pool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1);
            var pointer = stack.mallocPointer(1); check(vkAllocateCommandBuffers(device, allocation, pointer), "allocate submission command buffer"); cmd = new VkCommandBuffer(pointer.get(0), device);
            check(vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo.calloc(stack).sType$Default().flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)), "begin compute commands");
            if (timestampBits > 0) {
                var info = VkQueryPoolCreateInfo.calloc(stack).sType$Default().queryType(VK_QUERY_TYPE_TIMESTAMP).queryCount(2);
                var out = stack.mallocLong(1); check(vkCreateQueryPool(device, info, null, out), "create timestamp query pool"); queries = out.get(0);
                vkCmdResetQueryPool(cmd, queries, 0, 2);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
            }
            for (var command : commands) {
                try (var local = MemoryStack.stackPush()) {
                    barrier(cmd, false, local);
                    switch (command) {
                        case Fill f -> vkCmdFillBuffer(cmd, buffer(f.destination()), f.destination().offset(), f.destination().bytes(), f.pattern());
                        case Copy c -> copy(cmd, buffer(c.source()), c.source().offset(), buffer(c.destination()), c.destination().offset(), c.source().bytes(), local);
                        case Upload u -> {
                            var stage = (Allocation) allocate(new BufferDescriptor(u.data().length, Set.of(BufferUsage.TRANSFER_SOURCE), Placement.SHARED)); stages.add(stage);
                            map(stage, 0, stage.bytes(), Access.WRITE).put(u.data()); unmap(stage, Access.WRITE);
                            copy(cmd, stage.buffer(), 0, buffer(u.destination()), u.destination().offset(), u.destination().bytes(), local);
                        }
                        case Download d -> {
                            var stage = (Allocation) allocate(new BufferDescriptor(d.source().bytes(), Set.of(BufferUsage.TRANSFER_DESTINATION), Placement.SHARED)); stages.add(stage); reads.put(d.result(), stage);
                            copy(cmd, buffer(d.source()), d.source().offset(), stage.buffer(), 0, stage.bytes(), local);
                        }
                        case Dispatch d -> {
                            var p = (Pipeline) d.pipeline();
                            var setInfo = VkDescriptorSetAllocateInfo.calloc(local).sType$Default().descriptorPool(descriptorPool).pSetLayouts(local.longs(p.setLayout()));
                            var set = local.mallocLong(1); check(vkAllocateDescriptorSets(device, setInfo, set), "allocate compute descriptor set");
                            var writes = VkWriteDescriptorSet.calloc(d.bindings().size(), local);
                            for (int i = 0; i < d.bindings().size(); i++) {
                                var bound = d.bindings().get(i); var info = VkDescriptorBufferInfo.calloc(1, local); info.get(0).buffer(buffer(bound.region())).offset(bound.region().offset()).range(bound.region().bytes());
                                writes.get(i).sType$Default().dstSet(set.get(0)).dstBinding(bound.slot()).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1).pBufferInfo(info);
                            }
                            vkUpdateDescriptorSets(device, writes, null); vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline());
                            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout(), 0, set, null);
                            if (d.parameters().length > 0) vkCmdPushConstants(cmd, p.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, local.bytes(d.parameters()));
                            vkCmdDispatch(cmd, d.groups().x(), d.groups().y(), d.groups().z());
                        }
                    }
                }
            }
            if (queries != 0) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
            barrier(cmd, true, stack); check(vkEndCommandBuffer(cmd), "end compute commands");
            var out = stack.mallocLong(1); check(vkCreateFence(device, VkFenceCreateInfo.calloc(stack).sType$Default(), null, out), "create submission fence"); fence = out.get(0);
            var submit = VkSubmitInfo.calloc(stack).sType$Default().pCommandBuffers(stack.pointers(cmd.address())); check(vkQueueSubmit(queue, submit, fence), "submit compute commands");
        } catch (RuntimeException | Error ex) { cleanup(cmd, descriptorPool, fence, queries, stages); throw ex; }
        final VkCommandBuffer submitted = cmd; final long submittedPool = descriptorPool, submittedFence = fence, submittedQueries = queries;
        return new Job() {
            private OptionalLong gpuNanos = OptionalLong.empty();
            @Override public OptionalLong gpuNanos() { return gpuNanos; }
            @Override public GpuSubmission.Status poll() {
                int status = vkGetFenceStatus(device, submittedFence); if (status == VK_NOT_READY) return GpuSubmission.Status.PENDING; check(status, "submission fence status");
                if (submittedQueries != 0) {
                    try (var stack = MemoryStack.stackPush()) {
                        var ticks = stack.mallocLong(2);
                        if (vkGetQueryPoolResults(device, submittedQueries, 0, 2, ticks, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                            long mask = timestampBits >= 64 ? -1L : (1L << timestampBits) - 1;
                            gpuNanos = OptionalLong.of(Math.round(((ticks.get(1) - ticks.get(0)) & mask) * timestampPeriod));
                        }
                    }
                }
                reads.forEach((read, stage) -> {
                    byte[] result = new byte[Math.toIntExact(stage.bytes())]; map(stage, 0, stage.bytes(), Access.READ).get(result); unmap(stage, Access.READ); read.complete(result);
                });
                return GpuSubmission.Status.COMPLETE;
            }
            @Override public void close() { cleanup(submitted, submittedPool, submittedFence, submittedQueries, stages); }
        };
    }
    private void cleanup(VkCommandBuffer cmd, long descriptorPool, long fence, long queries, List<Allocation> stages) {
        if (fence != 0) vkDestroyFence(device, fence, null);
        if (queries != 0) vkDestroyQueryPool(device, queries, null);
        if (cmd != null) vkFreeCommandBuffers(device, pool, cmd);
        if (descriptorPool != 0) vkDestroyDescriptorPool(device, descriptorPool, null); stages.forEach(this::releaseBuffer);
    }
    @Override public void close() { vkDestroyCommandPool(device, pool, null); vkDestroyDevice(device, null); vkDestroyInstance(instance, null); }
}
