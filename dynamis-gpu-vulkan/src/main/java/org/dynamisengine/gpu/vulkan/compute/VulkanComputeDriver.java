package org.dynamisengine.gpu.vulkan.compute;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.spi.ComputeDriver;
import org.dynamisengine.gpu.api.compute.spi.MemoryPool;
import org.lwjgl.system.*;
import org.lwjgl.vulkan.*;
import java.nio.*;
import java.util.*;
import static org.lwjgl.vulkan.VK10.*;
import static org.dynamisengine.gpu.vulkan.compute.VulkanProvider.check;

/** Submission-scoped headless compute. A single queue family avoids ownership transfers. */
final class VulkanComputeDriver implements ComputeDriver {
    /** A pool block: one VkBuffer over its own memory, persistently mapped when host-visible (mapped != 0). */
    private record Block(long buffer, long memory, long bytes, boolean coherent, long mapped) { }
    /** A buffer: a range of a block; {@code bytes} is the size asked for. */
    private record Allocation(MemoryPool.Allocation<Block> range, long bytes) {
        long buffer() { return range.block().buffer(); }
        long base() { return range.offset(); }
    }
    /** Blocks of 64 MiB; buffers over 16 MiB get their own. */
    static final long BLOCK_BYTES = 64L << 20, DEDICATED_BYTES = 16L << 20;
    private final Map<Placement, MemoryPool<Block>> pools = new EnumMap<>(Placement.class);
    private record Pipeline(long pipeline, long layout, long setLayout) { }
    private final VkInstance instance;
    private final VkPhysicalDevice physical;
    private final VkDevice device;
    private final VkQueue queue;
    private final long pool;
    private final VulkanFeatures features;
    /** The device's VkPipelineCache, and the file it is loaded from and saved to (null: in memory only). */
    private final long pipelineCache;
    private final java.nio.file.Path pipelineCacheFile;
    /** System property naming a directory where pipeline caches persist across runs (opt-in). */
    static final String PIPELINE_CACHE_PROPERTY = "dynamisgpu.pipelineCache";
    /** Whether VK_EXT_memory_budget is enabled (budget and usage per heap). */
    private final boolean memoryBudget;
    /** Every pooled range starts at a multiple of this: at least 256 and the device's buffer offset alignments. */
    private final long alignment;
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
            long align = Math.max(256, Math.max(limits.limits().minStorageBufferOffsetAlignment(), limits.limits().minUniformBufferOffsetAlignment()));
            alignment = Long.highestOneBit(align) == align ? align : Long.highestOneBit(align) << 1;
            timestampBits = timestampPeriod > 0 ? VulkanProvider.timestampBits(physical, family) : 0;
            var info = VkDeviceCreateInfo.calloc(stack).sType$Default().pQueueCreateInfos(qi).pNext(features.enableChain(stack));
            List<String> extensions = new ArrayList<>(features.extensions());
            memoryBudget = VulkanProvider.extension(physical, "VK_EXT_memory_budget");
            if (memoryBudget) extensions.add("VK_EXT_memory_budget");
            if (VulkanProvider.extension(physical, "VK_KHR_portability_subset")) extensions.add("VK_KHR_portability_subset");
            if (!extensions.isEmpty()) {
                var names = stack.mallocPointer(extensions.size()); for (var name : extensions) names.put(stack.UTF8(name)); names.flip();
                info.ppEnabledExtensionNames(names);
            }
            var out = stack.mallocPointer(1); check(vkCreateDevice(physical, info, null, out), "create compute device");
            device = created = new VkDevice(out.get(0), physical, info);
            vkGetDeviceQueue(device, family, 0, out); queue = new VkQueue(out.get(0), device);
            pipelineCacheFile = cacheFile(physical);
            pipelineCache = createPipelineCache(stack, pipelineCacheFile);
            var poolInfo = VkCommandPoolCreateInfo.calloc(stack).sType$Default().flags(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT).queueFamilyIndex(family);
            var handle = stack.mallocLong(1); check(vkCreateCommandPool(device, poolInfo, null, handle), "create compute command pool"); pool = createdPool = handle.get(0);
        } catch (RuntimeException | Error ex) {
            if (createdPool != 0) vkDestroyCommandPool(created, createdPool, null);
            if (created != null) vkDestroyDevice(created, null); vkDestroyInstance(instance, null); throw ex;
        }
    }
    /** The cache file for this device and driver build, or null when caching to disk is off. */
    private static java.nio.file.Path cacheFile(VkPhysicalDevice physical) {
        String dir = System.getProperty(PIPELINE_CACHE_PROPERTY);
        if (dir == null || dir.isBlank()) return null;
        try (var stack = MemoryStack.stackPush()) {
            var p = VkPhysicalDeviceProperties.calloc(stack); vkGetPhysicalDeviceProperties(physical, p);
            var uuid = new StringBuilder(); var bytes = p.pipelineCacheUUID();
            for (int i = 0; i < bytes.remaining(); i++) uuid.append(String.format("%02x", bytes.get(i)));
            return java.nio.file.Path.of(dir, "vk-" + Integer.toHexString(p.vendorID()) + "-" + Integer.toHexString(p.deviceID()) + "-"
                    + Integer.toHexString(p.driverVersion()) + "-" + uuid + ".bin");
        }
    }
    /** A pipeline cache seeded from {@code file} when it exists (drivers ignore data from another device or build). */
    private long createPipelineCache(MemoryStack stack, java.nio.file.Path file) {
        var info = VkPipelineCacheCreateInfo.calloc(stack).sType$Default();
        java.nio.ByteBuffer initial = null;
        try {
            if (file != null && java.nio.file.Files.isRegularFile(file)) {
                byte[] data = java.nio.file.Files.readAllBytes(file);
                initial = MemoryUtil.memAlloc(data.length).put(data).flip(); info.pInitialData(initial);
            }
            var out = stack.mallocLong(1);
            int r = vkCreatePipelineCache(device, info, null, out);
            if (r != VK_SUCCESS && initial != null) { info.pInitialData(null); r = vkCreatePipelineCache(device, info, null, out); } // unusable data
            check(r, "create pipeline cache"); return out.get(0);
        } catch (java.io.IOException e) {
            var out = stack.mallocLong(1); info.pInitialData(null); check(vkCreatePipelineCache(device, info, null, out), "create pipeline cache"); return out.get(0);
        } finally { if (initial != null) MemoryUtil.memFree(initial); }
    }
    /** Writes the pipeline cache to its file (atomically; failures only lose the cache). */
    private void savePipelineCache() {
        if (pipelineCacheFile == null) return;
        try (var stack = MemoryStack.stackPush()) {
            var size = stack.mallocPointer(1);
            if (vkGetPipelineCacheData(device, pipelineCache, size, null) != VK_SUCCESS || size.get(0) == 0) return;
            var data = MemoryUtil.memAlloc((int) size.get(0));
            try {
                if (vkGetPipelineCacheData(device, pipelineCache, size, data) != VK_SUCCESS) return;
                byte[] bytes = new byte[(int) size.get(0)]; data.get(0, bytes);
                java.nio.file.Files.createDirectories(pipelineCacheFile.getParent());
                var tmp = java.nio.file.Files.createTempFile(pipelineCacheFile.getParent(), "vk-", ".tmp");
                java.nio.file.Files.write(tmp, bytes);
                java.nio.file.Files.move(tmp, pipelineCacheFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING, java.nio.file.StandardCopyOption.ATOMIC_MOVE);
            } catch (java.io.IOException e) {
                System.getLogger(VulkanComputeDriver.class.getName()).log(System.Logger.Level.WARNING, "pipeline cache not saved to " + pipelineCacheFile, e);
            } finally { MemoryUtil.memFree(data); }
        }
    }
    @Override public Object allocate(BufferDescriptor d) {
        if (d.usages().stream().noneMatch(u -> u == BufferUsage.STORAGE || u == BufferUsage.UNIFORM || u == BufferUsage.TRANSFER_SOURCE || u == BufferUsage.TRANSFER_DESTINATION))
            throw new ComputeException(ComputeException.Code.INVALID_ARGUMENT, "Vulkan buffers require at least one GPU usage");
        return new Allocation(pool(d.placement()).allocate(d.bytes()), d.bytes());
    }
    private MemoryPool<Block> pool(Placement placement) {
        return pools.computeIfAbsent(placement, p -> new MemoryPool<>(new MemoryPool.Blocks<>() {
            @Override public Block create(long bytes) { return createBlock(p, bytes); }
            @Override public void destroy(Block b) { destroyBlock(b); }
        }, BLOCK_BYTES, alignment, DEDICATED_BYTES));
    }
    /** A block buffer with every usage a buffer may ask for, bound to its own memory of the placement's type. */
    private Block createBlock(Placement placement, long bytes) {
        long buffer = 0, memory = 0;
        try (var stack = MemoryStack.stackPush()) {
            int usages = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            var info = VkBufferCreateInfo.calloc(stack).sType$Default().size(bytes).usage(usages).sharingMode(VK_SHARING_MODE_EXCLUSIVE);
            var out = stack.mallocLong(1); check(vkCreateBuffer(device, info, null, out), "create buffer"); buffer = out.get(0);
            var requirements = VkMemoryRequirements.calloc(stack); vkGetBufferMemoryRequirements(device, buffer, requirements);
            var properties = VkPhysicalDeviceMemoryProperties.calloc(stack); vkGetPhysicalDeviceMemoryProperties(physical, properties);
            int required = placement == Placement.SHARED ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            int selected = -1;
            for (int i = 0; i < properties.memoryTypeCount(); i++) {
                int flags = properties.memoryTypes(i).propertyFlags();
                if ((requirements.memoryTypeBits() & (1 << i)) != 0 && (flags & required) == required) {
                    selected = i; if (placement == Placement.PRIVATE || (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) break;
                }
            }
            if (selected < 0) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "No memory type for Vulkan " + placement);
            int flags = properties.memoryTypes(selected).propertyFlags();
            var allocation = VkMemoryAllocateInfo.calloc(stack).sType$Default().allocationSize(requirements.size()).memoryTypeIndex(selected);
            check(vkAllocateMemory(device, allocation, null, out), "allocate buffer memory"); memory = out.get(0);
            check(vkBindBufferMemory(device, buffer, memory, 0), "bind buffer memory");
            long mapped = 0;
            if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
                var pointer = stack.mallocPointer(1); check(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, pointer), "map block memory"); mapped = pointer.get(0);
            }
            return new Block(buffer, memory, bytes, (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0, mapped);
        } catch (RuntimeException | Error ex) {
            if (buffer != 0) vkDestroyBuffer(device, buffer, null); if (memory != 0) vkFreeMemory(device, memory, null); throw ex;
        }
    }
    /** The device-local heaps' budget and usage (VK_EXT_memory_budget), or their size and unknown usage. */
    private long[] budget() {
        try (var stack = MemoryStack.stackPush()) {
            var properties = VkPhysicalDeviceMemoryProperties2.calloc(stack).sType$Default();
            var budgets = VkPhysicalDeviceMemoryBudgetPropertiesEXT.calloc(stack).sType$Default();
            if (memoryBudget) properties.pNext(budgets.address());
            VK11.vkGetPhysicalDeviceMemoryProperties2(physical, properties);
            var memory = properties.memoryProperties();
            long budget = 0, usage = 0;
            for (int h = 0; h < memory.memoryHeapCount(); h++) {
                if ((memory.memoryHeaps(h).flags() & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
                budget += memoryBudget ? budgets.heapBudget(h) : memory.memoryHeaps(h).size();
                usage += memoryBudget ? budgets.heapUsage(h) : 0;
            }
            return new long[] {budget, memoryBudget ? usage : -1};
        }
    }
    private void destroyBlock(Block b) {
        if (b.mapped() != 0) vkUnmapMemory(device, b.memory());
        vkDestroyBuffer(device, b.buffer(), null); vkFreeMemory(device, b.memory(), null);
    }
    @Override public org.dynamisengine.gpu.api.compute.MemoryStats memoryStats() {
        var total = new org.dynamisengine.gpu.api.compute.MemoryStats(0, 0, 0, 0, 0);
        for (var p : pools.values()) total = total.plus(p.stats());
        long[] budget = budget();
        return total.withBudget(budget[0], budget[1]);
    }
    @Override public void releaseBuffer(Object object) {
        var a = (Allocation) object;
        for (var p : pools.values()) if (p.owns(a.range())) { p.free(a.range()); return; }
        throw new IllegalStateException("buffer from no pool of this device");
    }
    @Override public ByteBuffer map(Object object, long offset, long bytes, Access access) {
        var a = (Allocation) object; var block = a.range().block();
        if (block.mapped() == 0) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "Buffer memory is not host-visible");
        if (!block.coherent() && access != Access.WRITE) {
            try (var stack = MemoryStack.stackPush()) {
                var range = VkMappedMemoryRange.calloc(1, stack); range.get(0).sType$Default().memory(block.memory()).offset(0).size(VK_WHOLE_SIZE);
                check(vkInvalidateMappedMemoryRanges(device, range), "invalidate noncoherent mapped memory");
            }
        }
        return MemoryUtil.memByteBuffer(block.mapped() + a.base() + offset, Math.toIntExact(bytes)).order(ByteOrder.nativeOrder());
    }
    @Override public void unmap(Object object, Access access) {
        var block = ((Allocation) object).range().block();
        if (!block.coherent() && access != Access.READ) {
            try (var stack = MemoryStack.stackPush()) {
                var range = VkMappedMemoryRange.calloc(1, stack); range.get(0).sType$Default().memory(block.memory()).offset(0).size(VK_WHOLE_SIZE);
                check(vkFlushMappedMemoryRanges(device, range), "flush noncoherent mapped memory");
            }
        }
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
            check(vkCreateComputePipelines(device, pipelineCache, info, null, out), "create compute pipeline"); pipeline = out.get(0);
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
    /** A region's offset in its block's VkBuffer. */
    private static long offset(Region r) { return ((Allocation) r.allocation()).base() + r.offset(); }
    /**
     * The barrier between commands: compute and transfer work, earlier writes visible to later reads and writes. Host
     * writes made before submission are visible through vkQueueSubmit, so the host stage only appears in the final
     * barrier ({@code host}), which makes GPU writes visible to the host readbacks.
     */
    private static void barrier(VkCommandBuffer cmd, boolean host, MemoryStack stack) {
        int stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        var memory = VkMemoryBarrier.calloc(1, stack); memory.get(0).sType$Default()
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)
                .dstAccessMask(host ? VK_ACCESS_HOST_READ_BIT
                        : VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdPipelineBarrier(cmd, stages, host ? VK_PIPELINE_STAGE_HOST_BIT : stages, 0, memory, null, null);
    }
    private static void copy(VkCommandBuffer cmd, long src, long srcOffset, long dst, long dstOffset, long bytes, MemoryStack stack) {
        var region = VkBufferCopy.calloc(1, stack); region.get(0).srcOffset(srcOffset).dstOffset(dstOffset).size(bytes); vkCmdCopyBuffer(cmd, src, dst, region);
    }
    /**
     * The GPU objects of one command list: a descriptor set per dispatch, staging for uploads and downloads, a
     * timestamp query pool, a fence and a command buffer. A one-shot submission builds one and frees it on
     * completion; a recording keeps it and resubmits, re-recording only when parameters or bindings change.
     */
    private final class Batch implements Prepared {
        long descriptorPool, fence, queries;
        long[] sets;                       // per command index; 0 for non-dispatches
        Allocation[] stages;               // per command index: upload source or download destination
        final Map<GpuReadback, Allocation> reads = new LinkedHashMap<>();
        VkCommandBuffer cmd;
        List<Command> recorded;            // the commands the command buffer holds
        final boolean profile;             // a timestamp pair around each dispatch, after the submission's pair
        int queryCount;

        Batch(List<Command> commands, boolean profile) {
            this.profile = profile;
            try (var stack = MemoryStack.stackPush()) {
                int n = commands.size(), dispatches = 0, descriptors = 0;
                sets = new long[n]; stages = new Allocation[n];
                for (var command : commands) if (command instanceof Dispatch d) { dispatches++; descriptors += d.bindings().size(); }
                var out = stack.mallocLong(1);
                if (dispatches > 0) {
                    var sizes = VkDescriptorPoolSize.calloc(1, stack); sizes.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(Math.max(1, descriptors));
                    var info = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().maxSets(dispatches).pPoolSizes(sizes);
                    check(vkCreateDescriptorPool(device, info, null, out), "create submission descriptor pool"); descriptorPool = out.get(0);
                }
                for (int i = 0; i < n; i++) {
                    switch (commands.get(i)) {
                        case Dispatch d -> {
                            var setInfo = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default().descriptorPool(descriptorPool).pSetLayouts(stack.longs(((Pipeline) d.pipeline()).setLayout()));
                            check(vkAllocateDescriptorSets(device, setInfo, out), "allocate compute descriptor set"); sets[i] = out.get(0);
                            write(sets[i], d);
                        }
                        case Upload u -> {
                            var stage = (Allocation) allocate(new BufferDescriptor(u.data().length, Set.of(BufferUsage.TRANSFER_SOURCE), Placement.SHARED)); stages[i] = stage;
                            map(stage, 0, stage.bytes(), Access.WRITE).put(u.data()); unmap(stage, Access.WRITE);
                        }
                        case Download d -> {
                            var stage = (Allocation) allocate(new BufferDescriptor(d.source().bytes(), Set.of(BufferUsage.TRANSFER_DESTINATION), Placement.SHARED));
                            stages[i] = stage; reads.put(d.result(), stage);
                        }
                        default -> { }
                    }
                }
                if (timestampBits > 0) {
                    queryCount = 2 + (profile ? 2 * dispatches : 0);
                    var info = VkQueryPoolCreateInfo.calloc(stack).sType$Default().queryType(VK_QUERY_TYPE_TIMESTAMP).queryCount(queryCount);
                    check(vkCreateQueryPool(device, info, null, out), "create timestamp query pool"); queries = out.get(0);
                }
                check(vkCreateFence(device, VkFenceCreateInfo.calloc(stack).sType$Default(), null, out), "create submission fence"); fence = out.get(0);
                var allocation = VkCommandBufferAllocateInfo.calloc(stack).sType$Default().commandPool(pool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1);
                var pointer = stack.mallocPointer(1); check(vkAllocateCommandBuffers(device, allocation, pointer), "allocate submission command buffer");
                cmd = new VkCommandBuffer(pointer.get(0), device);
                record(commands);
            } catch (RuntimeException | Error ex) { close(); throw ex; }
        }

        private void write(long set, Dispatch d) {
            try (var stack = MemoryStack.stackPush()) {
                var writes = VkWriteDescriptorSet.calloc(d.bindings().size(), stack);
                for (int i = 0; i < d.bindings().size(); i++) {
                    var bound = d.bindings().get(i); var info = VkDescriptorBufferInfo.calloc(1, stack); info.get(0).buffer(buffer(bound.region())).offset(offset(bound.region())).range(bound.region().bytes());
                    writes.get(i).sType$Default().dstSet(set).dstBinding(bound.slot()).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1).pBufferInfo(info);
                }
                vkUpdateDescriptorSets(device, writes, null);
            }
        }

        /** (Re-)records the command buffer for {@code commands}, which match the batch's command list in shape. */
        void record(List<Command> commands) {
            try (var stack = MemoryStack.stackPush()) {
                if (recorded != null) check(vkResetCommandBuffer(cmd, 0), "reset compute commands");
                check(vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo.calloc(stack).sType$Default()), "begin compute commands");
                if (queries != 0) { vkCmdResetQueryPool(cmd, queries, 0, queryCount); vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0); }
                int dispatchNumber = 0;
                for (int i = 0; i < commands.size(); i++) {
                    try (var local = MemoryStack.stackPush()) {
                        barrier(cmd, false, local);
                        switch (commands.get(i)) {
                            case Fill f -> vkCmdFillBuffer(cmd, buffer(f.destination()), offset(f.destination()), f.destination().bytes(), f.pattern());
                            case Copy c -> copy(cmd, buffer(c.source()), offset(c.source()), buffer(c.destination()), offset(c.destination()), c.source().bytes(), local);
                            case Upload u -> copy(cmd, stages[i].buffer(), stages[i].base(), buffer(u.destination()), offset(u.destination()), u.destination().bytes(), local);
                            case Download d -> copy(cmd, buffer(d.source()), offset(d.source()), stages[i].buffer(), stages[i].base(), stages[i].bytes(), local);
                            case Dispatch d -> {
                                var p = (Pipeline) d.pipeline();
                                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline());
                                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout(), 0, local.longs(sets[i]), null);
                                if (d.parameters().length > 0) vkCmdPushConstants(cmd, p.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, local.bytes(d.parameters()));
                                boolean timed = profile && queries != 0;
                                if (timed) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 2 + 2 * dispatchNumber);
                                vkCmdDispatch(cmd, d.groups().x(), d.groups().y(), d.groups().z());
                                if (timed) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 3 + 2 * dispatchNumber);
                                dispatchNumber++;
                            }
                        }
                    }
                }
                if (queries != 0) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
                barrier(cmd, true, stack); check(vkEndCommandBuffer(cmd), "end compute commands");
                recorded = List.copyOf(commands);
            }
        }

        /** Brings the batch up to date with {@code commands} (new parameters or bindings) and submits it. */
        Job submit(List<Command> commands, boolean oneShot) {
            if (recorded != null && !commands.equals(recorded)) {
                boolean rerecord = false;
                for (int i = 0; i < commands.size(); i++) {
                    if (commands.get(i) instanceof Dispatch now && recorded.get(i) instanceof Dispatch before) {
                        // updating a descriptor set invalidates command buffers that bound it, so rebinding re-records too
                        if (!now.bindings().equals(before.bindings())) { write(sets[i], now); rerecord = true; }
                        if (!java.util.Arrays.equals(now.parameters(), before.parameters()) || now.pipeline() != before.pipeline()
                                || !now.groups().equals(before.groups())) rerecord = true;
                    } else if (!commands.get(i).equals(recorded.get(i))) rerecord = true;
                }
                if (rerecord) record(commands); else recorded = List.copyOf(commands);
            }
            try (var stack = MemoryStack.stackPush()) {
                check(vkResetFences(device, fence), "reset submission fence");
                var submit = VkSubmitInfo.calloc(stack).sType$Default().pCommandBuffers(stack.pointers(cmd.address()));
                check(vkQueueSubmit(queue, submit, fence), "submit compute commands");
            }
            return new Job() {
                private OptionalLong gpuNanos = OptionalLong.empty();
                private long[] dispatchNanos;
                @Override public OptionalLong gpuNanos() { return gpuNanos; }
                @Override public long[] dispatchNanos() { return dispatchNanos; }
                @Override public boolean await(long nanos) {
                    int r = vkWaitForFences(device, new long[] {fence}, true, Math.max(0, nanos));
                    if (r == VK_TIMEOUT) return false; check(r, "wait for submission fence"); return true;
                }
                @Override public GpuSubmission.Status poll() {
                    int status = vkGetFenceStatus(device, fence); if (status == VK_NOT_READY) return GpuSubmission.Status.PENDING; check(status, "submission fence status");
                    if (queries != 0) {
                        try (var stack = MemoryStack.stackPush()) {
                            var ticks = stack.mallocLong(queryCount);
                            if (vkGetQueryPoolResults(device, queries, 0, queryCount, ticks, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                                long mask = timestampBits >= 64 ? -1L : (1L << timestampBits) - 1;
                                gpuNanos = OptionalLong.of(Math.round(((ticks.get(1) - ticks.get(0)) & mask) * timestampPeriod));
                                if (profile) {
                                    dispatchNanos = new long[(queryCount - 2) / 2];
                                    for (int k = 0; k < dispatchNanos.length; k++)
                                        dispatchNanos[k] = Math.round(((ticks.get(3 + 2 * k) - ticks.get(2 + 2 * k)) & mask) * timestampPeriod);
                                }
                            }
                        }
                    }
                    reads.forEach((read, stage) -> {
                        byte[] result = new byte[Math.toIntExact(stage.bytes())]; map(stage, 0, stage.bytes(), Access.READ).get(result); unmap(stage, Access.READ); read.complete(result);
                    });
                    return GpuSubmission.Status.COMPLETE;
                }
                @Override public void close() { if (oneShot) Batch.this.close(); }
            };
        }

        @Override public void close() {
            if (fence != 0) { vkDestroyFence(device, fence, null); fence = 0; }
            if (queries != 0) { vkDestroyQueryPool(device, queries, null); queries = 0; }
            if (cmd != null) { vkFreeCommandBuffers(device, pool, cmd); cmd = null; }
            if (descriptorPool != 0) { vkDestroyDescriptorPool(device, descriptorPool, null); descriptorPool = 0; }
            if (stages != null) { for (var st : stages) if (st != null) releaseBuffer(st); stages = null; }
        }
    }

    @Override public Job submit(List<Command> commands) { return submit(commands, false); }
    @Override public Job submit(List<Command> commands, boolean profile) {
        var batch = new Batch(commands, profile);
        try { return batch.submit(commands, true); } catch (RuntimeException | Error ex) { batch.close(); throw ex; }
    }
    @Override public Prepared prepare(List<Command> commands) { return prepare(commands, false); }
    @Override public Prepared prepare(List<Command> commands, boolean profile) { return new Batch(commands, profile); }
    @Override public Job submit(Prepared prepared, List<Command> commands) { return ((Batch) prepared).submit(commands, false); }
    @Override public void close() {
        pools.values().forEach(MemoryPool::destroyAll);
        savePipelineCache(); vkDestroyPipelineCache(device, pipelineCache, null);
        vkDestroyCommandPool(device, pool, null); vkDestroyDevice(device, null); vkDestroyInstance(instance, null);
    }
}
