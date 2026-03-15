package org.dynamisengine.gpu.vulkan.descriptor;

import org.dynamisengine.gpu.api.BindlessHeapStats;
import org.dynamisengine.gpu.api.error.GpuException;
import org.lwjgl.vulkan.*;

import java.util.ArrayDeque;
import java.util.logging.Logger;

import static org.lwjgl.vulkan.VK10.*;

/**
 * Vulkan descriptor-indexing heap that allocates typed handles and manages deferred retirement.
 */
public final class VulkanBindlessDescriptorHeap {
    public static final int JOINT_CAPACITY = 8192;
    public static final int MORPH_DELTA_CAPACITY = 4096;
    public static final int MORPH_WEIGHT_CAPACITY = 4096;
    public static final int INSTANCE_CAPACITY = 4096;
    public static final int DRAW_META_CAPACITY = 8192;

    private static final Logger LOG = Logger.getLogger(VulkanBindlessDescriptorHeap.class.getName());

    private static final long SLOT_MASK = 0xFFFF_FFFFL;
    private static final long GEN_MASK = 0x00FF_FFFFL;
    private static final long TYPE_MASK = 0xFFL;

    private final VkDevice device;
    private final boolean active;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long descriptorSet;
    private final int retirementFrames;

    private final TypeState jointState;
    private final TypeState morphDeltaState;
    private final TypeState morphWeightState;
    private final TypeState instanceState;

    private final ArrayDeque<Retirement> retirements = new ArrayDeque<>();
    private long allocationCount;
    private long freesQueuedCount;
    private long freesRetiredCount;
    private long staleHandleRejectCount;
    private int drawMetaCount;
    private int invalidIndexWriteCount;

    private VulkanBindlessDescriptorHeap(
            VkDevice device,
            boolean active,
            long descriptorSetLayout,
            long descriptorPool,
            long descriptorSet,
            int retirementFrames,
            TypeState jointState,
            TypeState morphDeltaState,
            TypeState morphWeightState,
            TypeState instanceState
    ) {
        this.device = device;
        this.active = active;
        this.descriptorSetLayout = descriptorSetLayout;
        this.descriptorPool = descriptorPool;
        this.descriptorSet = descriptorSet;
        this.retirementFrames = Math.max(1, retirementFrames);
        this.jointState = jointState;
        this.morphDeltaState = morphDeltaState;
        this.morphWeightState = morphWeightState;
        this.instanceState = instanceState;
    }

    /**
     * Creates an active bindless descriptor heap when descriptor indexing is supported.
     */
    public static VulkanBindlessDescriptorHeap create(
            VkDevice device,
            VkPhysicalDevice physicalDevice,
            boolean requestedEnabled,
            int framesInFlight
    ) throws GpuException {
        return VulkanBindlessHeapFactory.create(device, physicalDevice, requestedEnabled, framesInFlight);
    }

    /**
     * Returns a disabled no-op heap implementation.
     */
    public static VulkanBindlessDescriptorHeap disabled() {
        return VulkanBindlessHeapFactory.disabled();
    }

    /**
     * Package-private factory used by {@link VulkanBindlessHeapFactory}.
     */
    static VulkanBindlessDescriptorHeap createActive(
            VkDevice device,
            long layout,
            long pool,
            long set,
            int framesInFlight
    ) {
        return new VulkanBindlessDescriptorHeap(
                device,
                true,
                layout,
                pool,
                set,
                Math.max(1, framesInFlight),
                TypeState.create(HeapType.JOINT_PALETTE),
                TypeState.create(HeapType.MORPH_DELTA),
                TypeState.create(HeapType.MORPH_WEIGHT),
                TypeState.create(HeapType.INSTANCE_DATA)
        );
    }

    /**
     * Package-private factory used by {@link VulkanBindlessHeapFactory}.
     */
    static VulkanBindlessDescriptorHeap createDisabled() {
        return new VulkanBindlessDescriptorHeap(
                null,
                false,
                VK_NULL_HANDLE,
                VK_NULL_HANDLE,
                VK_NULL_HANDLE,
                1,
                TypeState.empty(HeapType.JOINT_PALETTE),
                TypeState.empty(HeapType.MORPH_DELTA),
                TypeState.empty(HeapType.MORPH_WEIGHT),
                TypeState.empty(HeapType.INSTANCE_DATA)
        );
    }

    /**
     * @return true when heap is active and descriptor indexing is enabled
     */
    public boolean active() {
        return active;
    }

    /**
     * @return Vulkan descriptor set layout handle
     */
    public long descriptorSetLayout() {
        return descriptorSetLayout;
    }

    /**
     * @return Vulkan descriptor pool handle
     */
    public long descriptorPool() {
        return descriptorPool;
    }

    /**
     * @return Vulkan descriptor set handle
     */
    public long descriptorSet() {
        return descriptorSet;
    }

    /**
     * @return the Vulkan device associated with this heap
     */
    VkDevice device() {
        return device;
    }

    /**
     * Allocates a typed heap handle.
     */
    public synchronized long allocate(HeapType type) {
        TypeState state = state(type);
        if (!active || state == null || state.top <= 0) {
            LOG.warning("BINDLESS_HEAP_CAPACITY_EXHAUSTED type=" + safeTypeName(type));
            return 0L;
        }
        int slot = state.freeStack[--state.top];
        int gen = state.generations[slot];
        if (gen <= 0) {
            gen = 1;
            state.generations[slot] = gen;
        }
        allocationCount++;
        return packHandle(type, gen, slot);
    }

    /**
     * Queues a handle for deferred retirement.
     */
    public synchronized void retire(long handle, long currentFrame) {
        if (!active || handle == 0L) {
            return;
        }
        HeapType type = unpackType(handle);
        TypeState state = state(type);
        int slot = unpackSlot(handle);
        int handleGen = unpackGeneration(handle);
        int currentGen = currentGeneration(state, slot);
        if (state == null || slot < 0 || slot >= state.capacity || handleGen != currentGen) {
            logStale(type, slot, handleGen, currentGen, currentFrame);
            return;
        }
        retirements.addLast(new Retirement(type, slot, handleGen, currentFrame + retirementFrames));
        freesQueuedCount++;
    }

    /**
     * Processes queued retirements eligible at the current frame.
     *
     * @return number of handles retired
     */
    public synchronized int processRetirements(long currentFrame) {
        if (!active || retirements.isEmpty()) {
            return 0;
        }
        int processed = 0;
        int count = retirements.size();
        for (int i = 0; i < count; i++) {
            Retirement retirement = retirements.removeFirst();
            if (retirement.retireFrame > currentFrame) {
                retirements.addLast(retirement);
                continue;
            }
            TypeState state = state(retirement.type);
            if (state == null || retirement.slot < 0 || retirement.slot >= state.capacity) {
                continue;
            }
            int currentGen = state.generations[retirement.slot];
            if (currentGen != retirement.generation) {
                logStale(retirement.type, retirement.slot, retirement.generation, currentGen, currentFrame);
                continue;
            }
            state.generations[retirement.slot] = Math.max(1, currentGen + 1);
            if (state.top < state.freeStack.length) {
                state.freeStack[state.top++] = retirement.slot;
                processed++;
                freesRetiredCount++;
            }
        }
        return processed;
    }

    /**
     * Resolves a heap handle to its backing descriptor slot.
     *
     * @return slot index, or {@code -1} when handle is invalid/stale
     */
    public synchronized int resolveSlot(long handle, long currentFrame) {
        if (!active || handle == 0L) {
            return -1;
        }
        HeapType type = unpackType(handle);
        TypeState state = state(type);
        int slot = unpackSlot(handle);
        int handleGen = unpackGeneration(handle);
        int currentGen = currentGeneration(state, slot);
        if (state == null || slot < 0 || slot >= state.capacity || handleGen != currentGen) {
            logStale(type, slot, handleGen, currentGen, currentFrame);
            return -1;
        }
        return slot;
    }

    /**
     * Updates descriptor payload for a joint palette handle.
     *
     * @return true when update was accepted
     */
    public synchronized boolean updateJointPaletteDescriptor(long handle, long currentFrame, long bufferHandle, long rangeBytes) {
        return VulkanDescriptorUpdater.updateJointPaletteDescriptor(this, handle, currentFrame, bufferHandle, rangeBytes);
    }

    /**
     * Updates descriptor payload for a morph delta handle.
     *
     * @return true when update was accepted
     */
    public synchronized boolean updateMorphDeltaDescriptor(long handle, long currentFrame, long bufferHandle, long rangeBytes) {
        return VulkanDescriptorUpdater.updateMorphDeltaDescriptor(this, handle, currentFrame, bufferHandle, rangeBytes);
    }

    /**
     * Updates descriptor payload for a morph weight handle.
     *
     * @return true when update was accepted
     */
    public synchronized boolean updateMorphWeightDescriptor(long handle, long currentFrame, long bufferHandle, long rangeBytes) {
        return VulkanDescriptorUpdater.updateMorphWeightDescriptor(this, handle, currentFrame, bufferHandle, rangeBytes);
    }

    /**
     * Updates descriptor payload for an instance-data handle.
     *
     * @return true when update was accepted
     */
    public synchronized boolean updateInstanceDataDescriptor(long handle, long currentFrame, long bufferHandle, long rangeBytes) {
        return VulkanDescriptorUpdater.updateInstanceDataDescriptor(this, handle, currentFrame, bufferHandle, rangeBytes);
    }

    /**
     * Updates draw metadata-related counters.
     */
    public synchronized void updateDrawMetaStats(int drawMetaCount, int invalidIndexWrites) {
        this.drawMetaCount = Math.max(0, drawMetaCount);
        this.invalidIndexWriteCount = Math.max(0, invalidIndexWrites);
    }

    /**
     * Returns current heap counters and utilization snapshot.
     */
    public synchronized BindlessHeapStats stats() {
        return new BindlessHeapStats(
                usedCount(jointState),
                JOINT_CAPACITY,
                usedCount(morphDeltaState),
                MORPH_DELTA_CAPACITY,
                usedCount(morphWeightState),
                MORPH_WEIGHT_CAPACITY,
                usedCount(instanceState),
                INSTANCE_CAPACITY,
                allocationCount,
                freesQueuedCount,
                freesRetiredCount,
                staleHandleRejectCount,
                drawMetaCount,
                invalidIndexWriteCount
        );
    }

    /**
     * Destroys Vulkan objects owned by this heap.
     */
    public void destroy(VkDevice device) {
        if (device == null) {
            return;
        }
        if (descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptorPool, null);
        }
        if (descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, null);
        }
    }

    static long packHandle(HeapType type, int generation, int slot) {
        long t = (long) (type.id & TYPE_MASK);
        long g = ((long) generation) & GEN_MASK;
        long s = ((long) slot) & SLOT_MASK;
        return (t << 56) | (g << 32) | s;
    }

    static HeapType unpackType(long handle) {
        int id = (int) ((handle >>> 56) & TYPE_MASK);
        for (HeapType type : HeapType.values()) {
            if (type.id == id) {
                return type;
            }
        }
        return null;
    }

    static int unpackGeneration(long handle) {
        return (int) ((handle >>> 32) & GEN_MASK);
    }

    static int unpackSlot(long handle) {
        return (int) (handle & SLOT_MASK);
    }

    private static String safeTypeName(HeapType type) {
        return type == null ? "UNKNOWN" : type.name();
    }

    private int currentGeneration(TypeState state, int slot) {
        if (state == null || slot < 0 || slot >= state.capacity) {
            return -1;
        }
        return state.generations[slot];
    }

    private TypeState state(HeapType type) {
        if (type == null) {
            return null;
        }
        return switch (type) {
            case JOINT_PALETTE -> jointState;
            case MORPH_DELTA -> morphDeltaState;
            case MORPH_WEIGHT -> morphWeightState;
            case INSTANCE_DATA -> instanceState;
        };
    }

    private void logStale(HeapType type, int slot, int handleGen, int currentGen, long frame) {
        staleHandleRejectCount++;
        LOG.warning("[BINDLESS_HEAP] stale_handle type=" + safeTypeName(type)
                + " slot=" + Integer.toUnsignedString(Math.max(0, slot))
                + " handleGen=" + Integer.toUnsignedString(Math.max(0, handleGen))
                + " currentGen=" + Integer.toUnsignedString(Math.max(0, currentGen))
                + " frame=" + Long.toUnsignedString(Math.max(0L, frame)));
    }

    private static int usedCount(TypeState state) {
        if (state == null || state.capacity <= 0) {
            return 0;
        }
        return Math.max(0, state.capacity - state.top);
    }

    /**
     * Typed handle domains tracked in the bindless heap.
     */
    public enum HeapType {
        JOINT_PALETTE(0, JOINT_CAPACITY),
        MORPH_DELTA(1, MORPH_DELTA_CAPACITY),
        MORPH_WEIGHT(2, MORPH_WEIGHT_CAPACITY),
        INSTANCE_DATA(3, INSTANCE_CAPACITY);

        final int id;
        private final int capacity;

        HeapType(int id, int capacity) {
            this.id = id;
            this.capacity = capacity;
        }
    }

    static final class TypeState {
        final HeapType type;
        final int capacity;
        final int[] generations;
        final int[] freeStack;
        int top;

        private TypeState(HeapType type, int capacity, int[] generations, int[] freeStack, int top) {
            this.type = type;
            this.capacity = capacity;
            this.generations = generations;
            this.freeStack = freeStack;
            this.top = top;
        }

        static TypeState create(HeapType type) {
            int capacity = Math.max(1, type.capacity);
            int[] generations = new int[capacity];
            int[] free = new int[capacity];
            for (int i = 0; i < capacity; i++) {
                generations[i] = 1;
                free[i] = capacity - 1 - i;
            }
            return new TypeState(type, capacity, generations, free, capacity);
        }

        static TypeState empty(HeapType type) {
            return new TypeState(type, 0, new int[0], new int[0], 0);
        }
    }

    private record Retirement(HeapType type, int slot, int generation, long retireFrame) {
    }
}
