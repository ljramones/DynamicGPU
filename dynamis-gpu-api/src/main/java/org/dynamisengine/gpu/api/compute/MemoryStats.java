package org.dynamisengine.gpu.api.compute;

/**
 * A device's buffer memory: what its pools hold, how often they have gone to the driver, and the device's own budget.
 *
 * @param reservedBytes bytes of the blocks the device holds now (used or free)
 * @param usedBytes bytes of those blocks allocated to live buffers, staging included, after alignment
 * @param blocks the blocks held now
 * @param allocations the live allocations (buffers and in-flight staging)
 * @param blockAllocations driver allocations made since the device opened: a pooled device makes far fewer than it
 *        allocates buffers
 * @param budgetBytes how much device memory the process should use, or -1 if unknown. On Vulkan it is the
 *        device-local heaps' budget (VK_EXT_memory_budget) or their size; on Metal the recommended working-set size.
 * @param deviceUsageBytes how much of that the process uses now (all allocations, not only these pools), or -1 if
 *        unknown
 * @since 1.5.0
 */
public record MemoryStats(long reservedBytes, long usedBytes, int blocks, long allocations, long blockAllocations,
        long budgetBytes, long deviceUsageBytes) {
    /** No pooling information (a backend without pools). */
    public static final MemoryStats NONE = new MemoryStats(0, 0, 0, 0, 0);

    /** Pool figures with the budget unknown. */
    public MemoryStats(long reservedBytes, long usedBytes, int blocks, long allocations, long blockAllocations) {
        this(reservedBytes, usedBytes, blocks, allocations, blockAllocations, -1, -1);
    }

    /** These figures plus {@code other}'s pools (budget and usage kept from this). */
    public MemoryStats plus(MemoryStats other) {
        return new MemoryStats(reservedBytes + other.reservedBytes, usedBytes + other.usedBytes, blocks + other.blocks,
                allocations + other.allocations, blockAllocations + other.blockAllocations, budgetBytes, deviceUsageBytes);
    }

    /** These figures with the device's budget and usage. */
    public MemoryStats withBudget(long budgetBytes, long deviceUsageBytes) {
        return new MemoryStats(reservedBytes, usedBytes, blocks, allocations, blockAllocations, budgetBytes, deviceUsageBytes);
    }

    /** Bytes still available under the budget, or -1 if unknown. */
    public long headroomBytes() { return budgetBytes < 0 || deviceUsageBytes < 0 ? -1 : Math.max(0, budgetBytes - deviceUsageBytes); }
}
