package org.dynamisengine.gpu.api.compute;

/**
 * A device's buffer memory: what its pools hold and how often they have gone to the driver.
 *
 * @param reservedBytes bytes of the blocks the device holds now (used or free)
 * @param usedBytes bytes of those blocks allocated to live buffers, staging included, after alignment
 * @param blocks the blocks held now
 * @param allocations the live allocations (buffers and in-flight staging)
 * @param blockAllocations driver allocations made since the device opened: a pooled device makes far fewer than it
 *        allocates buffers
 * @since 1.5.0
 */
public record MemoryStats(long reservedBytes, long usedBytes, int blocks, long allocations, long blockAllocations) {
    /** No pooling information (a backend without pools). */
    public static final MemoryStats NONE = new MemoryStats(0, 0, 0, 0, 0);
}
