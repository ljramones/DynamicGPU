package org.dynamisengine.gpu.api.compute.spi;

import org.dynamisengine.gpu.api.compute.MemoryStats;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Sub-allocates aligned ranges of large backend blocks, so that most buffer allocations never reach the driver.
 * A backend supplies the blocks (a large buffer each: a {@code VkBuffer} with its memory, an {@code MTLBuffer}) and
 * binds, copies and maps a range at its block offset.
 * <ul>
 * <li>Requests up to {@code dedicatedThreshold} share blocks of {@code blockSize} bytes, first fit in creation
 * order; freed ranges merge with their free neighbours.</li>
 * <li>Larger requests get a dedicated block of exactly their (aligned) size, destroyed when they are freed.</li>
 * <li>When a shared block becomes empty it is destroyed, except one empty block that is kept to absorb churn.</li>
 * </ul>
 * Offsets and sizes are multiples of {@code alignment}. Not thread-safe: devices are confined to their thread.
 *
 * @param <B> the backend's block type
 */
public final class MemoryPool<B> {
    /** Creates and destroys backend blocks. */
    public interface Blocks<B> {
        /** A new block of exactly {@code bytes}; throws {@code ComputeException} (RESOURCE_EXHAUSTED) when out of memory. */
        B create(long bytes);
        void destroy(B block);
    }

    /** A range of a block. */
    public record Allocation<B>(B block, long offset, long bytes, boolean dedicated) { }

    private final class Block {
        final B handle;
        final long size;
        final boolean dedicated;
        /** Free ranges: offset to size. */
        final TreeMap<Long, Long> free = new TreeMap<>();
        long used;
        Block(B handle, long size, boolean dedicated) {
            this.handle = handle; this.size = size; this.dedicated = dedicated;
            if (!dedicated) free.put(0L, size);
        }
    }

    private final Blocks<B> blocks;
    private final long blockSize, alignment, dedicatedThreshold;
    private final List<Block> shared = new ArrayList<>();
    private final Map<B, Block> byHandle = new java.util.IdentityHashMap<>();
    private long used, allocations, blockAllocations;

    /**
     * @param alignment a power of two that every offset and size is rounded to
     * @param dedicatedThreshold requests larger than this get their own block; at most {@code blockSize}
     */
    public MemoryPool(Blocks<B> blocks, long blockSize, long alignment, long dedicatedThreshold) {
        if (alignment <= 0 || Long.bitCount(alignment) != 1) throw new IllegalArgumentException("alignment must be a power of two: " + alignment);
        if (blockSize <= 0 || blockSize % alignment != 0) throw new IllegalArgumentException("block size must be a positive multiple of the alignment");
        if (dedicatedThreshold <= 0 || dedicatedThreshold > blockSize) throw new IllegalArgumentException("dedicated threshold out of range");
        this.blocks = blocks; this.blockSize = blockSize; this.alignment = alignment; this.dedicatedThreshold = dedicatedThreshold;
    }

    private long align(long bytes) { return (bytes + alignment - 1) & -alignment; }

    /** A range of at least {@code bytes} bytes. */
    public Allocation<B> allocate(long bytes) {
        if (bytes <= 0) throw new IllegalArgumentException("allocation must be positive: " + bytes);
        long size = align(bytes);
        if (size < bytes) throw new IllegalArgumentException("allocation too large: " + bytes);
        if (size > dedicatedThreshold) {
            Block b = new Block(blocks.create(size), size, true); blockAllocations++;
            byHandle.put(b.handle, b); b.used = size; used += size; allocations++;
            return new Allocation<>(b.handle, 0, size, true);
        }
        for (Block b : shared) {
            var range = take(b, size);
            if (range >= 0) return taken(b, range, size);
        }
        Block b = new Block(blocks.create(blockSize), blockSize, false); blockAllocations++;
        shared.add(b); byHandle.put(b.handle, b);
        return taken(b, take(b, size), size);
    }

    private Allocation<B> taken(Block b, long offset, long size) {
        b.used += size; used += size; allocations++;
        return new Allocation<>(b.handle, offset, size, false);
    }

    /** First fit in {@code b}: removes and returns the offset of a range of {@code size}, or -1. */
    private long take(Block b, long size) {
        for (var e : b.free.entrySet()) {
            long offset = e.getKey(), length = e.getValue();
            if (length >= size) {
                b.free.remove(offset);
                if (length > size) b.free.put(offset + size, length - size);
                return offset;
            }
        }
        return -1;
    }

    /** Returns {@code a} to the pool. */
    public void free(Allocation<B> a) {
        Block b = byHandle.get(a.block());
        if (b == null) throw new IllegalArgumentException("allocation is not from this pool");
        b.used -= a.bytes(); used -= a.bytes(); allocations--;
        if (b.dedicated) {
            byHandle.remove(b.handle); blocks.destroy(b.handle); return;
        }
        long offset = a.offset(), size = a.bytes();
        if (b.free.containsKey(offset)) throw new IllegalStateException("double free at " + offset);
        var next = b.free.ceilingEntry(offset);
        if (next != null && offset + size > next.getKey()) throw new IllegalStateException("free overlaps a free range at " + offset);
        if (next != null && offset + size == next.getKey()) { size += next.getValue(); b.free.remove(next.getKey()); }
        var prev = b.free.floorEntry(offset);
        if (prev != null) {
            if (prev.getKey() + prev.getValue() > offset) throw new IllegalStateException("free overlaps a free range at " + offset);
            if (prev.getKey() + prev.getValue() == offset) { offset = prev.getKey(); size += prev.getValue(); }
        }
        b.free.put(offset, size);
        if (b.used == 0) {
            boolean anotherEmpty = shared.stream().anyMatch(o -> o != b && o.used == 0);
            if (anotherEmpty) { shared.remove(b); byHandle.remove(b.handle); blocks.destroy(b.handle); }
        }
    }

    /** Whether {@code a}'s block belongs to this pool. */
    public boolean owns(Allocation<B> a) { return byHandle.containsKey(a.block()); }

    /** What the pool holds now, and how many blocks it has created so far. */
    public MemoryStats stats() {
        long reserved = 0;
        for (Block b : byHandle.values()) reserved += b.size;
        return new MemoryStats(reserved, used, byHandle.size(), allocations, blockAllocations);
    }

    /** Destroys every block, whether or not ranges are still allocated (device close). */
    public void destroyAll() {
        for (Block b : byHandle.values()) blocks.destroy(b.handle);
        byHandle.clear(); shared.clear(); used = 0; allocations = 0;
    }
}
