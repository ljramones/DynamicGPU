package org.dynamisengine.gpu.api.compute;

import org.dynamisengine.gpu.api.compute.spi.MemoryPool;
import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

/** MemoryPool against a model: no overlaps, alignment, reuse, merging, dedicated blocks and accounting. */
class MemoryPoolTest {
    /** Fake blocks: integers, recording creations and destructions. */
    static final class Fake implements MemoryPool.Blocks<Integer> {
        int next; final Map<Integer, Long> live = new HashMap<>(); int created, destroyed;
        @Override public Integer create(long bytes) { created++; live.put(next, bytes); return next++; }
        @Override public void destroy(Integer block) { destroyed++; assertNotNull(live.remove(block), "destroyed twice"); }
    }

    @Test void randomAllocateAndFreeAgainstAModel() {
        var fake = new Fake(); var pool = new MemoryPool<>(fake, 1 << 20, 256, 1 << 18);
        var r = new SplittableRandom(7); List<MemoryPool.Allocation<Integer>> live = new ArrayList<>();
        long liveBytes = 0;
        for (int step = 0; step < 20000; step++) {
            if (live.isEmpty() || r.nextInt(100) < 55) {
                long bytes = r.nextInt(10) == 0 ? r.nextLong(1, 600_000) : r.nextLong(1, 20_000);
                var a = pool.allocate(bytes);
                assertTrue(a.bytes() >= bytes && a.bytes() % 256 == 0 && a.offset() % 256 == 0, "aligned and large enough");
                assertEquals(bytes > (1 << 18), a.dedicated());
                assertTrue(a.offset() + a.bytes() <= fake.live.get(a.block()), "inside its block");
                live.add(a); liveBytes += a.bytes();
            } else {
                var a = live.remove(r.nextInt(live.size())); liveBytes -= a.bytes(); pool.free(a);
            }
            if (step % 997 == 0) checkNoOverlap(live);
            var st = pool.stats();
            assertEquals(liveBytes, st.usedBytes()); assertEquals(live.size(), st.allocations());
            assertEquals(fake.live.size(), st.blocks()); assertEquals(fake.created, st.blockAllocations());
        }
        checkNoOverlap(live);
        for (var a : live) pool.free(a);
        var st = pool.stats();
        assertEquals(0, st.usedBytes()); assertTrue(st.blocks() <= 1, "at most one empty block kept, got " + st.blocks());
        assertTrue(fake.created < 20000 / 10, "blocks are reused: " + fake.created + " created");
        pool.destroyAll(); assertTrue(fake.live.isEmpty());
    }

    static void checkNoOverlap(List<MemoryPool.Allocation<Integer>> live) {
        Map<Integer, List<long[]>> byBlock = new HashMap<>();
        for (var a : live) byBlock.computeIfAbsent(a.block(), k -> new ArrayList<>()).add(new long[] {a.offset(), a.offset() + a.bytes()});
        for (var ranges : byBlock.values()) {
            ranges.sort(Comparator.comparingLong(x -> x[0]));
            for (int i = 1; i < ranges.size(); i++) assertTrue(ranges.get(i - 1)[1] <= ranges.get(i)[0], "ranges overlap");
        }
    }

    @Test void freedRangesMergeSoAFullBlockFitsAgain() {
        var fake = new Fake(); var pool = new MemoryPool<>(fake, 4096, 256, 4096);
        var a = pool.allocate(1024); var b = pool.allocate(1024); var c = pool.allocate(2048);
        assertEquals(1, fake.created);
        pool.free(b); pool.free(a); pool.free(c);   // merges into one free range
        var whole = pool.allocate(4096);
        assertEquals(0, whole.offset()); assertEquals(1, fake.created, "the merged block is reused");
    }

    @Test void dedicatedBlocksAreDestroyedOnFreeAndOneEmptySharedBlockIsKept() {
        var fake = new Fake(); var pool = new MemoryPool<>(fake, 4096, 256, 2048);
        var big = pool.allocate(3000); assertTrue(big.dedicated()); assertEquals(3072, big.bytes());
        pool.free(big); assertEquals(1, fake.destroyed);
        var x1 = pool.allocate(2048); var x2 = pool.allocate(2048);   // fill the first shared block
        var y = pool.allocate(1024);                                  // a second shared block
        assertEquals(3, fake.created);
        pool.free(x1); pool.free(x2); assertEquals(1, fake.destroyed, "the first empty block is kept");
        pool.free(y); assertEquals(2, fake.destroyed, "a second empty block is destroyed");
    }

    @Test void rejectsMisuse() {
        var fake = new Fake(); var pool = new MemoryPool<>(fake, 4096, 256, 4096);
        var a = pool.allocate(100); pool.free(a);
        assertThrows(IllegalStateException.class, () -> pool.free(a));
        assertThrows(IllegalArgumentException.class, () -> pool.allocate(0));
        assertThrows(IllegalArgumentException.class, () -> new MemoryPool<>(fake, 4096, 100, 4096));
        assertThrows(IllegalArgumentException.class, () -> pool.free(new MemoryPool.Allocation<>(99, 0, 256, false)));
    }
}
