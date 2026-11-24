package org.dynamisengine.gpu.api.compute.kit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Set;
import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.api.compute.Dimensions;
import org.dynamisengine.gpu.api.compute.GpuCapabilities;
import org.dynamisengine.gpu.api.compute.KernelLibrary;
import org.junit.jupiter.api.Test;

class KitTest {
    private static final KernelLibrary LIB = KernelLibrary.load(KitTest.class, "testkit");

    @Test void viewsCoverTheCommonLayouts() {
        var m = StridedView.matrix(10, 3, 4, 6);   // 3x4 inside a 6-wide matrix at 10
        assertEquals(12, m.elements());
        assertEquals(10 + 2 * 6 + 3, m.maxIndex());
        assertEquals(10 + 6 + 2, m.index(1, 2));
        assertFalse(m.isContiguous()); assertTrue(StridedView.matrix(3, 4).isContiguous());
        var t = m.transpose(0, 1);
        assertEquals(m.index(1, 2), t.index(2, 1)); assertEquals(m.maxIndex(), t.maxIndex());
        var s = m.slice(1, 1, 2);
        assertEquals(m.index(2, 1), s.index(2, 0)); assertEquals(6, s.elements());
        var row = m.select(0, 2);
        assertEquals(1, row.rank()); assertEquals(m.index(2, 3), row.index(3));
        var b = m.batch(5, 100);
        assertEquals(3, b.rank()); assertEquals(m.index(1, 1) + 400, b.index(4, 1, 1));
        var field = StridedView.interleaved(0, 100, 12, 7);   // field 7 of 100 records of 12 floats
        assertEquals(7 + 99 * 12, field.maxIndex()); assertTrue(field.isWritable());
        var soa = StridedView.planes(0, 3, 1000, 1024);
        assertEquals(2 * 1024 + 999, soa.maxIndex()); assertTrue(soa.isWritable());
        assertTrue(StridedView.contiguous(0).isEmpty()); assertEquals(-1, StridedView.contiguous(5, 0).maxIndex());
        assertEquals(m, StridedView.of(10, new long[] {3, 4}, new long[] {6, 1}));
        assertEquals(m.hashCode(), StridedView.of(10, new long[] {3, 4}, new long[] {6, 1}).hashCode());
    }

    @Test void writabilityRejectsAliasing() {
        assertFalse(StridedView.contiguous(10).batch(4, 0).isWritable(), "a broadcast repeats elements");
        assertTrue(StridedView.contiguous(10).batch(4, 0).withRank(3).select(0, 0).batch(1, 0).slice(1, 0, 1).isWritable(), "one copy");
        assertFalse(StridedView.of(0, new long[] {4, 4}, new long[] {3, 1}).isWritable(), "rows overlap");
        assertTrue(StridedView.of(0, new long[] {4, 4}, new long[] {4, 1}).isWritable());
        assertTrue(StridedView.matrix(0, 4, 4, 9).transpose(0, 1).isWritable(), "a transpose");
    }

    @Test void viewsRejectBadShapes() {
        assertThrows(IllegalArgumentException.class, () -> StridedView.matrix(0, 2, 5, 4));
        assertThrows(IllegalArgumentException.class, () -> StridedView.of(-1, new long[] {1}, new long[] {1}));
        assertThrows(IllegalArgumentException.class, () -> StridedView.of(0, new long[] {1, 1, 1, 1, 1}, new long[5]));
        assertThrows(IllegalArgumentException.class, () -> StridedView.contiguous(4).slice(0, 2, 3));
        assertThrows(IndexOutOfBoundsException.class, () -> StridedView.contiguous(4).index(4));
        assertThrows(IllegalArgumentException.class, () -> StridedView.interleaved(0, 4, 3, 3));
        assertThrows(ArithmeticException.class, () -> StridedView.of(0, new long[] {Long.MAX_VALUE, 3}, new long[] {2, 1}).maxIndex());
    }

    private static GpuCapabilities caps(int maxX, int maxY) {
        return new GpuCapabilities(1L << 30, 1L << 30, 16, 8, 128, new Dimensions(1024, 1024, 64), 1024,
                new Dimensions(maxX, maxY, 1), Set.of(GpuCapabilities.FP32), Set.of(), false);
    }

    @Test void gridsFoldIntoTheLimits() {
        var g = Grid.linear(caps(65535, 65535), 100_000);
        assertEquals(new Dimensions(65535, 2, 1), g.groups()); assertEquals(65535, g.gridX()); assertEquals(100_000, g.count());
        assertEquals(new Dimensions(7, 1, 1), Grid.linear(caps(65535, 65535), 7).groups());
        assertEquals(new Dimensions(1, 1, 1), Grid.linear(caps(65535, 65535), 0).groups());
        assertEquals(new Dimensions(4, 1, 1), Grid.forThreads(caps(65535, 65535), 256, 64).groups());
        assertEquals(new Dimensions(5, 1, 1), Grid.forThreads(caps(65535, 65535), 257, 64).groups());
        assertEquals(ComputeException.Code.UNSUPPORTED, assertThrows(ComputeException.class, () -> Grid.linear(caps(10, 10), 101)).code());
    }

    @Test void parameterBlocksAreWrittenByName() {
        var layout = ParameterLayout.of(LIB, "k");
        assertEquals(36, layout.bytes());
        assertEquals(List.of("n", "alpha", "shift", "flag", "xOffset", "xExtent0", "xStride0", "xExtent1", "xStride1"), layout.fields());
        byte[] bytes = layout.block().view("x", StridedView.matrix(10, 3, 4, 6)).bool("flag", true).i32("shift", -3)
                .f32("alpha", 1.5f).u32("n", 0xFFFF_FFFFL).bytes();
        var b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(-1, b.getInt(0)); assertEquals(1.5f, b.getFloat(4)); assertEquals(-3, b.getInt(8)); assertEquals(1, b.getInt(12));
        assertArrayEquals(new int[] {10, 3, 6, 4, 1}, new int[] {b.getInt(16), b.getInt(20), b.getInt(24), b.getInt(28), b.getInt(32)});
        // a rank-1 view is padded with a leading unit dimension; a rank-3 one with a unit batch is narrowed
        b = ByteBuffer.wrap(layout.block().view("x", StridedView.contiguous(5, 7)).bool("flag", false).i32("shift", 0)
                .f32("alpha", 0).u32("n", 0).bytes()).order(ByteOrder.LITTLE_ENDIAN);
        assertArrayEquals(new int[] {5, 1, 0, 7, 1}, new int[] {b.getInt(16), b.getInt(20), b.getInt(24), b.getInt(28), b.getInt(32)});
        layout.block().view("x", StridedView.matrix(3, 4).batch(1, 99));
        assertThrows(IllegalArgumentException.class, () -> layout.block().view("x", StridedView.matrix(3, 4).batch(2, 99)), "rank too high");
        assertThrows(IllegalArgumentException.class, () -> layout.block().u32("missing", 1));
        assertThrows(IllegalArgumentException.class, () -> layout.block().f32("n", 1), "wrong type");
        assertThrows(IllegalArgumentException.class, () -> layout.block().u32("n", -1), "out of range");
        assertThrows(IllegalArgumentException.class, () -> layout.block().u32("n", 1).u32("n", 2), "twice");
        assertThrows(IllegalStateException.class, () -> layout.block().u32("n", 1).bytes(), "unwritten fields");
        assertThrows(IllegalArgumentException.class, () -> layout.block().view("y", StridedView.contiguous(1)), "no such view");
    }

    @Test void checksMatchTheCpuContract() {
        assertEquals(5, Checks.u32(5, "x"));
        assertThrows(IllegalArgumentException.class, () -> Checks.u32(1L << 32, "x"));
        assertThrows(IllegalArgumentException.class, () -> Checks.notNull(null, "x"));
        assertThrows(IllegalArgumentException.class, () -> Checks.nonNegative(-1, "x"));
    }
}
