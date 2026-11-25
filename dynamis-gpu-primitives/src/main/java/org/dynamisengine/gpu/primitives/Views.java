package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.StridedView;

/** Shape rules shared by the kit primitives: rank-4 padding, broadcasting and row views. */
final class Views {
    private Views() { }

    /** {@code in} padded to rank 4, checked to broadcast to {@code out4}: each extent equal or 1. */
    static StridedView broadcast(StridedView in, StridedView out4, String name) {
        var in4 = in.withRank(4);
        for (int d = 0; d < 4; d++) {
            if (in4.extent(d) != out4.extent(d) && in4.extent(d) != 1) {
                throw new IllegalArgumentException(name + " " + in + " does not broadcast to the output's shape " + out4);
            }
        }
        return in4;
    }

    /**
     * The rank-4 output view of a reduction along {@code x4}'s last dimension: {@code y} has {@code x}'s shape without
     * its last dimension (a 1-element view when {@code x} has rank 1); a unit last dimension is appended.
     */
    static StridedView rows(StridedView x, StridedView x4, StridedView y, String name) {
        StridedView y4;
        if (x.rank() == 1) {
            if (y.elements() != 1) throw new IllegalArgumentException(name + " of a rank-1 input must hold 1 element: " + y);
            y4 = StridedView.of(y.maxIndex(), new long[] {1, 1, 1, 1}, new long[] {0, 0, 0, 0});
        } else {
            if (y.rank() != x.rank() - 1) throw new IllegalArgumentException(name + " needs rank " + (x.rank() - 1) + ": " + y);
            long[] e = new long[y.rank() + 1], s = new long[y.rank() + 1];
            for (int d = 0; d < y.rank(); d++) { e[d] = y.extent(d); s[d] = y.stride(d); }
            e[y.rank()] = 1;
            y4 = StridedView.of(y.offset(), e, s).withRank(4);
        }
        for (int d = 0; d < 3; d++) {
            if (y4.extent(d) != x4.extent(d)) throw new IllegalArgumentException(name + " " + y + " is not the input's shape without its last dimension");
        }
        return y4;
    }

    /**
     * Writes a broadcast input ({@link #broadcast}'s result) as {@code prefixOffset} and {@code prefixStride0..3}, the
     * stride 0 along each dimension of extent 1: the kernel reads it with the output's coordinates (1.14.0).
     */
    static org.dynamisengine.gpu.api.compute.kit.ParameterLayout.Block broadcastStrides(
            org.dynamisengine.gpu.api.compute.kit.ParameterLayout.Block block, String prefix, StridedView in4) {
        block.u32(prefix + "Offset", in4.offset());
        for (int d = 0; d < 4; d++) block.u32(prefix + "Stride" + d, in4.extent(d) == 1 ? 0 : in4.stride(d));
        return block;
    }

    /** The rows of a rank-4 view: the product of its first three extents, within 32 bits. */
    static long rowCount(StridedView x4) {
        long rows = x4.extent(0) * x4.extent(1) * x4.extent(2);
        Checks.u32(rows, "row count");
        Checks.u32(x4.elements(), "element count");
        return rows;
    }

    /** The element index in {@code v} (rank 4) of row {@code row}, column {@code j}. */
    static long at(StridedView v4, long row, long j) {
        long i2 = row % v4.extent(2), rest = row / v4.extent(2);
        long i1 = rest % v4.extent(1), i0 = rest / v4.extent(1);
        return v4.offset() + i0 * v4.stride(0) + i1 * v4.stride(1) + i2 * v4.stride(2) + j * v4.stride(3);
    }
}
