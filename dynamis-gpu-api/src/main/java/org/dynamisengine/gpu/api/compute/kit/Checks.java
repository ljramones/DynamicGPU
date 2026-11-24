package org.dynamisengine.gpu.api.compute.kit;

import org.dynamisengine.gpu.api.compute.GpuBuffer;

/**
 * Argument checks shared by kernel families. They throw {@link IllegalArgumentException}, as a CPU method given the
 * same arguments would.
 *
 * @since 1.7.0
 */
public final class Checks {
    /** The largest element index a kernel can address: kernels index with 32-bit unsigned integers. */
    public static final long MAX_INDEX = 0xFFFF_FFFFL;

    private Checks() { }

    /** Rejects a null argument. */
    public static <T> T notNull(T value, String name) {
        if (value == null) throw new IllegalArgumentException(name + " is null");
        return value;
    }

    /** Rejects a negative count or size. */
    public static long nonNegative(long value, String name) {
        if (value < 0) throw new IllegalArgumentException(name + " must not be negative: " + value);
        return value;
    }

    /** The number of whole elements of {@code elementBytes} bytes in {@code buffer}. */
    public static long elements(GpuBuffer buffer, int elementBytes) {
        if (elementBytes <= 0) throw new IllegalArgumentException("element size must be positive: " + elementBytes);
        return buffer.bytes() / elementBytes;
    }

    /**
     * Checks that {@code n} elements of {@code elementBytes} bytes fit {@code buffer} from element {@code offset}, with
     * 32-bit indices. An empty range is never read, so only its offset is checked.
     */
    public static void range(GpuBuffer buffer, long offset, long n, int elementBytes, String name) {
        notNull(buffer, name);
        long size = elements(buffer, elementBytes);
        if (n < 0 || offset < 0 || offset > size - n || (n > 0 && offset + n - 1 > MAX_INDEX)) {
            throw new IllegalArgumentException(name + " must hold " + n + " elements from element " + offset + " (has "
                    + size + ")");
        }
    }

    /**
     * Checks that output {@code out} is input {@code in} only at the same view: a kernel may update in place, but
     * never reads one element while another thread writes it. Returns whether they are the same buffer.
     */
    public static boolean inPlace(GpuBuffer out, StridedView outView, String outName, GpuBuffer in, StridedView inView,
            String inName) {
        if (out != in) return false;
        if (!outView.equals(inView)) {
            throw new IllegalArgumentException(outName + " may be " + inName + " only at the same view: " + outView
                    + " vs " + inView);
        }
        return true;
    }

    /** Fits {@code value} into an unsigned 32-bit parameter field, or rejects it. */
    public static int u32(long value, String name) {
        if (value < 0 || value > MAX_INDEX) throw new IllegalArgumentException(name + " does not fit 32 bits: " + value);
        return (int) value;
    }
}
