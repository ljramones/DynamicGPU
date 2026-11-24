package org.dynamisengine.gpu.api.compute;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.EnumSet;

/**
 * The layout of a kernel error buffer: how a kernel reports a data-dependent error (an out-of-range index, a NaN
 * input) that the host could not check before the dispatch, so that it surfaces through completion instead of as
 * silently wrong results.
 * <p>
 * The buffer holds {@link #WORDS} little-endian 32-bit words, updated with atomics so the report does not depend on
 * which thread wins:
 * <ul>
 * <li>word 0: the bitwise OR of every reported code (codes are application-defined, nonzero bit masks);</li>
 * <li>word 1: the bitwise complement of the smallest reported index, kept by atomic max ({@code 0} means none);</li>
 * <li>word 2: the number of reports (wraps at 2^32);</li>
 * <li>word 3: reserved, zero.</li>
 * </ul>
 * Shader helpers implementing it ({@code reportError(code, index)}) ship as resources beside this class:
 * {@code shader/kernel_errors.slang}, {@code .glsl} and {@code .metal}. Bind the buffer read-write, record
 * {@link GpuCommandEncoder#resetErrors} before the dispatches and {@link GpuCommandEncoder#checkErrors} after them.
 *
 * @since 1.2.0
 */
public final class KernelErrors {
    /** The number of 32-bit words. */
    public static final int WORDS = 4;
    /** The size in bytes. */
    public static final int BYTES = WORDS * Integer.BYTES;

    private KernelErrors() { }

    /** A descriptor for an error buffer: device-private, bindable as storage and readable by transfer. */
    public static BufferDescriptor descriptor() {
        return new BufferDescriptor(BYTES, EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE,
                BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE);
    }

    /**
     * What the kernels reported.
     *
     * @param codes the OR of the reported codes
     * @param firstIndex the smallest reported index, or -1 if none
     * @param count the number of reports, as an unsigned 32-bit value
     */
    public record Report(int codes, long firstIndex, long count) {
        /** Whether any kernel reported an error. */
        public boolean any() { return codes != 0 || count != 0; }

        @Override public String toString() {
            return String.format("codes 0x%08x, first index %d, %d report(s)", codes, firstIndex, count);
        }
    }

    /** Parses the bytes of an error buffer, as read back. */
    public static Report parse(byte[] bytes) {
        if (bytes.length < BYTES) throw new IllegalArgumentException("an error buffer has " + BYTES + " bytes, got " + bytes.length);
        var b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int codes = b.getInt(0);
        int notIndex = b.getInt(4);
        long count = Integer.toUnsignedLong(b.getInt(8));
        return new Report(codes, notIndex == 0 ? -1 : Integer.toUnsignedLong(~notIndex), count);
    }
}
