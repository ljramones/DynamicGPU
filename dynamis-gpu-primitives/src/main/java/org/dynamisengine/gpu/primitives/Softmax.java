package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Row-wise softmax of FP32 scores, with an optional causal mask: for each row {@code r} of an {@code m x n} matrix,
 * {@code y[j] = exp(s[j] - max) / sum exp(s[j'] - max)} over the unmasked columns, and 0 where masked. It is the
 * middle step of attention, between the score and the value products, and writes FP16 to feed {@link GemmF16}
 * directly, or FP32.
 * <p>
 * <b>Causal mask.</b> Column {@code j} of row {@code r} is masked when {@code j > r + causalOffset}: query {@code r}
 * sees keys {@code 0 .. r + causalOffset}. With a cache of {@code n} keys and {@code m} new queries the offset is
 * {@code n - m}. A row with every column masked is all zeros.
 * <p>
 * <b>Accuracy.</b> The max and the sum are combined in a fixed order, so a device's result does not depend on
 * scheduling. {@code exp} is the device's own (within Vulkan's {@code 3 + 2|x|} ulp; Metal compiles with precise
 * functions), so results are bounded, not bit-identical across devices. For each unmasked element, with
 * {@code x = s - max} and {@code xbar} the softmax-weighted mean of {@code |x|} over the row,
 * {@code |y - exact| <= (n + 10 + 2|x| + 2 xbar) 2^-24 exact + 2^-126}, plus one rounding to half for
 * {@link Precision#FP16}. The {@code 2^-126} allows for devices that flush subnormal results to zero, as Apple GPUs do.
 */
public final class Softmax implements AutoCloseable {
    private static final long MAX_INDEX = 0xFFFF_FFFFL;

    /**
     * The rows to normalize. Leading dimensions and strides are in elements: row {@code r} of matrix {@code z} of the
     * scores starts at {@code z * strideS + r * lds}.
     *
     * @param m rows per matrix
     * @param n columns (keys)
     * @param causal whether columns after {@code r + causalOffset} are masked
     * @param batch the number of matrices (for example heads); the batch is the dispatch's z dimension
     */
    public record Shape(int m, int n, int lds, int ldy, boolean causal, int causalOffset, int batch, long strideS, long strideY) {
        public Shape {
            if (m <= 0 || n <= 0) throw new IllegalArgumentException("dimensions must be positive: " + m + "x" + n);
            if (lds < n || ldy < n) throw new IllegalArgumentException("leading dimension smaller than a row: lds=" + lds + " ldy=" + ldy);
            if (causalOffset < 0) throw new IllegalArgumentException("causalOffset must not be negative: " + causalOffset);
            if (batch < 1) throw new IllegalArgumentException("batch must be positive: " + batch);
            for (long st : new long[] {strideS, strideY})
                if (st < 0 || st > MAX_INDEX) throw new IllegalArgumentException("stride out of range: " + st);
            if (batch > 1 && strideY < (long) (m - 1) * ldy + n)
                throw new IllegalArgumentException("output matrices overlap: strideY " + strideY + " < " + ((long) (m - 1) * ldy + n));
        }
        /** Densely packed rows, no mask. */
        public static Shape of(int m, int n) { return new Shape(m, n, n, n, false, 0, 1, 0, 0); }
        /** This shape with a causal mask at {@code offset}. */
        public Shape causal(int offset) { return new Shape(m, n, lds, ldy, true, offset, batch, strideS, strideY); }
        /** This shape as a batch of densely packed matrices. */
        public Shape batched(int count) { return withBatch(count, (long) m * lds, (long) m * ldy); }
        /** This shape as a batch with the given strides. */
        public Shape withBatch(int count, long strideS, long strideY) {
            return new Shape(m, n, lds, ldy, causal, causalOffset, count, strideS, strideY);
        }
        /** Elements the scores and the output span. */
        public long sElements() { return (batch - 1) * strideS + (long) (m - 1) * lds + n; }
        public long yElements() { return (batch - 1) * strideY + (long) (m - 1) * ldy + n; }
        /** Whether column {@code j} of row {@code r} is unmasked. */
        public boolean visible(int r, int j) { return !causal || j <= (long) r + causalOffset; }
    }

    static final KernelLibrary KERNELS = GemmF16.KERNELS;

    private final GpuDevice device;
    private final GpuComputePipeline pipeline;
    private final Precision output;

    private Softmax(GpuDevice device, GpuComputePipeline pipeline, Precision output) {
        this.device = device; this.pipeline = pipeline; this.output = output;
    }

    /** Compiles the kernel writing {@code output}; FP16 output needs {@link GpuCapabilities#FP16}. */
    public static Softmax create(GpuDevice device, Precision output) {
        Set<String> types = output == Precision.FP16 ? Set.of(GpuCapabilities.FP32, GpuCapabilities.FP16) : Set.of(GpuCapabilities.FP32);
        return new Softmax(device, device.pipeline(KERNELS.pipeline(kernel(output), List.of(Access.READ, Access.READ_WRITE), types, Set.of())), output);
    }

    private static String kernel(Precision output) { return output == Precision.FP32 ? "softmax_f32" : "softmax_f16"; }

    /** The element type of the output. */
    public Precision output() { return output; }

    /** Records the softmax of {@code scores} into {@code out}. */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer scores, GpuBuffer out, Shape s) {
        check(scores, s.sElements(), 4, "scores"); check(out, s.yElements(), output.bytes(), "output");
        if ((long) s.m() + s.causalOffset() + 1 > MAX_INDEX) throw new IllegalArgumentException("causalOffset too large");
        byte[] params = ByteBuffer.allocate(KERNELS.parameterBytes(kernel(output))).order(ByteOrder.LITTLE_ENDIAN)
                .putInt(s.m()).putInt(s.n()).putInt(s.lds()).putInt(s.ldy()).putInt(s.causal() ? 1 : 0).putInt(s.causalOffset())
                .putInt((int) s.strideS()).putInt((int) s.strideY()).array();
        return encoder.dispatch(pipeline, Map.of(0, scores, 1, out), params, new Dimensions(s.m(), 1, s.batch()));
    }

    private static void check(GpuBuffer buffer, long elements, int elementBytes, String name) {
        if (buffer.bytes() < elements * elementBytes)
            throw new IllegalArgumentException(name + " holds " + buffer.bytes() + " bytes, needs " + elements * elementBytes);
        if (elements - 1 > MAX_INDEX) throw new IllegalArgumentException(name + " spans more than 2^32 elements");
    }

    /** The device this softmax runs on. */
    public GpuDevice device() { return device; }

    @Override public void close() { pipeline.close(); }
}
