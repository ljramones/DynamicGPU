package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;

/**
 * Scaled dot-product attention for every head at once: {@code O = softmax(scale Q K^T + mask) V}. Q is
 * {@code [heads][queries][headDim]}, K and V {@code [heads][keys][headDim]}, all FP16, densely packed. O is
 * {@code [heads][queries][headDim]} in FP16 or FP32.
 * <p>
 * <b>The steps.</b> It records three dispatches.
 * <ol>
 * <li>{@code S = scale Q K^T}: {@link GemmF16} with transposed B, FP32, into the scores buffer.</li>
 * <li>{@code P = softmax(S)}: {@link Softmax}, FP16, causal if asked, into the probabilities buffer.</li>
 * <li>{@code O = P V}: {@link GemmF16}.</li>
 * </ol>
 * The scratch buffers hold {@code heads * queries * keys} elements ({@link Shape#scoresBytes()},
 * {@link Shape#probsBytes()}). The memory grows with the square of the sequence length. A fused (flash) kernel
 * that keeps the scores on chip is a later step.
 * <p>
 * <b>Causal.</b> Query {@code i} attends keys {@code 0 .. i + keys - queries}, so new queries at the end of a KV
 * cache see the whole cache and themselves. That needs {@code keys >= queries}.
 * <p>
 * <b>Accuracy.</b> Each step keeps its own bound; the error of O is their composition.
 * <ul>
 * <li>The scores' error from {@link GemmF16} perturbs P by at most {@code 2 P max|dS|}.</li>
 * <li>{@link Softmax} and the FP16 rounding of P add their bounds.</li>
 * <li>The value product adds {@code (keys + 2) 2^-24 sum P |V|}, then any rounding of O.</li>
 * </ul>
 * The conformance tests check every element of O against that composition, computed in double precision.
 */
public final class AttentionF16 implements AutoCloseable {
    /**
     * The attention's dimensions.
     *
     * @param scale the factor on {@code Q K^T}, usually {@code 1 / sqrt(headDim)}
     * @param causal whether query {@code i} is limited to keys {@code 0 .. i + keys - queries}
     */
    public record Shape(int heads, int queries, int keys, int headDim, float scale, boolean causal) {
        public Shape {
            if (heads <= 0 || queries <= 0 || keys <= 0 || headDim <= 0)
                throw new IllegalArgumentException("dimensions must be positive: " + heads + "x" + queries + "x" + keys + "x" + headDim);
            if (causal && keys < queries) throw new IllegalArgumentException("causal attention needs keys >= queries: " + keys + " < " + queries);
            if (!Float.isFinite(scale)) throw new IllegalArgumentException("scale must be finite: " + scale);
        }
        /** No mask, {@code scale = 1 / sqrt(headDim)}. */
        public static Shape of(int heads, int queries, int keys, int headDim) {
            return new Shape(heads, queries, keys, headDim, (float) (1 / Math.sqrt(headDim)), false);
        }
        /** This shape with the causal mask. */
        public Shape asCausal() { return new Shape(heads, queries, keys, headDim, scale, true); }
        public long qBytes() { return 2L * heads * queries * headDim; }
        public long kBytes() { return 2L * heads * keys * headDim; }
        public long vBytes() { return kBytes(); }
        /** The bytes O must hold for {@code output}. */
        public long oBytes(Precision output) { return (long) output.bytes() * heads * queries * headDim; }
        /** The FP32 scores scratch: {@code heads * queries * keys} floats. */
        public long scoresBytes() { return 4L * heads * queries * keys; }
        /** The FP16 probabilities scratch: {@code heads * queries * keys} halves. */
        public long probsBytes() { return 2L * heads * queries * keys; }
        /** Floating-point operations of the two products. */
        public double flops() { return 4.0 * heads * queries * keys * headDim; }

        GemmF16.Shape scoresShape() { return GemmF16.Shape.ofTransposedB(queries, keys, headDim, scale, 0f).batched(heads); }
        Softmax.Shape softmaxShape() {
            var s = Softmax.Shape.of(queries, keys).batched(heads);
            return causal ? s.causal(keys - queries) : s;
        }
        GemmF16.Shape outputShape() { return GemmF16.Shape.of(queries, headDim, keys, 1f, 0f).batched(heads); }
    }

    private final GemmF16 scores;
    private final Softmax softmax;
    private final GemmF16 values;

    private AttentionF16(GemmF16 scores, Softmax softmax, GemmF16 values) {
        this.scores = scores; this.softmax = softmax; this.values = values;
    }

    /** Compiles the steps with the fastest GEMM the device has, writing O in {@code output}. Needs {@code FP16}. */
    public static AttentionF16 create(GpuDevice device, Precision output) {
        GemmF16 s = null; Softmax p = null;
        try {
            s = GemmF16.create(device, Precision.FP32);
            p = Softmax.create(device, Precision.FP16);
            return new AttentionF16(s, p, GemmF16.create(device, s.algorithm(), output));
        } catch (RuntimeException e) {
            if (p != null) p.close();
            if (s != null) s.close();
            throw e;
        }
    }

    /** The GEMM algorithm both products use. */
    public GemmF16.Algorithm algorithm() { return scores.algorithm(); }

    /** The element type of O. */
    public Precision output() { return values.output(); }

    /** Records the attention; {@code scores} and {@code probs} are scratch (see {@link Shape#scoresBytes()}). */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer o,
            GpuBuffer scoresScratch, GpuBuffer probsScratch, Shape s) {
        scores.encode(encoder, q, k, scoresScratch, s.scoresShape());
        softmax.encode(encoder, scoresScratch, probsScratch, s.softmaxShape());
        return values.encode(encoder, probsScratch, v, o, s.outputShape());
    }

    @Override public void close() {
        try { values.close(); } finally { try { softmax.close(); } finally { scores.close(); } }
    }
}
