package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.KitOptions;
import org.dynamisengine.gpu.api.compute.kit.ParameterLayout;
import org.dynamisengine.gpu.api.compute.kit.PipelineSet;
import org.dynamisengine.gpu.primitives.GemmF16.Elements;
import org.dynamisengine.gpu.primitives.GemmF16.Epilogue;
import org.dynamisengine.gpu.primitives.GemmF16.Input;
import org.dynamisengine.gpu.primitives.GemmF16.Shape;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Matrix multiply with block-quantized weights (1.9.0): {@code C = epilogue(alpha * A * W^T + beta * C)}, with A
 * ({@code m x k}) FP16 or BF16 activations, W a packed {@link QuantizedMatrix} ({@code n x k}, the weights of a linear
 * layer, as {@link Quantize} or {@link QuantizedMatrix#quantize} writes it) dequantized inside the kernel, and C in FP32
 * or FP16. The shape is a {@link GemmF16.Shape} with B transposed ({@code n} and {@code k} those of W; {@code ldb} is
 * not used), optionally a batch of A and C sharing W; the {@link GemmF16.Epilogue}s apply as in {@code GemmF16}.
 * <p>
 * <b>Why:</b> decode is limited by memory bandwidth, and 4-bit weights with block 32 take 0.28 of FP16's bytes.
 * <p>
 * <b>Accuracy.</b> Products of an activation and a code are exact in FP32, so each block's sum {@code s = sum a q}
 * depends only on its order, and the block adds {@code fl(d s)} (Q4M also {@code fl(m sum a)}) to its total. Both
 * algorithms fix every order, so the result is the same bits on every device: {@link #reference} computes them. Against
 * the FP32 product of A and the dequantized W the error is at most {@code (k + 2) 2^-24 sum |a w|} plus the blocks'
 * roundings; the quantization error itself is the format's ({@link QuantizedMatrix#errorBound}).
 */
public final class QuantizedGemm implements AutoCloseable {
    /** How the multiply is computed. */
    public enum Algorithm {
        /**
         * Decode ({@code m <= 16}): a row of W's blocks is dealt to 32 lanes (block {@code bl} to lane {@code bl % 32});
         * each lane adds its blocks' terms in order, then the lanes combine in a fixed tree.
         */
        SKINNY,
        /** Any m: 64 x 64 tiles, each element adding its blocks' terms in order. */
        TILED
    }

    /** The largest m {@link Algorithm#SKINNY} computes. */
    public static final int SKINNY_MAX_M = 16;
    static final int LANES = 32, SKINNY_COLUMNS = 16, TILE = 64;
    private static final List<Access> ACCESSES = List.of(Access.READ, Access.READ, Access.READ_WRITE, Access.READ, Access.READ, Access.READ);

    private final GpuDevice device;
    /** The algorithms this instance runs. */
    private final List<Algorithm> algorithms;
    /** Pipelines by algorithm, format, block size and (SKINNY) rows tier, compiled on first use. */
    private final Map<String, GpuComputePipeline> pipelines = new java.util.HashMap<>();
    private final boolean dispatch;
    private final Input input;
    private final Precision output;
    private final ParameterLayout layout;
    private PipelineSet placeholders;

    private QuantizedGemm(GpuDevice device, List<Algorithm> algorithms, boolean dispatch, Input input, Precision output) {
        this.device = device; this.algorithms = algorithms; this.dispatch = dispatch; this.input = input; this.output = output;
        this.layout = ParameterLayout.of(GemmF16.KERNELS, kernel(Algorithm.SKINNY, Input.FP16, Precision.FP32));
    }

    /** Both algorithms, chosen per shape: SKINNY for {@code m <= 16}, TILED beyond. */
    public static QuantizedGemm create(GpuDevice device, Input input, Precision output) {
        return compile(device, List.of(Algorithm.SKINNY, Algorithm.TILED), true, input, output);
    }

    /** One algorithm for every shape. */
    public static QuantizedGemm create(GpuDevice device, Algorithm algorithm, Input input, Precision output) {
        return compile(device, List.of(Checks.notNull(algorithm, "algorithm")), false, input, output);
    }

    private static QuantizedGemm compile(GpuDevice device, List<Algorithm> algorithms, boolean dispatch, Input input, Precision output) {
        Checks.notNull(device, "device"); Checks.notNull(input, "input"); Checks.notNull(output, "output");
        if (input == Input.FP32)
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, "quantized GEMM takes FP16 or BF16 activations, whose products with codes are exact");
        if (!device.capabilities().dataTypes().containsAll(types(input, output)))
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, "quantized GEMM with " + input + " activations and " + output + " C needs FP16");
        return new QuantizedGemm(device, List.copyOf(algorithms), dispatch, input, output);
    }

    private static Set<String> types(Input input, Precision output) {
        return input == Input.FP16 || output == Precision.FP16 ? Set.of(GpuCapabilities.FP32, GpuCapabilities.FP16) : Set.of(GpuCapabilities.FP32);
    }

    /** The rows of A a SKINNY pipeline computes: 1, 4 or 16, so decode holds one row of accumulators. */
    static int rowsTier(int m) { return m <= 1 ? 1 : m <= 4 ? 4 : SKINNY_MAX_M; }

    /** The pipeline of {@code algorithm} for {@code w}'s format and block (and SKINNY's rows tier for m), compiled once. */
    private GpuComputePipeline pipeline(Algorithm algorithm, QuantizedMatrix w, int m) {
        int rows = algorithm == Algorithm.SKINNY ? rowsTier(m) : 0;
        String key = algorithm + "/" + w.format() + "/" + w.block() + "/" + rows;
        var p = pipelines.get(key);
        if (p == null) {
            Map<String, Object> constants = new java.util.HashMap<>(Map.of("QFORMAT", w.format().ordinal(), "QBLOCK", w.block()));
            if (algorithm == Algorithm.SKINNY) constants.put("QROWS", rows);
            p = device.pipeline(GemmF16.KERNELS.pipeline(kernel(algorithm, input, output), ACCESSES, types(input, output), Set.of(), constants));
            pipelines.put(key, p);
        }
        return p;
    }

    static String kernel(Algorithm algorithm, Input input, Precision output) {
        return "gemm_q_" + (algorithm == Algorithm.SKINNY ? "skinny" : "tiled") + (input == Input.BF16 ? "_bf16" : "")
                + (output == Precision.FP16 ? "_h" : "");
    }

    /** The algorithm {@link #encode} runs for {@code s}. */
    public Algorithm algorithm(Shape s) {
        if (!dispatch) return algorithms.getFirst();
        return s.m() <= SKINNY_MAX_M && !KitOptions.current().reproducible() ? Algorithm.SKINNY : Algorithm.TILED;
    }

    public Input input() { return input; }
    public Precision output() { return output; }

    /** Records {@code C = alpha A W^T + beta C}. */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer a, QuantizedMatrix w, GpuBuffer packed, GpuBuffer c, Shape s) {
        return encode(encoder, a, w, packed, c, s, Epilogue.NONE, null, null, null);
    }

    /** Records {@code C = epilogue(alpha A W^T + beta C)}; the epilogue's operands as in {@code GemmF16.encode}. */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer a, QuantizedMatrix w, GpuBuffer packed, GpuBuffer c, Shape s,
            Epilogue epilogue, GpuBuffer bias, GpuBuffer gate, GpuBuffer residual) {
        Checks.notNull(w, "weights"); Checks.notNull(s, "shape"); Checks.notNull(epilogue, "epilogue");
        if (!s.transposeB() || s.transposeA()) throw new IllegalArgumentException("the weights are W^T, A is not transposed: " + s);
        if (s.n() != w.rows() || s.k() != w.cols()) throw new IllegalArgumentException("shape " + s.n() + "x" + s.k() + " does not match the weights " + w);
        if (s.strideB() != 0 && s.batch() > 1) throw new IllegalArgumentException("a batch shares the weights: strideB must be 0");
        Algorithm run = algorithm(s);
        if (run == Algorithm.SKINNY && s.m() > SKINNY_MAX_M) throw new IllegalArgumentException("SKINNY needs m <= " + SKINNY_MAX_M + ": " + s);
        Checks.range(a, 0, s.aElements(), input.bytes(), "A");
        Checks.range(packed, 0, w.bytes() / 4, 4, "weights");
        Checks.range(c, 0, s.cElements(), output.bytes(), "C");
        if (a.offset() % 4 != 0 || packed.offset() % 4 != 0) throw new IllegalArgumentException("A and the weights must start on 4 bytes");
        bias = operand(bias, epilogue.bias() != Epilogue.Bias.NONE, epilogue.bias() == Epilogue.Bias.ROW ? s.m() : s.n(), "bias");
        gate = operand(gate, epilogue.gated(), s.cElements(), "gate");
        residual = operand(residual, epilogue.residual(), s.cElements(), "residual");
        byte[] params = layout.block().u32("m", s.m()).u32("n", s.n()).u32("k", s.k()).u32("lda", s.lda()).u32("ldb", s.k())
                .u32("ldc", s.ldc()).f32("alpha", s.alpha()).f32("beta", s.beta()).bool("transB", true).u32("strideA", s.strideA())
                .u32("strideB", 0).u32("strideC", s.strideC()).bool("transA", false).u32("bias", epilogue.bias().ordinal())
                .u32("act", epilogue.activation().ordinal()).bool("gated", epilogue.gated()).bool("residual", epilogue.residual())
                .u32("qparams", w.parameterOffset() / 4).bytes();
        var groups = run == Algorithm.SKINNY ? new Dimensions(Math.ceilDiv(s.n(), SKINNY_COLUMNS), 1, s.batch())
                : new Dimensions(Math.ceilDiv(s.n(), TILE), Math.ceilDiv(s.m(), TILE), s.batch());
        return encoder.dispatch(pipeline(run, w, s.m()), Map.of(0, a, 1, packed, 2, c, 3, bias, 4, gate, 5, residual), params, groups,
                run == Algorithm.SKINNY ? "qgemm.skinny" : "qgemm.tiled");
    }

    private GpuBuffer operand(GpuBuffer buffer, boolean used, long elements, String name) {
        if (!used) {
            if (buffer != null) throw new IllegalArgumentException(name + " given, but the epilogue does not use it");
            if (placeholders == null) placeholders = PipelineSet.compile(device, List.of());
            return placeholders.placeholder();
        }
        if (buffer == null) throw new IllegalArgumentException("the epilogue needs " + name);
        Checks.range(buffer, 0, elements, output.bytes(), name);
        return buffer;
    }

    // --- the CPU reference -----------------------------------------------------------------------------------------

    /** Block {@code bl}'s term for row i of A (matrix z) and row j of W: {@code fl(d s)} or {@code fl(fl(d s) + fl(m sum a))}. */
    private static float blockTerm(Elements a, QuantizedMatrix w, byte[] packed, Shape s, int z, int i, int j, int bl) {
        float sum = 0, sumA = 0;
        for (int t = 0; t < w.block(); t++) {
            int col = bl * w.block() + t;
            float av = a.at(s.aIndex(z, i, col));
            sumA = sumA + av;
            sum = Math.fma(av, (float) w.code(packed, j, col), sum);   // an exact product
        }
        float term = w.parameter(packed, j, bl, 0) * sum;
        if (w.format() == QuantizedMatrix.Format.Q4M) term = term + w.parameter(packed, j, bl, 1) * sumA;
        return term;
    }

    private static float sum(Algorithm algorithm, Elements a, QuantizedMatrix w, byte[] packed, Shape s, int z, int i, int j) {
        int blocks = w.blocksPerRow();
        if (algorithm == Algorithm.TILED) {
            float acc = 0;
            for (int bl = 0; bl < blocks; bl++) acc = acc + blockTerm(a, w, packed, s, z, i, j, bl);
            return acc;
        }
        float[] part = new float[LANES];
        for (int bl = 0; bl < blocks; bl++) part[bl % LANES] = part[bl % LANES] + blockTerm(a, w, packed, s, z, i, j, bl);
        for (int wd = LANES / 2; wd > 0; wd /= 2) for (int t = 0; t < wd; t++) part[t] += part[t + wd];
        return part[0];
    }

    /**
     * The CPU result of {@code algorithm} with an FP32 C, bit for bit on every device:
     * {@code C = epilogue(fl(fl(alpha acc) + fl(beta c)))}, with {@code acc} the algorithm's sum of block terms. The
     * epilogue's GELU and SiLU use {@code Math.exp} (the device's {@code exp} differs; see {@code GemmF16}).
     */
    public static void reference(Algorithm algorithm, Elements a, QuantizedMatrix w, byte[] packed, float[] c, Shape s, Epilogue epilogue,
            float[] bias, float[] gate, float[] residual) {
        for (int z = 0; z < s.batch(); z++) {
            for (int i = 0; i < s.m(); i++) {
                for (int j = 0; j < s.n(); j++) {
                    float acc = sum(algorithm, a, w, packed, s, z, i, j);
                    int idx = (int) s.cIndex(z, i, j);
                    float r = s.beta() == 0 ? s.alpha() * acc : s.alpha() * acc + s.beta() * c[idx];
                    GemmF16.Operand b = bias == null ? x -> { throw new IllegalArgumentException("bias missing"); } : x -> bias[x];
                    c[idx] = GemmF16.epilogue(r, i, j, idx, epilogue, b, gate == null ? null : x -> gate[x], residual == null ? null : x -> residual[x]);
                }
            }
        }
    }

    /** As {@link #reference} with an FP16 C and FP16 epilogue operands (raw bits), rounded once to half. */
    public static void referenceF16(Algorithm algorithm, Elements a, QuantizedMatrix w, byte[] packed, short[] c, Shape s, Epilogue epilogue,
            short[] bias, short[] gate, short[] residual) {
        for (int z = 0; z < s.batch(); z++) {
            for (int i = 0; i < s.m(); i++) {
                for (int j = 0; j < s.n(); j++) {
                    float acc = sum(algorithm, a, w, packed, s, z, i, j);
                    int idx = (int) s.cIndex(z, i, j);
                    float r = s.beta() == 0 ? s.alpha() * acc : s.alpha() * acc + s.beta() * Float.float16ToFloat(c[idx]);
                    GemmF16.Operand b = bias == null ? x -> { throw new IllegalArgumentException("bias missing"); } : x -> Float.float16ToFloat(bias[x]);
                    c[idx] = Float.floatToFloat16(GemmF16.epilogue(r, i, j, idx, epilogue, b, gate == null ? null : x -> Float.float16ToFloat(gate[x]),
                            residual == null ? null : x -> Float.float16ToFloat(residual[x])));
                }
            }
        }
    }

    /** The contract of {@code algorithm} with {@code input} activations, writing {@code output} through {@code epilogue}. */
    public static AccuracyContract contract(Algorithm algorithm, Input input, Precision output, Epilogue epilogue) {
        String order = algorithm == Algorithm.SKINNY
                ? "FP32: per block s = sum a q in order of k (exact products), then fl(d s) [+ fl(m sum a)]; blocks dealt to 32 lanes, each in order, then a fixed tree"
                : "FP32: per block s = sum a q in order of k (exact products), then fl(d s) [+ fl(m sum a)]; blocks in order";
        String bound = "against A times the dequantized W: |C - exact| <= (k + 2) 2^-24 sum |a w|"
                + (epilogue.equals(Epilogue.NONE) ? "" : "; then the epilogue in FP32") + (output == Precision.FP16 ? "; then one rounding to half" : "");
        var b = epilogue.exact() ? new AccuracyContract.Bound(AccuracyContract.Bound.Kind.EXACT, 0, "bit for bit QuantizedGemm.reference; " + bound)
                : AccuracyContract.Bound.formula(bound + "; GELU and SiLU use the device's exp");
        return new AccuracyContract("QuantizedGemm " + algorithm + ", " + input + " activations, " + output + " C", List.of(input.name(), "Q8/Q4/Q4M blocks"),
                order, output + ", epilogue(fl(fl(alpha acc) + fl(beta c)))", AccuracyContract.Order.FIXED, b, "QuantizedGemm.reference");
    }

    @Override public void close() {
        var all = new java.util.ArrayList<AutoCloseable>(pipelines.values());
        all.add(placeholders);
        PipelineSet.closeAll(all.toArray(AutoCloseable[]::new));
    }
}
