package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.KitOptions;
import org.dynamisengine.gpu.api.compute.kit.PipelineSet;
import org.dynamisengine.gpu.api.compute.kit.ShapeRegistry;
import org.dynamisengine.gpu.api.compute.kit.ParameterLayout;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Half-precision matrix multiply: {@code C = alpha * A * B + beta * C}, with A ({@code m x k}) and B ({@code k x n},
 * or {@code n x k} when transposed) in FP16 and C ({@code m x n}) in FP32 or FP16, all row-major, one multiply or a
 * batch. It is the dense layer of a neural network ({@code x W^T} with transposed B), and with a batch, the per-head
 * products of attention.
 * <p>
 * <b>Accuracy.</b> The contract is a bound: before {@code alpha} and {@code beta},
 * {@code |error| <= k * 2^-24 * sum |a b|}, plus one rounding to half for {@link Precision#FP16}. Later kernel versions
 * may sum in another order, for example on matrix hardware, within the same bound.
 * <p>
 * <b>Algorithms.</b> {@link Algorithm#TILED} runs everywhere and is reproducible; {@link Algorithm#SIMDGROUP_MATRIX}
 * uses Apple GPU matrix hardware ({@link GpuCapabilities#SUBGROUP_MATRIX_F16} on native Metal) and is faster, within
 * the same bound but summing in the hardware's order. {@link #create(GpuDevice)} picks the fastest per shape
 * ({@link #REGISTRY}; 1.7.0), or TILED in reproducible mode.
 * <p>
 * <b>Reproducibility.</b> The tiled algorithm gives the same bits on every device that rounds FP32 add and multiply
 * correctly (tested on native Metal, MoltenVK and lavapipe):
 * <ul>
 * <li>Products of halves are exact in FP32, so each element of C is {@code acc = fl(acc + a b)} over {@code k} in
 * order, fused multiply-add or not.</li>
 * <li>The result is {@code r = fl(fl(alpha acc) + fl(beta c))}, stored as is (FP32) or rounded once to the nearest
 * half (FP16).</li>
 * <li>{@link #reference} and {@link #referenceF16} compute the same bits on the CPU.</li>
 * </ul>
 * <p>
 * <b>Use.</b> Create one per device and output type (it compiles a pipeline), then record multiplies into encoders.
 * With {@code beta == 0}, C is not read, as in BLAS. Requires {@link GpuCapabilities#FP16}; {@code create} throws
 * {@link ComputeException.Code#UNSUPPORTED} without it.
 */
public final class GemmF16 implements AutoCloseable {
    static final KernelLibrary KERNELS = KernelLibrary.load(GemmF16.class, "kernels", "kernels.properties", "dynamisgpu.primitives.");
    /** The output tile one group computes (both algorithms). */
    static final int TILE = 64;
    private static final long MAX_INDEX = Checks.MAX_INDEX;

    /** How the multiply is computed. */
    public enum Algorithm {
        /** Shared-memory tiles and FP32 fused multiply-adds: every device with FP16, bit for bit with {@link #reference}. */
        TILED,
        /**
         * SIMD-group matrices (8 x 8 multiply-accumulate in hardware): native Metal on Apple7 and later GPUs, which
         * report {@link GpuCapabilities#SUBGROUP_MATRIX_F16}. Within the bound; not promised bit for bit with
         * {@link #reference}.
         */
        SIMDGROUP_MATRIX
    }

    /**
     * A multiply's shape, scalars, B layout and batch. Leading dimensions and strides are in elements: row {@code r}
     * of matrix {@code z} of A starts at {@code z * strideA + r * lda}.
     *
     * @param m rows of A and C
     * @param n columns of C
     * @param k columns of A
     * @param transposeB B is stored {@code n x k} (row {@code j} holds column {@code j} of the product's B), as the
     *        weights of a linear layer are; {@code ldb} is then at least {@code k}
     * @param batch the number of multiplies; matrix {@code z} uses A, B and C at {@code z} times their strides
     * @param strideA elements between consecutive A matrices; 0 shares one A across the batch (likewise B)
     * @param strideC elements between consecutive C matrices; with a batch, C matrices must not overlap
     */
    public record Shape(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, boolean transposeB,
            int batch, long strideA, long strideB, long strideC) {
        public Shape {
            if (m <= 0 || n <= 0 || k <= 0) throw new IllegalArgumentException("dimensions must be positive: " + m + "x" + n + "x" + k);
            if (lda < k || ldc < n || ldb < (transposeB ? k : n))
                throw new IllegalArgumentException("leading dimension smaller than a row: lda=" + lda + " ldb=" + ldb + " ldc=" + ldc);
            if (batch < 1) throw new IllegalArgumentException("batch must be positive: " + batch);
            for (long st : new long[] {strideA, strideB, strideC})
                if (st < 0 || st > MAX_INDEX) throw new IllegalArgumentException("stride out of range: " + st);
            if (batch > 1 && strideC < (long) (m - 1) * ldc + n)
                throw new IllegalArgumentException("C matrices overlap: strideC " + strideC + " < " + ((long) (m - 1) * ldc + n));
        }
        /** One multiply with B not transposed. */
        public Shape(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta) {
            this(m, n, k, lda, ldb, ldc, alpha, beta, false, 1, 0, 0, 0);
        }
        /** Densely packed matrices: {@code lda = k}, {@code ldb = ldc = n}. */
        public static Shape of(int m, int n, int k, float alpha, float beta) { return new Shape(m, n, k, k, n, n, alpha, beta); }
        /** Densely packed, B stored {@code n x k}: {@code lda = ldb = k}, {@code ldc = n}. {@code C = alpha A B^T + beta C}. */
        public static Shape ofTransposedB(int m, int n, int k, float alpha, float beta) {
            return new Shape(m, n, k, k, k, n, alpha, beta, true, 1, 0, 0, 0);
        }
        /** This shape as a batch of densely packed matrices, one after another. */
        public Shape batched(int count) {
            return withBatch(count, (long) m * lda, (long) (transposeB ? n : k) * ldb, (long) m * ldc);
        }
        /** This shape as a batch with the given strides. */
        public Shape withBatch(int count, long strideA, long strideB, long strideC) {
            return new Shape(m, n, k, lda, ldb, ldc, alpha, beta, transposeB, count, strideA, strideB, strideC);
        }
        /** Elements A, B and C span. */
        public long aElements() { return (batch - 1) * strideA + (long) (m - 1) * lda + k; }
        public long bElements() { return (batch - 1) * strideB + (long) ((transposeB ? n : k) - 1) * ldb + (transposeB ? k : n); }
        public long cElements() { return (batch - 1) * strideC + (long) (m - 1) * ldc + n; }
        /** The bytes A, B and an FP32 C must hold. */
        public long aBytes() { return 2 * aElements(); }
        public long bBytes() { return 2 * bElements(); }
        public long cBytes() { return 4 * cElements(); }
        /** The bytes C must hold for {@code output}. */
        public long cBytes(Precision output) { return output.bytes() * cElements(); }
        /** Floating-point operations: {@code 2 m n k} per multiply. */
        public double flops() { return 2.0 * m * n * k * batch; }
        /** Index of B's element at row {@code q} (of k) and column {@code j} (of n) in matrix {@code z}. */
        long bIndex(int z, int q, int j) { return z * strideB + (transposeB ? (long) j * ldb + q : (long) q * ldb + j); }
    }

    /**
     * The algorithms by shape (1.7.0). Costs are relative GPU times measured on an Apple M4 Max (the
     * {@code -Dgemm.shapes=true} sweep): the matrix kernel's 64 x 64 tiles waste most of their work when {@code m < 64},
     * where TILED runs 1.5 to 1.9 times faster, except that with B transposed the matrix kernel still wins up to
     * {@code m = 16}; from {@code m = 64} the matrix kernel runs 1.2 to 2.5 times faster.
     */
    public static final ShapeRegistry<Shape, Algorithm> REGISTRY = ShapeRegistry.<Shape, Algorithm>builder()
            .add("gemm.tiled", Algorithm.TILED, (d, s) -> supports(d, Algorithm.TILED), (d, s) -> 1.0,
                    (d, s) -> descriptor(d, Algorithm.TILED, Precision.FP32))
            .add("gemm.simdgroup", Algorithm.SIMDGROUP_MATRIX, (d, s) -> supports(d, Algorithm.SIMDGROUP_MATRIX),
                    (d, s) -> s.m() >= 64 ? 0.5 : s.transposeB() && s.m() <= 16 ? 0.75 : 1.5,
                    (d, s) -> descriptor(d, Algorithm.SIMDGROUP_MATRIX, Precision.FP32))
            .build();

    private final GpuDevice device;
    private final GpuComputePipeline pipeline;
    /** With shape dispatch (the default create on a device with both algorithms): the TILED pipeline; else null. */
    private final GpuComputePipeline tiled;
    private final Algorithm algorithm;
    private final Precision output;
    private final ParameterLayout layout;

    private GemmF16(GpuDevice device, GpuComputePipeline pipeline, GpuComputePipeline tiled, Algorithm algorithm, Precision output) {
        this.device = device; this.pipeline = pipeline; this.tiled = tiled; this.algorithm = algorithm; this.output = output;
        this.layout = ParameterLayout.of(KERNELS, kernel(output));   // the simdgroup kernel shares the block
    }

    /** As {@link #create(GpuDevice, Precision)} with an FP32 C. */
    public static GemmF16 create(GpuDevice device) { return create(device, Precision.FP32); }

    /**
     * The fastest GEMM for each shape, with C of type {@code output}: where the device has both algorithms, both are
     * compiled and {@link #encode} picks one per shape from {@link #REGISTRY} ({@link #algorithm(Shape)}). In
     * reproducible mode ({@code KitOptions}) it is TILED only.
     */
    public static GemmF16 create(GpuDevice device, Precision output) {
        Algorithm best = fastest(device);
        if (best == Algorithm.TILED) return create(device, Algorithm.TILED, output);
        var primary = device.pipeline(descriptor(device, best, output));
        try {
            return new GemmF16(device, primary, device.pipeline(descriptor(device, Algorithm.TILED, output)), best, output);
        } catch (RuntimeException e) {
            primary.close();
            throw e;
        }
    }

    /** Compiles {@code algorithm} with an FP32 C. */
    public static GemmF16 create(GpuDevice device, Algorithm algorithm) { return create(device, algorithm, Precision.FP32); }

    /** Compiles {@code algorithm} with C of type {@code output}; throws {@code UNSUPPORTED} where the device lacks it. */
    public static GemmF16 create(GpuDevice device, Algorithm algorithm, Precision output) {
        return new GemmF16(device, device.pipeline(descriptor(device, algorithm, output)), null, algorithm, output);
    }

    private static PipelineDescriptor descriptor(GpuDevice device, Algorithm algorithm, Precision output) {
        var types = Set.of(GpuCapabilities.FP32, GpuCapabilities.FP16);
        var accesses = List.of(Access.READ, Access.READ, Access.READ_WRITE);
        PipelineDescriptor descriptor = switch (algorithm) {
            case TILED -> KERNELS.pipeline(kernel(output), accesses, types, Set.of());
            case SIMDGROUP_MATRIX -> {
                if (device.adapter().backend() != Backend.METAL)
                    throw new ComputeException(ComputeException.Code.UNSUPPORTED, "SIMD-group matrix GEMM needs native Metal");
                List<BindingLayout.BufferBinding> b = List.of(new BindingLayout.BufferBinding(0, Access.READ, 4),
                        new BindingLayout.BufferBinding(1, Access.READ, 4), new BindingLayout.BufferBinding(2, Access.READ_WRITE, 4));
                yield new PipelineDescriptor(simdgroupBundle(), kernel(output) + "_simdgroup",
                        new BindingLayout(b, KERNELS.parameterBytes(kernel(output))), new Dimensions(128, 1, 1), Map.of(), types,
                        Set.of(GpuCapabilities.SUBGROUP_MATRIX_F16));
            }
        };
        return descriptor;
    }

    private static String kernel(Precision output) { return output == Precision.FP32 ? "gemm_f16" : "gemm_f16_h"; }

    /** The fastest algorithm, or TILED in reproducible mode ({@link KitOptions#reproducible()}). */
    private static Algorithm fastest(GpuDevice device) {
        if (KitOptions.current().reproducible()) return Algorithm.TILED;
        return supports(device, Algorithm.SIMDGROUP_MATRIX) ? Algorithm.SIMDGROUP_MATRIX : Algorithm.TILED;
    }

    /**
     * The accuracy contract of {@code algorithm} writing {@code output}. TILED is fixed-order: bit for bit
     * {@link #reference} (FP32 C) or {@link #referenceF16} (FP16 C) on every device. SIMDGROUP_MATRIX sums in the
     * hardware's order within the same bound. @since 1.7.0
     */
    public static AccuracyContract contract(Algorithm algorithm, Precision output) {
        String bound = "before alpha and beta, |C - exact| <= k 2^-24 sum |a b|"
                + (output == Precision.FP16 ? "; then one rounding to half" : "");
        String name = "GemmF16 " + algorithm + ", " + output + " C";
        var inputs = List.of("FP16", "FP16");
        String out = output + ", fl(fl(alpha acc) + fl(beta c))" + (output == Precision.FP16 ? " rounded once to half" : "");
        if (algorithm == Algorithm.TILED) {
            return new AccuracyContract(name, inputs, "FP32, in order of k", out, AccuracyContract.Order.FIXED,
                    new AccuracyContract.Bound(AccuracyContract.Bound.Kind.EXACT, 0, "bit for bit the reference; " + bound),
                    output == Precision.FP32 ? "GemmF16.reference" : "GemmF16.referenceF16");
        }
        return new AccuracyContract(name, inputs, "FP32, the hardware's order", out, AccuracyContract.Order.DEVICE,
                AccuracyContract.Bound.formula(bound), "");
    }

    /** This instance's accuracy contract. @since 1.7.0 */
    public AccuracyContract contract() { return contract(algorithm, output); }

    /** Whether {@code device} can run {@code algorithm}. */
    public static boolean supports(GpuDevice device, Algorithm algorithm) {
        var c = device.capabilities();
        if (!c.dataTypes().contains(GpuCapabilities.FP16)) return false;
        return algorithm == Algorithm.TILED || device.adapter().backend() == Backend.METAL
                && c.shaderFeatures().contains(GpuCapabilities.SUBGROUP_MATRIX_F16);
    }

    private static KernelBundle simdgroupBundle() {
        try (var in = GemmF16.class.getResourceAsStream("metal/gemm_f16_simdgroup.metal")) {
            if (in == null) throw new IllegalStateException("missing metal/gemm_f16_simdgroup.metal");
            byte[] source = in.readAllBytes();
            return new KernelBundle("dynamisgpu.primitives.gemm_f16_simdgroup", "2",
                    Map.of(Backend.METAL, new KernelBundle.Executable(KernelBundle.Format.MSL_SOURCE, source)));
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    /** The algorithm this instance runs. */
    public Algorithm algorithm() { return algorithm; }

    /** The algorithm {@link #encode} runs for {@code s}: with shape dispatch, the registry's choice. @since 1.7.0 */
    public Algorithm algorithm(Shape s) { return tiled == null ? algorithm : REGISTRY.select(device, s); }

    /** The element type of C. */
    public Precision output() { return output; }

    /** Records {@code C = alpha * A * B + beta * C} (for each matrix of the batch). */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer a, GpuBuffer b, GpuBuffer c, Shape s) {
        Checks.range(a, 0, s.aElements(), 2, "A"); Checks.range(b, 0, s.bElements(), 2, "B");
        Checks.range(c, 0, s.cElements(), output.bytes(), "C");
        byte[] params = layout.block().u32("m", s.m()).u32("n", s.n()).u32("k", s.k()).u32("lda", s.lda()).u32("ldb", s.ldb())
                .u32("ldc", s.ldc()).f32("alpha", s.alpha()).f32("beta", s.beta()).bool("transB", s.transposeB())
                .u32("strideA", s.strideA()).u32("strideB", s.strideB()).u32("strideC", s.strideC()).bytes();
        var groups = new Dimensions(Math.ceilDiv(s.n(), TILE), Math.ceilDiv(s.m(), TILE), s.batch());
        Algorithm run = algorithm(s);
        var p = run == algorithm ? pipeline : tiled;
        return encoder.dispatch(p, Map.of(0, a, 1, b, 2, c), params, groups, run == Algorithm.TILED ? "gemm.tiled" : "gemm.simdgroup");
    }

    private static float product(short[] a, short[] b, Shape s, int z, int i, int j) {
        float acc = 0;
        for (int q = 0; q < s.k(); q++) {
            acc += Float.float16ToFloat(a[(int) (z * s.strideA() + (long) i * s.lda() + q)])
                    * Float.float16ToFloat(b[(int) s.bIndex(z, q, j)]);
        }
        return acc;
    }

    /**
     * The CPU result of the {@link Algorithm#TILED} kernel with an FP32 C, bit for bit:
     * {@code C = fl(fl(alpha acc) + fl(beta c))}, where {@code acc} sums the products in order of {@code k} in FP32,
     * from halves given as raw bits.
     */
    public static void reference(short[] a, short[] b, float[] c, Shape s) {
        for (int z = 0; z < s.batch(); z++) {
            for (int i = 0; i < s.m(); i++) {
                for (int j = 0; j < s.n(); j++) {
                    float acc = product(a, b, s, z, i, j);
                    int idx = (int) (z * s.strideC() + (long) i * s.ldc() + j);
                    c[idx] = s.beta() == 0 ? s.alpha() * acc : s.alpha() * acc + s.beta() * c[idx];
                }
            }
        }
    }

    /** As {@link #reference}, with an FP16 C (raw bits): the FP32 result rounded once to the nearest half. */
    public static void referenceF16(short[] a, short[] b, short[] c, Shape s) {
        for (int z = 0; z < s.batch(); z++) {
            for (int i = 0; i < s.m(); i++) {
                for (int j = 0; j < s.n(); j++) {
                    float acc = product(a, b, s, z, i, j);
                    int idx = (int) (z * s.strideC() + (long) i * s.ldc() + j);
                    float r = s.beta() == 0 ? s.alpha() * acc : s.alpha() * acc + s.beta() * Float.float16ToFloat(c[idx]);
                    c[idx] = Float.floatToFloat16(r);
                }
            }
        }
    }

    /** The device this multiply runs on. */
    public GpuDevice device() { return device; }

    @Override public void close() { PipelineSet.closeAll(pipeline, tiled); }
}
