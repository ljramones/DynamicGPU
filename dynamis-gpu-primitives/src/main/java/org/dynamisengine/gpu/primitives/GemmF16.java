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
 * Half-precision matrix multiply: {@code C = alpha * A * B + beta * C}, with A ({@code m x k}, or {@code k x m} when
 * transposed) and B ({@code k x n}, or {@code n x k} when transposed) in FP16 and C ({@code m x n}) in FP32 or FP16,
 * all row-major, one multiply or a batch, with an optional fused {@link Epilogue} (bias, activation, gate, residual;
 * 1.8.0). It is the dense layer of a neural network ({@code x W^T} with transposed B), with a batch the per-head
 * products of attention, and with transposed A the weight gradient ({@code dY^T X}).
 * <p>
 * <b>Column-major operands.</b> There are no column-major kernels: a column-major product is the row-major product
 * {@code C^T = B^T A^T}, which {@link Shape#columnMajor} describes. Encode it with A and B swapped.
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
        SIMDGROUP_MATRIX,
        /**
         * The skinny GEMM for {@code m <= 16} ({@link #SKINNY_MAX_M}), the decode step of an LLM: B is streamed once
         * with every row of A, so at {@code m = 1} it is limited by memory bandwidth. Every device with FP16; the
         * same bits everywhere, bit for bit with {@link #referenceSkinny}, which sums in a fixed split order rather
         * than in order of k. A and B are bound at offsets that are multiples of 4 bytes. @since 1.8.0
         */
        SKINNY,
        /**
         * Small multiplies (such as the layers of a game-AI network, usually as a batch): 16 x 16 tiles, one element
         * per thread, where TILED's 64 x 64 tiles would be mostly padding or too few to fill the GPU. Every device with
         * FP16; sums in order of k like TILED, so bit for bit {@link #reference}. Any shape runs; the registry picks
         * it by estimated time ({@link Costs}). @since 1.8.0
         */
        SMALL
    }

    static final int SMALL_TILE = 16;

    /**
     * The element type of A and B (1.8.0). Every algorithm takes FP16 and FP32; all but SIMDGROUP_MATRIX take BF16.
     * FP16 and BF16 widen exactly to FP32 in the kernel, and their products are exact in FP32, so the fixed-order
     * algorithms are bit for bit their references everywhere. FP32 products round, so each step is one fused
     * multiply-add: bit for bit the references (which use {@code Math.fma}) on devices that report
     * {@link GpuCapabilities#FUSED_FMA}, within the same bound elsewhere. BF16 is never narrowed to FP16, whose
     * exponent range is smaller. Devices that flush FP32 subnormals (Apple GPUs) flush BF16 subnormal inputs too.
     */
    public enum Input {
        FP16(2, "f16"), BF16(2, "bf16"), FP32(4, "f32");
        private final int bytes;
        private final String suffix;
        Input(int bytes, String suffix) { this.bytes = bytes; this.suffix = suffix; }
        /** Bytes per element. */
        public int bytes() { return bytes; }
        /** Whether a product of two elements is exact in FP32, so fma and a multiply then an add agree. */
        public boolean exactProducts() { return this != FP32; }
    }

    /** Reads element {@code index} of A or B as FP32, exactly, for the CPU references. @since 1.8.0 */
    @FunctionalInterface
    public interface Elements {
        float at(long index);
        /** Halves given as raw bits. */
        static Elements fp16(short[] h) { return i -> Float.float16ToFloat(h[Math.toIntExact(i)]); }
        /** BF16 values given as raw bits (the upper half of an FP32). */
        static Elements bf16(short[] b) { return i -> bf16ToFloat(b[Math.toIntExact(i)]); }
        static Elements fp32(float[] f) { return i -> f[Math.toIntExact(i)]; }
    }

    /** The FP32 value of BF16 bits: exact. @since 1.8.0 */
    public static float bf16ToFloat(short bits) { return Float.intBitsToFloat((bits & 0xFFFF) << 16); }

    /** {@code value} rounded to the nearest BF16 (ties to even), as raw bits; NaN stays a quiet NaN. @since 1.8.0 */
    public static short floatToBf16(float value) {
        int x = Float.floatToRawIntBits(value);
        if (Float.isNaN(value)) return (short) ((x >>> 16) | 0x40);
        return (short) ((x + 0x7FFF + ((x >>> 16) & 1)) >>> 16);
    }

    /** The largest {@code m} {@link Algorithm#SKINNY} computes. @since 1.8.0 */
    public static final int SKINNY_MAX_M = 16;
    /** The skinny kernel's lanes per column: B transposed, and not (the summation order, see {@link #referenceSkinny}). */
    static final int SKINNY_LANES_T = 32, SKINNY_LANES_N = 16, SKINNY_CHUNK_T = 8, SKINNY_CHUNK_N = 2;
    /** Columns a group of the skinny kernel computes: B transposed, and not. */
    static final int SKINNY_COLUMNS_T = 16, SKINNY_COLUMNS_N = 64;

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
     * @param transposeA A is stored {@code k x m} ({@code lda} at least {@code m}): {@code C = A^T B}, the weight
     *        gradient of a linear layer (1.8.0)
     */
    public record Shape(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, boolean transposeB,
            int batch, long strideA, long strideB, long strideC, boolean transposeA) {
        public Shape {
            if (m <= 0 || n <= 0 || k <= 0) throw new IllegalArgumentException("dimensions must be positive: " + m + "x" + n + "x" + k);
            if (lda < (transposeA ? m : k) || ldc < n || ldb < (transposeB ? k : n))
                throw new IllegalArgumentException("leading dimension smaller than a row: lda=" + lda + " ldb=" + ldb + " ldc=" + ldc);
            if (batch < 1) throw new IllegalArgumentException("batch must be positive: " + batch);
            for (long st : new long[] {strideA, strideB, strideC})
                if (st < 0 || st > MAX_INDEX) throw new IllegalArgumentException("stride out of range: " + st);
            if (batch > 1 && strideC < (long) (m - 1) * ldc + n)
                throw new IllegalArgumentException("C matrices overlap: strideC " + strideC + " < " + ((long) (m - 1) * ldc + n));
        }
        /** A not transposed. */
        public Shape(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, boolean transposeB,
                int batch, long strideA, long strideB, long strideC) {
            this(m, n, k, lda, ldb, ldc, alpha, beta, transposeB, batch, strideA, strideB, strideC, false);
        }
        /** One multiply with B not transposed. */
        public Shape(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta) {
            this(m, n, k, lda, ldb, ldc, alpha, beta, false, 1, 0, 0, 0);
        }
        /** Densely packed, A stored {@code k x m}: {@code lda = m}, {@code ldb = ldc = n}. {@code C = alpha A^T B + beta C}. @since 1.8.0 */
        public static Shape ofTransposedA(int m, int n, int k, float alpha, float beta) {
            return new Shape(m, n, k, m, n, n, alpha, beta, false, 1, 0, 0, 0, true);
        }
        /**
         * The row-major shape of a column-major product (BLAS's convention): {@code C = alpha op(A) op(B) + beta C} with
         * A, B and C column-major and leading dimensions {@code lda}, {@code ldb}, {@code ldc}, computed as the
         * row-major {@code C^T = op(B)^T op(A)^T}. Encode it with the buffers of A and B swapped: the returned shape's
         * A is the column-major B. @since 1.8.0
         */
        public static Shape columnMajor(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta,
                boolean transposeA, boolean transposeB) {
            // column-major X (r x c, ld) is row-major X^T (c x r, ld): C^T (n x m) = op(B)^T (n x k) op(A)^T (k x m)
            return new Shape(n, m, k, ldb, lda, ldc, alpha, beta, transposeA, 1, 0, 0, 0, transposeB);
        }
        /** Densely packed matrices: {@code lda = k}, {@code ldb = ldc = n}. */
        public static Shape of(int m, int n, int k, float alpha, float beta) { return new Shape(m, n, k, k, n, n, alpha, beta); }
        /** Densely packed, B stored {@code n x k}: {@code lda = ldb = k}, {@code ldc = n}. {@code C = alpha A B^T + beta C}. */
        public static Shape ofTransposedB(int m, int n, int k, float alpha, float beta) {
            return new Shape(m, n, k, k, k, n, alpha, beta, true, 1, 0, 0, 0);
        }
        /** This shape as a batch of densely packed matrices, one after another. */
        public Shape batched(int count) {
            return withBatch(count, (long) (transposeA ? k : m) * lda, (long) (transposeB ? n : k) * ldb, (long) m * ldc);
        }
        /** This shape as a batch with the given strides. */
        public Shape withBatch(int count, long strideA, long strideB, long strideC) {
            return new Shape(m, n, k, lda, ldb, ldc, alpha, beta, transposeB, count, strideA, strideB, strideC, transposeA);
        }
        /** Elements A, B and C span. */
        public long aElements() { return (batch - 1) * strideA + (transposeA ? (long) (k - 1) * lda + m : (long) (m - 1) * lda + k); }
        public long bElements() { return (batch - 1) * strideB + (long) ((transposeB ? n : k) - 1) * ldb + (transposeB ? k : n); }
        public long cElements() { return (batch - 1) * strideC + (long) (m - 1) * ldc + n; }
        /** The bytes A, B and an FP32 C must hold. */
        public long aBytes() { return 2 * aElements(); }
        public long bBytes() { return 2 * bElements(); }
        public long cBytes() { return 4 * cElements(); }
        /** The bytes A and B must hold for {@code input}. @since 1.8.0 */
        public long aBytes(Input input) { return input.bytes() * aElements(); }
        public long bBytes(Input input) { return input.bytes() * bElements(); }
        /** The bytes C must hold for {@code output}. */
        public long cBytes(Precision output) { return output.bytes() * cElements(); }
        /** Floating-point operations: {@code 2 m n k} per multiply. */
        public double flops() { return 2.0 * m * n * k * batch; }
        /** Index of B's element at row {@code q} (of k) and column {@code j} (of n) in matrix {@code z}. */
        long bIndex(int z, int q, int j) { return z * strideB + (transposeB ? (long) j * ldb + q : (long) q * ldb + j); }
        /** Index of A's element at row {@code i} (of m) and column {@code q} (of k) in matrix {@code z}. */
        long aIndex(int z, int i, int q) { return z * strideA + (transposeA ? (long) q * lda + i : (long) i * lda + q); }
        /** Index of C's element at row {@code i} and column {@code j} in matrix {@code z}. */
        long cIndex(int z, int i, int j) { return z * strideC + (long) i * ldc + j; }
    }

    /**
     * What is applied to each element of the product before its one rounding to the output type, in FP32 and in this
     * order (1.8.0):
     * <ol>
     * <li>{@code r = fl(fl(alpha acc) + fl(beta c))}, as without an epilogue;</li>
     * <li>{@code r + bias}: per column ({@code bias[j]}, the bias of a linear layer) or per row ({@code bias[i]}),
     * shared by every matrix of a batch;</li>
     * <li>the activation: ReLU ({@code Math.max(r, 0)}, so NaN stays NaN), GELU in its tanh form
     * {@code 0.5 r (1 + tanh(sqrt(2/pi) (r + 0.044715 r^3)))}, or SiLU {@code r / (1 + exp(-r))};</li>
     * <li>{@code r * gate}, the gated linear unit: with SiLU and the up projection as gate, SwiGLU ({@link #swiGlu});</li>
     * <li>{@code r + residual}.</li>
     * </ol>
     * Gate and residual are laid out as C ({@code ldc}, {@code strideC}); bias, gate and residual have C's element
     * type. Bias, ReLU, gate and residual round as written, so the fixed-order algorithms stay bit for bit with the
     * references; GELU and SiLU use the device's {@code exp}, so results are within a bound instead ({@link #exact}).
     */
    public record Epilogue(Bias bias, Activation activation, boolean gated, boolean residual) {
        /** Where a bias is added. */
        public enum Bias { NONE, COLUMN, ROW }
        /** The activation function. */
        public enum Activation { NONE, RELU, GELU, SILU }
        /** No epilogue. */
        public static final Epilogue NONE = new Epilogue(Bias.NONE, Activation.NONE, false, false);
        public Epilogue {
            Checks.notNull(bias, "bias");
            Checks.notNull(activation, "activation");
        }
        /** SiLU then the gate: {@code silu(x W_gate^T) * (x W_up^T)} with the up projection's result as gate. */
        public static Epilogue swiGlu() { return NONE.withActivation(Activation.SILU).withGate(); }
        public Epilogue withBias(Bias b) { return new Epilogue(b, activation, gated, residual); }
        public Epilogue withActivation(Activation a) { return new Epilogue(bias, a, gated, residual); }
        public Epilogue withGate() { return new Epilogue(bias, activation, true, residual); }
        public Epilogue withResidual() { return new Epilogue(bias, activation, gated, true); }
        /** Whether every step rounds as written (no GELU or SiLU), so fixed-order algorithms stay bit for bit. */
        public boolean exact() { return activation == Activation.NONE || activation == Activation.RELU; }
    }

    /**
     * One multiply of a grouped GEMM ({@link #encodeGrouped}; 1.8.0): its sizes, leading dimensions, and element
     * offsets into the shared A, B and C (also gate and residual) buffers and into bias. On the GPU it is a record of
     * {@link #BYTES} bytes, 12 uint32 in native order: {@code m, n, k, lda, ldb, ldc, offsetA, offsetB, offsetC,
     * offsetBias, 0, 0}, so a kernel (a mixture-of-experts router, say) can write the table itself. Sizes may be 0
     * (an expert with no tokens).
     */
    public record Problem(int m, int n, int k, int lda, int ldb, int ldc, long offsetA, long offsetB, long offsetC, long offsetBias) {
        /** Bytes of one record. */
        public static final int BYTES = 48;
        public Problem {
            if (m < 0 || n < 0 || k < 0 || lda < 0 || ldb < 0 || ldc < 0)
                throw new IllegalArgumentException("sizes must not be negative: " + m + "x" + n + "x" + k + " lda=" + lda + " ldb=" + ldb + " ldc=" + ldc);
            for (long o : new long[] {offsetA, offsetB, offsetC, offsetBias})
                if (o < 0 || o > MAX_INDEX) throw new IllegalArgumentException("offset out of range: " + o);
        }
        /** Densely packed operands at the given offsets: leading dimensions from the layouts. */
        public static Problem dense(int m, int n, int k, boolean transposeA, boolean transposeB, long offsetA, long offsetB,
                long offsetC, long offsetBias) {
            return new Problem(m, n, k, transposeA ? m : k, transposeB ? k : n, n, offsetA, offsetB, offsetC, offsetBias);
        }
        /** The GPU records of {@code problems}, in native byte order. */
        public static byte[] encode(List<Problem> problems) {
            var out = java.nio.ByteBuffer.allocate(BYTES * problems.size()).order(java.nio.ByteOrder.nativeOrder());
            for (var q : problems) {
                out.putInt(q.m).putInt(q.n).putInt(q.k).putInt(q.lda).putInt(q.ldb).putInt(q.ldc).putInt((int) q.offsetA)
                        .putInt((int) q.offsetB).putInt((int) q.offsetC).putInt((int) q.offsetBias).putInt(0).putInt(0);
            }
            return out.array();
        }
    }

    /**
     * What every problem of a grouped GEMM shares (1.8.0): their number, bounds on their {@code m} and {@code n}
     * (which size the dispatch; a problem beyond them is reported as an error), the scalars and the layouts.
     */
    public record Grouped(int count, int maxM, int maxN, float alpha, float beta, boolean transposeA, boolean transposeB) {
        /** The most problems one dispatch takes (the minimum group count of a dispatch's third dimension). */
        public static final int MAX_COUNT = 65535;
        public Grouped {
            if (count < 1 || count > MAX_COUNT) throw new IllegalArgumentException("count out of range: " + count);
            if (maxM < 1 || maxN < 1) throw new IllegalArgumentException("maxM and maxN must be positive: " + maxM + ", " + maxN);
        }
    }

    /**
     * Estimated GPU time in microseconds of each algorithm for a shape, the shape registry's costs (1.8.0). Fitted to
     * the {@code -Dgemm.shapes=true} and {@code -Dgemm.decode=true} sweeps on an Apple M4 Max (native Metal and
     * MoltenVK agree): on all 118 measured shapes (46 from 1.8.0, 72 from the 1.15 Phase 0 probe) it picks the fastest
     * algorithm or one within 1.2 times of it, with the matrix kernel and without
     * ({@code GemmF16Test.costModelPicksTheMeasuredFastest}):
     * <ul>
     * <li>TILED runs 5.5 TFLOP/s over its 64-padded work when the GPU is full, which takes about 100 of its groups:
     * a 128^3 multiply (4 groups) runs at 4% of that;</li>
     * <li>SMALL runs 2.15 TFLOP/s over its 16-padded work, full from about 190 groups; it wins wherever TILED's tiles
     * are mostly padding (m or n up to 32) or too few to fill the GPU;</li>
     * <li>SKINNY streams B at 400 GB/s with B transposed (300 without) when full (256 groups of 16 columns; 64 of 64),
     * slower by 17% (45%) per extra row of A. With B transposed and fewer than 32 groups it is latency-bound instead,
     * at about 1.2 + 1.5 m + 0.00064 m k microseconds (1.15.0). It is offered only where it can win
     * ({@link #skinnyApplies}).</li>
     * <li>SIMDGROUP_MATRIX runs in 0.5 of TILED's time from m = 64, 0.75 for B transposed and m up to 16, else 1.5
     * (1.7.0).</li>
     * </ul>
     */
    static final class Costs {
        private Costs() { }
        static long pad(long x, int to) { return Math.ceilDiv(x, to) * to; }
        static double fill(double groups, double full) { return Math.min(1, groups / full); }
        static double tiled(Shape s) {
            double groups = (double) Math.ceilDiv(s.m(), TILE) * Math.ceilDiv(s.n(), TILE) * s.batch();
            return 2.0 * pad(s.m(), TILE) * pad(s.n(), TILE) * s.k() * s.batch() / (5.5e6 * fill(groups, 100));
        }
        static double small(Shape s) {
            double groups = (double) Math.ceilDiv(s.m(), SMALL_TILE) * Math.ceilDiv(s.n(), SMALL_TILE) * s.batch();
            return 2.0 * pad(s.m(), SMALL_TILE) * pad(s.n(), SMALL_TILE) * s.k() * s.batch() / (2.15e6 * fill(groups, 190));
        }
        static double skinny(Shape s) {
            double groups = (double) Math.ceilDiv(s.n(), s.transposeB() ? SKINNY_COLUMNS_T : SKINNY_COLUMNS_N) * s.batch();
            if (s.transposeB() && groups < 32) {
                // too few groups to stream: the larger of streaming B at the full rate and the groups' latency, which
                // grows with m k (1.15.0; fitted to one- and two-group products in the Phase 0 probe: 4 to 16 rows, k
                // 2048 to 8192). From 32 groups the bandwidth estimate below holds, as it did before 1.15.0.
                double stream = s.bBytes() / 4e5 * (1 + 0.17 * (s.m() - 1));
                double latency = 1.2 + 1.5 * s.m() + 0.00064 * s.m() * s.k();
                return Math.max(stream, latency);
            }
            return s.bBytes() / (3e5 * fill(groups, 64)) * (1 + 0.45 * (s.m() - 1));
        }
        /**
         * Whether the registry offers SKINNY for {@code s}: m up to 16, A not transposed, and k at least 256. With a
         * shorter k the kernel is latency-bound, which the bandwidth estimate does not see: 8 x 300 x 200 with B^T took
         * twice SMALL's time. With B not transposed and more than one row of A it also needs 32 groups of columns: below
         * that SMALL is 1.2 to 4 times faster.
         * <p>
         * Until 1.15.0 the 32-group rule applied to every shape. That withheld SKINNY from small-n products with B
         * transposed, and from m = 1 with n up to 1024. There SKINNY is 1.3 to 25 times faster than SMALL (the 1.15
         * Phase 0 probe), and its cost estimate already ranks it first.
         */
        static boolean skinnyApplies(Shape s) {
            if (s.m() > SKINNY_MAX_M || s.transposeA() || s.k() < 256) return false;
            if (s.transposeB() || s.m() == 1) return true;
            return Math.ceilDiv(s.n(), SKINNY_COLUMNS_N) * (long) s.batch() >= 32;
        }
        static double simdgroup(Shape s) {
            return tiled(s) * (s.m() >= 64 ? 0.5 : s.transposeB() && s.m() <= 16 ? 0.75 : 1.5);
        }
    }

    /** The algorithms by shape: each supported one with its estimated time ({@link Costs}); 1.7.0, costs 1.8.0. */
    public static final ShapeRegistry<Shape, Algorithm> REGISTRY = ShapeRegistry.<Shape, Algorithm>builder()
            .add("gemm.tiled", Algorithm.TILED, (d, s) -> supports(d, Algorithm.TILED), (d, s) -> Costs.tiled(s),
                    (d, s) -> descriptor(d, Algorithm.TILED, Input.FP16, Precision.FP32))
            .add("gemm.simdgroup", Algorithm.SIMDGROUP_MATRIX, (d, s) -> supports(d, Algorithm.SIMDGROUP_MATRIX),
                    (d, s) -> Costs.simdgroup(s), (d, s) -> descriptor(d, Algorithm.SIMDGROUP_MATRIX, Input.FP16, Precision.FP32))
            .add("gemm.skinny", Algorithm.SKINNY, (d, s) -> supports(d, Algorithm.SKINNY) && Costs.skinnyApplies(s),
                    (d, s) -> Costs.skinny(s), (d, s) -> descriptor(d, Algorithm.SKINNY, Input.FP16, Precision.FP32))
            .add("gemm.small", Algorithm.SMALL, (d, s) -> supports(d, Algorithm.SMALL), (d, s) -> Costs.small(s),
                    (d, s) -> descriptor(d, Algorithm.SMALL, Input.FP16, Precision.FP32))
            .build();

    private final GpuDevice device;
    /** The compiled algorithms: one, or with shape dispatch every algorithm the device supports. */
    private final Map<Algorithm, GpuComputePipeline> pipelines;
    /** With SKINNY: its vector build, for aligned operands with B transposed. */
    private final GpuComputePipeline skinnyVector;
    /** The single algorithm, or with shape dispatch the fastest overall (as reported by {@link #algorithm()}). */
    private final Algorithm algorithm;
    private final boolean dispatch;
    private final Input input;
    private final Precision output;
    private final ParameterLayout layout;
    /** Binds the absent epilogue operands. */
    private PipelineSet placeholders;
    /** The grouped kernel, compiled on first use. */
    private GpuComputePipeline grouped;
    private ParameterLayout groupedLayout;

    private GemmF16(GpuDevice device, Map<Algorithm, GpuComputePipeline> pipelines, GpuComputePipeline skinnyVector,
            Algorithm algorithm, boolean dispatch, Input input, Precision output) {
        this.device = device; this.pipelines = pipelines; this.skinnyVector = skinnyVector;
        this.algorithm = algorithm; this.dispatch = dispatch; this.input = input; this.output = output;
        this.layout = ParameterLayout.of(KERNELS, kernel(output));   // every algorithm shares the block
    }

    /** As {@link #create(GpuDevice, Precision)} with an FP32 C. */
    public static GemmF16 create(GpuDevice device) { return create(device, Precision.FP32); }

    /**
     * The fastest GEMM for each shape, with C of type {@code output}: where the device has both algorithms, both are
     * compiled and {@link #encode} picks one per shape from {@link #REGISTRY} ({@link #algorithm(Shape)}). In
     * reproducible mode ({@code KitOptions}) it is TILED only.
     */
    public static GemmF16 create(GpuDevice device, Precision output) { return create(device, Input.FP16, output); }

    /**
     * As {@link #create(GpuDevice, Precision)} with A and B of type {@code input}: every algorithm the device runs for
     * that input is compiled, and {@link #encode} picks one per shape. For FP32 inputs the matrix algorithm is left
     * out: on an M4 Max its FP32 SIMD-group matrices ran at 2.7 to 3.0 TFLOP/s at 4096^3, against the tiled kernel's
     * 4.9 to 5.3 (create it explicitly to use it). @since 1.8.0
     */
    public static GemmF16 create(GpuDevice device, Input input, Precision output) {
        if (KitOptions.current().reproducible()) return create(device, Algorithm.TILED, input, output);
        var algorithms = java.util.Arrays.stream(Algorithm.values()).filter(a -> supports(device, a, input))
                .filter(a -> a != Algorithm.SIMDGROUP_MATRIX || input != Input.FP32).toList();
        if (algorithms.isEmpty()) return create(device, Algorithm.TILED, input, output);   // throws UNSUPPORTED
        return compile(device, algorithms, fastest(device, input), true, input, output);
    }

    /** Compiles {@code algorithms} (and SKINNY's vector build) all or none. */
    private static GemmF16 compile(GpuDevice device, List<Algorithm> algorithms, Algorithm primary, boolean dispatch, Input input,
            Precision output) {
        var descriptors = new java.util.ArrayList<PipelineDescriptor>();
        for (var a : algorithms) descriptors.add(descriptor(device, a, input, output));
        boolean skinny = algorithms.contains(Algorithm.SKINNY);
        if (skinny) descriptors.add(KERNELS.pipeline(kernel("gemm_skinny_v", input, output), ACCESSES, types(input, output), Set.of()));
        var set = PipelineSet.compile(device, descriptors);
        var pipelines = new java.util.EnumMap<Algorithm, GpuComputePipeline>(Algorithm.class);
        for (int i = 0; i < algorithms.size(); i++) pipelines.put(algorithms.get(i), set.get(i));
        return new GemmF16(device, pipelines, skinny ? set.get(algorithms.size()) : null, primary, dispatch, input, output);
    }

    /** A kernel's name: the family, then the input type unless FP16 (the tiled family always names it), then _h for an FP16 C. */
    static String kernel(String family, Input input, Precision output) {
        String in = family.equals("gemm") ? "_" + input.suffix : input == Input.FP16 ? "" : "_" + input.suffix;
        return family + in + (output == Precision.FP16 ? "_h" : "");
    }

    /** The data types a kernel needs: FP16 only for FP16 inputs or C. */
    private static Set<String> types(Input input, Precision output) {
        return input == Input.FP16 || output == Precision.FP16 ? Set.of(GpuCapabilities.FP32, GpuCapabilities.FP16) : Set.of(GpuCapabilities.FP32);
    }

    /** Compiles {@code algorithm} with an FP32 C. */
    public static GemmF16 create(GpuDevice device, Algorithm algorithm) { return create(device, algorithm, Precision.FP32); }

    /** Compiles {@code algorithm} with C of type {@code output}; throws {@code UNSUPPORTED} where the device lacks it. */
    public static GemmF16 create(GpuDevice device, Algorithm algorithm, Precision output) {
        return create(device, algorithm, Input.FP16, output);
    }

    /** Compiles {@code algorithm} for {@code input} and {@code output}; {@code UNSUPPORTED} where the device lacks it. @since 1.8.0 */
    public static GemmF16 create(GpuDevice device, Algorithm algorithm, Input input, Precision output) {
        if (algorithm == Algorithm.SIMDGROUP_MATRIX && input == Input.BF16)
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, "the SIMD-group matrix GEMM takes FP16 or FP32 inputs, not BF16");
        return compile(device, List.of(algorithm), algorithm, false, input, output);
    }

    /** A, B, C, then the epilogue's bias, gate and residual. */
    private static final List<Access> ACCESSES = List.of(Access.READ, Access.READ, Access.READ_WRITE, Access.READ, Access.READ, Access.READ);

    private static PipelineDescriptor descriptor(GpuDevice device, Algorithm algorithm, Input input, Precision output) {
        var types = types(input, output);
        var accesses = ACCESSES;
        PipelineDescriptor descriptor = switch (algorithm) {
            case TILED -> KERNELS.pipeline(kernel("gemm", input, output), accesses, types, Set.of());
            case SKINNY -> KERNELS.pipeline(kernel("gemm_skinny", input, output), accesses, types, Set.of());
            case SMALL -> KERNELS.pipeline(kernel("gemm_small", input, output), accesses, types, Set.of());
            case SIMDGROUP_MATRIX -> {
                if (device.adapter().backend() != Backend.METAL)
                    throw new ComputeException(ComputeException.Code.UNSUPPORTED, "SIMD-group matrix GEMM needs native Metal");
                var b = new java.util.ArrayList<BindingLayout.BufferBinding>();
                for (int i = 0; i < ACCESSES.size(); i++) b.add(new BindingLayout.BufferBinding(i, ACCESSES.get(i), 4));
                yield new PipelineDescriptor(simdgroupBundle(), kernel("gemm", input, output) + "_simdgroup",
                        new BindingLayout(b, KERNELS.parameterBytes(kernel(output))), new Dimensions(128, 1, 1), Map.of(), types,
                        Set.of(GpuCapabilities.SUBGROUP_MATRIX_F16));
            }
        };
        return descriptor;
    }

    private static String kernel(Precision output) { return output == Precision.FP32 ? "gemm_f16" : "gemm_f16_h"; }

    /** The fastest algorithm for large shapes, or TILED in reproducible mode ({@link KitOptions#reproducible()}). */
    private static Algorithm fastest(GpuDevice device, Input input) {
        if (KitOptions.current().reproducible()) return Algorithm.TILED;
        return input != Input.FP32 && supports(device, Algorithm.SIMDGROUP_MATRIX, input) ? Algorithm.SIMDGROUP_MATRIX : Algorithm.TILED;
    }

    /**
     * The accuracy contract of {@code algorithm} writing {@code output}. TILED is fixed-order: bit for bit
     * {@link #reference} (FP32 C) or {@link #referenceF16} (FP16 C) on every device. SIMDGROUP_MATRIX sums in the
     * hardware's order within the same bound. @since 1.7.0
     */
    public static AccuracyContract contract(Algorithm algorithm, Precision output) { return contract(algorithm, output, Epilogue.NONE); }

    /**
     * The accuracy contract of {@code algorithm} writing {@code output} through {@code epilogue}, which is applied in
     * FP32 before the output rounding. With an {@link Epilogue#exact} epilogue the fixed-order algorithms stay bit for
     * bit with {@link #reference(Algorithm, short[], short[], float[], Shape, Epilogue, float[], float[], float[])};
     * GELU and SiLU add the device's {@code exp} error, propagated through the activation. @since 1.8.0
     */
    public static AccuracyContract contract(Algorithm algorithm, Precision output, Epilogue epilogue) {
        return contract(algorithm, Input.FP16, output, epilogue);
    }

    /**
     * The accuracy contract of {@code algorithm} with A and B of type {@code input}. With FP32 inputs a fixed-order
     * algorithm is bit for bit its reference only where the device fuses fma ({@link GpuCapabilities#FUSED_FMA}), so
     * the contract states the bound, with exactness under that condition. @since 1.8.0
     */
    public static AccuracyContract contract(Algorithm algorithm, Input input, Precision output, Epilogue epilogue) {
        Checks.notNull(epilogue, "epilogue"); Checks.notNull(input, "input");
        String bound = "before alpha and beta, |C - exact| <= k 2^-24 sum |a b|"
                + (epilogue.equals(Epilogue.NONE) ? "" : "; then the epilogue in FP32 (" + epilogue + ")")
                + (epilogue.exact() ? "" : ", whose activation uses the device's exp")
                + (output == Precision.FP16 ? "; then one rounding to half" : "");
        String name = "GemmF16 " + algorithm + ", " + input + " A and B, " + output + " C" + (epilogue.equals(Epilogue.NONE) ? "" : ", " + epilogue);
        var inputs = List.of(input.name(), input.name());
        String out = output + ", epilogue(fl(fl(alpha acc) + fl(beta c)))" + (output == Precision.FP16 ? " rounded once to half" : "");
        String accumulation = switch (algorithm) {
            case TILED, SMALL -> "FP32, in order of k" + (input.exactProducts() ? "" : ", one fused multiply-add per step");
            case SKINNY -> "FP32, k dealt to 32 lanes in chunks of 8 (B^T) or 16 lanes in chunks of 2, each lane in order of k, then a fixed tree";
            case SIMDGROUP_MATRIX -> "FP32, the hardware's order";
        };
        if (algorithm == Algorithm.SIMDGROUP_MATRIX)
            return new AccuracyContract(name, inputs, accumulation, out, AccuracyContract.Order.DEVICE, AccuracyContract.Bound.formula(bound), "");
        String reference = epilogue.equals(Epilogue.NONE)
                ? algorithm != Algorithm.SKINNY ? (output == Precision.FP32 ? "GemmF16.reference" : "GemmF16.referenceF16")
                        : (output == Precision.FP32 ? "GemmF16.referenceSkinny" : "GemmF16.referenceSkinnyF16")
                : output == Precision.FP32 ? "GemmF16.reference(Algorithm, ...)" : "GemmF16.referenceF16(Algorithm, ...)";
        var b = epilogue.exact() && input.exactProducts()
                ? new AccuracyContract.Bound(AccuracyContract.Bound.Kind.EXACT, 0, "bit for bit the reference; " + bound)
                : epilogue.exact() ? AccuracyContract.Bound.formula(bound + "; bit for bit the reference where the device reports FUSED_FMA")
                : AccuracyContract.Bound.formula(bound);
        return new AccuracyContract(name, inputs, accumulation, out, AccuracyContract.Order.FIXED, b, reference);
    }

    /** This instance's accuracy contract. @since 1.7.0 */
    public AccuracyContract contract() { return contract(algorithm, input, output, Epilogue.NONE); }

    /** This instance's accuracy contract with {@code epilogue}. @since 1.8.0 */
    public AccuracyContract contract(Epilogue epilogue) { return contract(algorithm, input, output, epilogue); }

    /** The element type of A and B. @since 1.8.0 */
    public Input input() { return input; }

    /** Whether {@code device} can run {@code algorithm} with FP16 inputs. */
    public static boolean supports(GpuDevice device, Algorithm algorithm) { return supports(device, algorithm, Input.FP16); }

    /**
     * Whether {@code device} can run {@code algorithm} with {@code input} (and an FP32 C; an FP16 C also needs
     * {@link GpuCapabilities#FP16}). The matrix algorithm needs native Metal's SIMD-group matrices and takes FP16 or
     * FP32. @since 1.8.0
     */
    public static boolean supports(GpuDevice device, Algorithm algorithm, Input input) {
        var c = device.capabilities();
        boolean fp16 = c.dataTypes().contains(GpuCapabilities.FP16);
        if (input == Input.FP16 && !fp16) return false;
        if (algorithm != Algorithm.SIMDGROUP_MATRIX) return true;
        return input != Input.BF16 && fp16 && device.adapter().backend() == Backend.METAL
                && c.shaderFeatures().contains(GpuCapabilities.SUBGROUP_MATRIX_F16);
    }

    private static KernelBundle simdgroupBundle() {
        try (var in = GemmF16.class.getResourceAsStream("metal/gemm_f16_simdgroup.metal")) {
            if (in == null) throw new IllegalStateException("missing metal/gemm_f16_simdgroup.metal");
            byte[] source = in.readAllBytes();
            return new KernelBundle("dynamisgpu.primitives.gemm_f16_simdgroup", "3",
                    Map.of(Backend.METAL, new KernelBundle.Executable(KernelBundle.Format.MSL_SOURCE, source)));
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    /** The algorithm this instance runs: with shape dispatch, the fastest for large shapes. */
    public Algorithm algorithm() { return algorithm; }

    /** The algorithm {@link #encode} runs for {@code s}: with shape dispatch, the registry's choice. @since 1.7.0 */
    public Algorithm algorithm(Shape s) {
        if (!dispatch) return algorithm;
        for (var e : REGISTRY.ranked(device, s)) if (pipelines.containsKey(e.variant())) return e.variant();
        return Algorithm.TILED;
    }

    /** The element type of C. */
    public Precision output() { return output; }

    /** Records {@code C = alpha * A * B + beta * C} (for each matrix of the batch). */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer a, GpuBuffer b, GpuBuffer c, Shape s) {
        return encode(encoder, a, b, c, s, Epilogue.NONE, null, null, null);
    }

    /**
     * Records {@code C = epilogue(alpha * A * B + beta * C)} (for each matrix of the batch). {@code bias} ({@code n}
     * elements per column, {@code m} per row), {@code gate} and {@code residual} (laid out as C) have C's element type;
     * each is given exactly when the epilogue uses it, else null. @since 1.8.0
     */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer a, GpuBuffer b, GpuBuffer c, Shape s, Epilogue epilogue,
            GpuBuffer bias, GpuBuffer gate, GpuBuffer residual) {
        Checks.notNull(epilogue, "epilogue");
        Checks.range(a, 0, s.aElements(), input.bytes(), "A"); Checks.range(b, 0, s.bElements(), input.bytes(), "B");
        Checks.range(c, 0, s.cElements(), output.bytes(), "C");
        bias = operand(bias, epilogue.bias() != Epilogue.Bias.NONE, epilogue.bias() == Epilogue.Bias.ROW ? s.m() : s.n(), "bias");
        gate = operand(gate, epilogue.gated(), s.cElements(), "gate");
        residual = operand(residual, epilogue.residual(), s.cElements(), "residual");
        byte[] params = layout.block().u32("m", s.m()).u32("n", s.n()).u32("k", s.k()).u32("lda", s.lda()).u32("ldb", s.ldb())
                .u32("ldc", s.ldc()).f32("alpha", s.alpha()).f32("beta", s.beta()).bool("transB", s.transposeB())
                .u32("strideA", s.strideA()).u32("strideB", s.strideB()).u32("strideC", s.strideC()).bool("transA", s.transposeA())
                .u32("bias", epilogue.bias().ordinal()).u32("act", epilogue.activation().ordinal()).bool("gated", epilogue.gated())
                .bool("residual", epilogue.residual()).bytes();
        Algorithm run = algorithm(s);
        if (run == Algorithm.SKINNY) {
            boolean aligned = a.offset() % 4 == 0 && b.offset() % 4 == 0;
            if (!dispatch && s.m() > SKINNY_MAX_M) throw new IllegalArgumentException("SKINNY needs m <= " + SKINNY_MAX_M + ": " + s);
            if (!dispatch && s.transposeA()) throw new IllegalArgumentException("SKINNY does not transpose A: " + s);
            if (!dispatch && !aligned) throw new IllegalArgumentException("SKINNY needs A and B at offsets that are multiples of 4 bytes");
            if (!aligned) run = Algorithm.TILED;
        }
        var groups = switch (run) {
            case SKINNY -> new Dimensions(Math.ceilDiv(s.n(), s.transposeB() ? SKINNY_COLUMNS_T : SKINNY_COLUMNS_N), 1, s.batch());
            case SMALL -> new Dimensions(Math.ceilDiv(s.n(), SMALL_TILE), Math.ceilDiv(s.m(), SMALL_TILE), s.batch());
            default -> new Dimensions(Math.ceilDiv(s.n(), TILE), Math.ceilDiv(s.m(), TILE), s.batch());
        };
        var pipeline = run == Algorithm.SKINNY && vectorized(s, a, b) ? skinnyVector : pipelines.get(run);
        return encoder.dispatch(pipeline, Map.of(0, a, 1, b, 2, c, 3, bias, 4, gate, 5, residual), params, groups, label(run));
    }

    /**
     * Records a grouped GEMM: for each of {@code g.count()} problems in {@code problems} ({@link Problem} records,
     * which the GPU may have written), {@code C = epilogue(alpha op(A) op(B) + beta C)} at the problem's offsets into
     * the shared buffers, with the tiled kernel, so every problem is bit for bit {@link #referenceGrouped}. The
     * epilogue's operands are shared too (gate and residual laid out as C; bias at each problem's offset). A problem
     * that reaches outside a buffer, or beyond {@code maxM} or {@code maxN}, is skipped and fails the submission with
     * {@link ComputeException.Code#KERNEL_ERROR}. @since 1.8.0
     */
    public GpuCommandEncoder encodeGrouped(GpuCommandEncoder encoder, GpuBuffer a, GpuBuffer b, GpuBuffer c, GpuBuffer problems,
            Grouped g, Epilogue epilogue, GpuBuffer bias, GpuBuffer gate, GpuBuffer residual) {
        Checks.notNull(g, "grouped"); Checks.notNull(epilogue, "epilogue");
        Checks.range(problems, 0, (long) g.count() * Problem.BYTES / 4, 4, "problems");
        long cLength = c.bytes() / output.bytes();
        bias = operand(bias, epilogue.bias() != Epilogue.Bias.NONE, 1, "bias");
        gate = operand(gate, epilogue.gated(), 1, "gate");
        residual = operand(residual, epilogue.residual(), 1, "residual");
        if (epilogue.gated()) cLength = Math.min(cLength, gate.bytes() / output.bytes());
        if (epilogue.residual()) cLength = Math.min(cLength, residual.bytes() / output.bytes());
        if (grouped == null) {
            String kernel = kernel("gemm_grouped", input, output);
            var accesses = new java.util.ArrayList<>(ACCESSES);
            accesses.add(Access.READ); accesses.add(Access.READ_WRITE);
            grouped = device.pipeline(KERNELS.pipeline(kernel, accesses, types(input, output), Set.of()));
            groupedLayout = ParameterLayout.of(KERNELS, kernel);
        }
        byte[] params = groupedLayout.block().u32("m", g.maxM()).u32("n", g.maxN()).u32("k", 0).u32("lda", 0).u32("ldb", 0).u32("ldc", 0)
                .f32("alpha", g.alpha()).f32("beta", g.beta()).bool("transB", g.transposeB()).u32("strideA", 0).u32("strideB", 0)
                .u32("strideC", 0).bool("transA", g.transposeA()).u32("bias", epilogue.bias().ordinal())
                .u32("act", epilogue.activation().ordinal()).bool("gated", epilogue.gated()).bool("residual", epilogue.residual())
                .u32("count", g.count()).u32("aLength", Math.min(a.bytes() / input.bytes(), MAX_INDEX)).u32("bLength", Math.min(b.bytes() / input.bytes(), MAX_INDEX))
                .u32("cLength", Math.min(cLength, MAX_INDEX)).u32("biasLength", Math.min(bias.bytes() / output.bytes(), MAX_INDEX)).bytes();
        var groups = new Dimensions(Math.ceilDiv(g.maxN(), TILE), Math.ceilDiv(g.maxM(), TILE), g.count());
        // a fresh error buffer per dispatch: closing the handle keeps it alive for the recorded commands
        try (var errors = device.allocate(KernelErrors.descriptor().withLabel("gemm-grouped-errors"))) {
            encoder.resetErrors(errors);
            encoder.dispatch(grouped, Map.of(0, a, 1, b, 2, c, 3, bias, 4, gate, 5, residual, 6, problems, 7, errors), params, groups,
                    "gemm.grouped");
            encoder.checkErrors(errors);
        }
        return encoder;
    }

    /** {@code buffer}, checked to hold {@code elements} of C's type when {@code used}; else a placeholder. */
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

    /** Whether the skinny kernel's vector build applies: B transposed, every row on a multiple of 8 halves, 16-byte offsets. */
    private static boolean vectorized(Shape s, GpuBuffer a, GpuBuffer b) {
        return s.transposeB() && (s.k() | s.lda() | s.ldb() | s.strideA() | s.strideB()) % 8 == 0
                && a.offset() % 16 == 0 && b.offset() % 16 == 0;
    }

    static String label(Algorithm algorithm) {
        return switch (algorithm) {
            case TILED -> "gemm.tiled";
            case SIMDGROUP_MATRIX -> "gemm.simdgroup";
            case SKINNY -> "gemm.skinny";
            case SMALL -> "gemm.small";
        };
    }

    /** The fixed-order sum of {@code algorithm} for C[z][i][j], from halves given as raw bits. */
    private static float sum(Algorithm algorithm, Elements a, Elements b, Shape s, int z, int i, int j) {
        // one fused multiply-add per step, as the kernels: for FP16 and BF16 the products are exact, so it is acc + a b
        if (algorithm == Algorithm.TILED || algorithm == Algorithm.SMALL) {
            float acc = 0;
            for (int q = 0; q < s.k(); q++) acc = Math.fma(a.at(s.aIndex(z, i, q)), b.at(s.bIndex(z, q, j)), acc);
            return acc;
        }
        if (algorithm != Algorithm.SKINNY) throw new IllegalArgumentException(algorithm + " sums in the hardware's order: it has no reference");
        // element q goes to lane (q / chunk) % L; each lane sums in order of q; then lane t takes lane t + w, w = L/2 .. 1
        int lanes = s.transposeB() ? SKINNY_LANES_T : SKINNY_LANES_N, chunk = s.transposeB() ? SKINNY_CHUNK_T : SKINNY_CHUNK_N;
        float[] part = new float[lanes];
        for (int q = 0; q < s.k(); q++) {
            int l = (q / chunk) % lanes;
            part[l] = Math.fma(a.at(s.aIndex(z, i, q)), b.at(s.bIndex(z, q, j)), part[l]);
        }
        for (int w = lanes / 2; w > 0; w /= 2) for (int t = 0; t < w; t++) part[t] += part[t + w];
        return part[0];
    }

    /** Reads element {@code index} of an operand as a float. */
    interface Operand { float at(int index); }

    static final float GELU_SCALE = 0.7978845608f, GELU_CUBIC = 0.044715f;

    /** The epilogue in FP32, as gemm_common.slang writes it (GELU and SiLU with Math.exp, correctly rounded or nearly). */
    static float epilogue(float r, int i, int j, int idx, Epilogue e, Operand bias, Operand gate, Operand residual) {
        if (e.bias() == Epilogue.Bias.COLUMN) r = r + bias.at(j);
        else if (e.bias() == Epilogue.Bias.ROW) r = r + bias.at(i);
        switch (e.activation()) {
            case RELU -> r = Math.max(r, 0f);
            case GELU -> {
                float u = GELU_SCALE * (r + GELU_CUBIC * r * r * r);
                float t = 1f - 2f / ((float) Math.exp(2f * u) + 1f);
                r = 0.5f * r * (1f + t);
            }
            case SILU -> r = r / (1f + (float) Math.exp(-r));
            case NONE -> { }
        }
        if (e.gated()) r = r * gate.at(idx);
        if (e.residual()) r = r + residual.at(idx);
        return r;
    }

    private static Operand floats(float[] x) { return x == null ? i -> { throw new IllegalArgumentException("operand missing"); } : i -> x[i]; }
    private static Operand halves(short[] x) { return x == null ? i -> { throw new IllegalArgumentException("operand missing"); } : i -> Float.float16ToFloat(x[i]); }

    /**
     * The CPU result of a fixed-order {@code algorithm} (TILED or SKINNY) with an FP32 C and {@code epilogue}, bit for
     * bit where the epilogue is {@link Epilogue#exact}, else with {@code Math.exp} in place of the device's:
     * {@code C = epilogue(fl(fl(alpha acc) + fl(beta c)))}, with {@code acc} summed in the algorithm's order from
     * halves given as raw bits. Operands the epilogue does not use may be null. @since 1.8.0
     */
    public static void reference(Algorithm algorithm, short[] a, short[] b, float[] c, Shape s, Epilogue epilogue,
            float[] bias, float[] gate, float[] residual) {
        reference(algorithm, Elements.fp16(a), Elements.fp16(b), c, s, epilogue, bias, gate, residual);
    }

    /**
     * As {@link #reference(Algorithm, short[], short[], float[], Shape, Epilogue, float[], float[], float[])} for any
     * {@link Input}: A and B are read through {@code a} and {@code b}, and every step is one fused multiply-add.
     * @since 1.8.0
     */
    public static void reference(Algorithm algorithm, Elements a, Elements b, float[] c, Shape s, Epilogue epilogue,
            float[] bias, float[] gate, float[] residual) {
        for (int z = 0; z < s.batch(); z++) {
            for (int i = 0; i < s.m(); i++) {
                for (int j = 0; j < s.n(); j++) {
                    float acc = sum(algorithm, a, b, s, z, i, j);
                    int idx = (int) s.cIndex(z, i, j);
                    float r = s.beta() == 0 ? s.alpha() * acc : s.alpha() * acc + s.beta() * c[idx];
                    c[idx] = epilogue(r, i, j, idx, epilogue, floats(bias), floats(gate), floats(residual));
                }
            }
        }
    }

    /**
     * As {@link #reference(Algorithm, short[], short[], float[], Shape, Epilogue, float[], float[], float[])} with an
     * FP16 C and FP16 operands (raw bits): the FP32 result rounded once to the nearest half. @since 1.8.0
     */
    public static void referenceF16(Algorithm algorithm, short[] a, short[] b, short[] c, Shape s, Epilogue epilogue,
            short[] bias, short[] gate, short[] residual) {
        referenceF16(algorithm, Elements.fp16(a), Elements.fp16(b), c, s, epilogue, bias, gate, residual);
    }

    /** As {@link #referenceF16(Algorithm, short[], short[], short[], Shape, Epilogue, short[], short[], short[])} for any {@link Input}. @since 1.8.0 */
    public static void referenceF16(Algorithm algorithm, Elements a, Elements b, short[] c, Shape s, Epilogue epilogue,
            short[] bias, short[] gate, short[] residual) {
        for (int z = 0; z < s.batch(); z++) {
            for (int i = 0; i < s.m(); i++) {
                for (int j = 0; j < s.n(); j++) {
                    float acc = sum(algorithm, a, b, s, z, i, j);
                    int idx = (int) s.cIndex(z, i, j);
                    float r = s.beta() == 0 ? s.alpha() * acc : s.alpha() * acc + s.beta() * Float.float16ToFloat(c[idx]);
                    c[idx] = Float.floatToFloat16(epilogue(r, i, j, idx, epilogue, halves(bias), halves(gate), halves(residual)));
                }
            }
        }
    }

    /**
     * The CPU result of the {@link Algorithm#TILED} kernel with an FP32 C, bit for bit:
     * {@code C = fl(fl(alpha acc) + fl(beta c))}, where {@code acc} sums the products in order of {@code k} in FP32,
     * from halves given as raw bits.
     */
    public static void reference(short[] a, short[] b, float[] c, Shape s) {
        reference(Algorithm.TILED, a, b, c, s, Epilogue.NONE, null, null, null);
    }

    /** As {@link #reference}, with an FP16 C (raw bits): the FP32 result rounded once to the nearest half. */
    public static void referenceF16(short[] a, short[] b, short[] c, Shape s) {
        referenceF16(Algorithm.TILED, a, b, c, s, Epilogue.NONE, null, null, null);
    }

    /**
     * The CPU result of the {@link Algorithm#SKINNY} kernel with an FP32 C, bit for bit: as {@link #reference}, with
     * the sum dealt to lanes in a fixed order: element {@code q} of k goes to lane {@code (q / chunk) % L}, with
     * {@code L} 32 and {@code chunk} 8 when B is transposed, 16 and 2 when not; each lane sums its products in order of
     * {@code q} in FP32; then lane {@code t} takes lane {@code t + w} for {@code w = L/2, ..., 1}, and lane 0 holds the
     * sum. @since 1.8.0
     */
    public static void referenceSkinny(short[] a, short[] b, float[] c, Shape s) {
        reference(Algorithm.SKINNY, a, b, c, s, Epilogue.NONE, null, null, null);
    }

    /** As {@link #referenceSkinny}, with an FP16 C (raw bits): the FP32 result rounded once to the nearest half. @since 1.8.0 */
    public static void referenceSkinnyF16(short[] a, short[] b, short[] c, Shape s) {
        referenceF16(Algorithm.SKINNY, a, b, c, s, Epilogue.NONE, null, null, null);
    }

    /**
     * The CPU result of {@link #encodeGrouped} with an FP32 C, bit for bit where the epilogue is
     * {@link Epilogue#exact}: each problem as {@link #reference(Algorithm, short[], short[], float[], Shape, Epilogue,
     * float[], float[], float[])} with TILED's order, at its offsets. @since 1.8.0
     */
    public static void referenceGrouped(short[] a, short[] b, float[] c, List<Problem> problems, Grouped g, Epilogue epilogue,
            float[] bias, float[] gate, float[] residual) {
        referenceGrouped(Elements.fp16(a), Elements.fp16(b), c, problems, g, epilogue, bias, gate, residual);
    }

    /** As {@link #referenceGrouped(short[], short[], float[], List, Grouped, Epilogue, float[], float[], float[])} for any {@link Input}. @since 1.8.0 */
    public static void referenceGrouped(Elements a, Elements b, float[] c, List<Problem> problems, Grouped g, Epilogue epilogue,
            float[] bias, float[] gate, float[] residual) {
        for (var q : problems) {
            for (int i = 0; i < q.m(); i++) {
                for (int j = 0; j < q.n(); j++) {
                    int idx = (int) (q.offsetC() + (long) i * q.ldc() + j);
                    float acc = groupedSum(a, b, q, g, i, j);
                    float r = g.beta() == 0 ? g.alpha() * acc : g.alpha() * acc + g.beta() * c[idx];
                    Operand biasAt = bias == null ? floats(null) : x -> bias[(int) (q.offsetBias() + x)];
                    c[idx] = epilogue(r, i, j, idx, epilogue, biasAt, floats(gate), floats(residual));
                }
            }
        }
    }

    /** As {@link #referenceGrouped} with an FP16 C and FP16 operands (raw bits), rounded once to half. @since 1.8.0 */
    public static void referenceGroupedF16(short[] a, short[] b, short[] c, List<Problem> problems, Grouped g, Epilogue epilogue,
            short[] bias, short[] gate, short[] residual) {
        referenceGroupedF16(Elements.fp16(a), Elements.fp16(b), c, problems, g, epilogue, bias, gate, residual);
    }

    /** As {@link #referenceGroupedF16(short[], short[], short[], List, Grouped, Epilogue, short[], short[], short[])} for any {@link Input}. @since 1.8.0 */
    public static void referenceGroupedF16(Elements a, Elements b, short[] c, List<Problem> problems, Grouped g, Epilogue epilogue,
            short[] bias, short[] gate, short[] residual) {
        for (var q : problems) {
            for (int i = 0; i < q.m(); i++) {
                for (int j = 0; j < q.n(); j++) {
                    int idx = (int) (q.offsetC() + (long) i * q.ldc() + j);
                    float acc = groupedSum(a, b, q, g, i, j);
                    float r = g.beta() == 0 ? g.alpha() * acc : g.alpha() * acc + g.beta() * Float.float16ToFloat(c[idx]);
                    Operand biasAt = bias == null ? halves(null) : x -> Float.float16ToFloat(bias[(int) (q.offsetBias() + x)]);
                    c[idx] = Float.floatToFloat16(epilogue(r, i, j, idx, epilogue, biasAt, halves(gate), halves(residual)));
                }
            }
        }
    }

    private static float groupedSum(Elements a, Elements b, Problem q, Grouped g, int i, int j) {
        float acc = 0;
        for (int t = 0; t < q.k(); t++) {
            long ai = q.offsetA() + (g.transposeA() ? (long) t * q.lda() + i : (long) i * q.lda() + t);
            long bi = q.offsetB() + (g.transposeB() ? (long) j * q.ldb() + t : (long) t * q.ldb() + j);
            acc = Math.fma(a.at(ai), b.at(bi), acc);
        }
        return acc;
    }

    /** The device this multiply runs on. */
    public GpuDevice device() { return device; }

    @Override public void close() {
        var all = new java.util.ArrayList<AutoCloseable>(pipelines.values());
        all.add(skinnyVector);
        all.add(placeholders);
        all.add(grouped);
        PipelineSet.closeAll(all.toArray(AutoCloseable[]::new));
    }
}
