package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Half-precision matrix multiply: {@code C = alpha * A * B + beta * C}, with A ({@code m x k}) and B ({@code k x n})
 * in FP16 and C ({@code m x n}) in FP32, all row-major. It is the dense layer of a neural network, with half-precision
 * weights and activations accumulated in single precision.
 * <p>
 * <b>Accuracy.</b> The contract is a bound: before {@code alpha} and {@code beta},
 * {@code |error| <= k * 2^-24 * sum |a b|}. Later kernel versions may sum in another order, for example on matrix
 * hardware, within the same bound.
 * <p>
 * <b>Algorithms.</b> {@link Algorithm#TILED} runs everywhere and is reproducible; {@link Algorithm#SIMDGROUP_MATRIX}
 * uses Apple GPU matrix hardware ({@link GpuCapabilities#SUBGROUP_MATRIX_F16} on native Metal) and is faster, within
 * the same bound but summing in the hardware's order. {@link #create(GpuDevice)} picks the fastest the device has.
 * <p>
 * <b>Reproducibility.</b> The tiled algorithm gives the same bits on every device that rounds FP32 add and multiply
 * correctly (tested on native Metal, MoltenVK and lavapipe):
 * <ul>
 * <li>Products of halves are exact in FP32, so each element of C is {@code acc = fl(acc + a b)} over {@code k} in
 * order, fused multiply-add or not.</li>
 * <li>The result is {@code fl(fl(alpha acc) + fl(beta c))}.</li>
 * <li>{@link #reference} computes the same bits on the CPU.</li>
 * </ul>
 * <p>
 * <b>Use.</b> Create one per device (it compiles a pipeline), then record multiplies into encoders. With
 * {@code beta == 0}, C is not read, as in BLAS. Requires {@link GpuCapabilities#FP16}; {@code create} throws
 * {@link ComputeException.Code#UNSUPPORTED} without it.
 */
public final class GemmF16 implements AutoCloseable {
    static final KernelLibrary KERNELS = KernelLibrary.load(GemmF16.class, "kernels", "kernels.properties", "dynamisgpu.primitives.");
    private static final String KERNEL = "gemm_f16";
    /** The output tile one group computes (both algorithms). */
    static final int TILE = 64;
    private static final String SIMDGROUP_ENTRY = "gemm_f16_simdgroup";

    /** How the multiply is computed. */
    public enum Algorithm {
        /** Shared-memory tiles and FP32 fused multiply-adds: every device with FP16, bit for bit with {@link #reference}. */
        TILED,
        /**
         * SIMD-group matrices (8 x 8 multiply-accumulate in hardware): native Metal on Apple7 and later GPUs, which
         * report {@link GpuCapabilities#SUBGROUP_MATRIX_F16}. Within the bound, not bit for bit with {@link #reference}.
         */
        SIMDGROUP_MATRIX
    }

    /**
     * A multiply's shape and scalars. Leading dimensions are in elements: row {@code r} of A starts at {@code r * lda}.
     *
     * @param m rows of A and C
     * @param n columns of B and C
     * @param k columns of A, rows of B
     */
    public record Shape(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta) {
        public Shape {
            if (m <= 0 || n <= 0 || k <= 0) throw new IllegalArgumentException("dimensions must be positive: " + m + "x" + n + "x" + k);
            if (lda < k || ldb < n || ldc < n) throw new IllegalArgumentException("leading dimension smaller than a row: lda=" + lda + " ldb=" + ldb + " ldc=" + ldc);
        }
        /** Densely packed matrices: {@code lda = k}, {@code ldb = ldc = n}. */
        public static Shape of(int m, int n, int k, float alpha, float beta) { return new Shape(m, n, k, k, n, n, alpha, beta); }
        /** The bytes A, B and C must hold. */
        public long aBytes() { return 2L * ((long) (m - 1) * lda + k); }
        public long bBytes() { return 2L * ((long) (k - 1) * ldb + n); }
        public long cBytes() { return 4L * ((long) (m - 1) * ldc + n); }
        /** Floating-point operations: {@code 2 m n k}. */
        public double flops() { return 2.0 * m * n * k; }
    }

    private final GpuDevice device;
    private final GpuComputePipeline pipeline;
    private final Algorithm algorithm;

    private GemmF16(GpuDevice device, GpuComputePipeline pipeline, Algorithm algorithm) {
        this.device = device; this.pipeline = pipeline; this.algorithm = algorithm;
    }

    /** Compiles the fastest algorithm {@code device} supports. */
    public static GemmF16 create(GpuDevice device) {
        return create(device, supports(device, Algorithm.SIMDGROUP_MATRIX) ? Algorithm.SIMDGROUP_MATRIX : Algorithm.TILED);
    }

    /** Compiles {@code algorithm} for {@code device}; throws {@code UNSUPPORTED} where the device lacks it. */
    public static GemmF16 create(GpuDevice device, Algorithm algorithm) {
        var types = Set.of(GpuCapabilities.FP32, GpuCapabilities.FP16);
        var accesses = List.of(Access.READ, Access.READ, Access.READ_WRITE);
        PipelineDescriptor descriptor = switch (algorithm) {
            case TILED -> KERNELS.pipeline(KERNEL, accesses, types, Set.of());
            case SIMDGROUP_MATRIX -> {
                if (device.adapter().backend() != Backend.METAL)
                    throw new ComputeException(ComputeException.Code.UNSUPPORTED, "SIMD-group matrix GEMM needs native Metal");
                List<BindingLayout.BufferBinding> b = List.of(new BindingLayout.BufferBinding(0, Access.READ, 4),
                        new BindingLayout.BufferBinding(1, Access.READ, 4), new BindingLayout.BufferBinding(2, Access.READ_WRITE, 4));
                yield new PipelineDescriptor(simdgroupBundle(), SIMDGROUP_ENTRY, new BindingLayout(b, KERNELS.parameterBytes(KERNEL)),
                        new Dimensions(128, 1, 1), Map.of(), types, Set.of(GpuCapabilities.SUBGROUP_MATRIX_F16));
            }
        };
        return new GemmF16(device, device.pipeline(descriptor), algorithm);
    }

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
            return new KernelBundle("dynamisgpu.primitives." + SIMDGROUP_ENTRY, "1",
                    Map.of(Backend.METAL, new KernelBundle.Executable(KernelBundle.Format.MSL_SOURCE, source)));
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    /** The algorithm this instance runs. */
    public Algorithm algorithm() { return algorithm; }

    /** Records {@code C = alpha * A * B + beta * C}. */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer a, GpuBuffer b, GpuBuffer c, Shape s) {
        check(a, s.aBytes(), 2, "A"); check(b, s.bBytes(), 2, "B"); check(c, s.cBytes(), 4, "C");
        byte[] params = ByteBuffer.allocate(KERNELS.parameterBytes(KERNEL)).order(ByteOrder.LITTLE_ENDIAN)
                .putInt(s.m()).putInt(s.n()).putInt(s.k()).putInt(s.lda()).putInt(s.ldb()).putInt(s.ldc())
                .putFloat(s.alpha()).putFloat(s.beta()).array();
        var groups = new Dimensions(Math.ceilDiv(s.n(), TILE), Math.ceilDiv(s.m(), TILE), 1);
        return encoder.dispatch(pipeline, Map.of(0, a, 1, b, 2, c), params, groups);
    }

    private static void check(GpuBuffer buffer, long needed, int elementBytes, String name) {
        if (buffer.bytes() < needed) throw new IllegalArgumentException(name + " holds " + buffer.bytes() + " bytes, needs " + needed);
        // the kernel indexes elements with 32-bit unsigned integers
        if (needed / elementBytes > 0xFFFF_FFFFL) throw new IllegalArgumentException(name + " has more than 2^32 elements");
    }

    /**
     * The CPU result of the {@link Algorithm#TILED} kernel, bit for bit: {@code C = fl(fl(alpha acc) + fl(beta c))}, where
     * {@code acc} sums the products in order of {@code k} in FP32, from halves given as raw bits.
     */
    public static void reference(short[] a, short[] b, float[] c, Shape s) {
        for (int i = 0; i < s.m(); i++) {
            for (int j = 0; j < s.n(); j++) {
                float acc = 0;
                for (int q = 0; q < s.k(); q++) acc += Float.float16ToFloat(a[i * s.lda() + q]) * Float.float16ToFloat(b[q * s.ldb() + j]);
                int idx = i * s.ldc() + j;
                c[idx] = s.beta() == 0 ? s.alpha() * acc : s.alpha() * acc + s.beta() * c[idx];
            }
        }
    }

    /** The device this multiply runs on. */
    public GpuDevice device() { return device; }

    @Override public void close() { pipeline.close(); }
}
