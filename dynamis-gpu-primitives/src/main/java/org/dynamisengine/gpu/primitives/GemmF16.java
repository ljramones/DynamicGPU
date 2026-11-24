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
 * <b>Reproducibility.</b> This version gives the same bits on every device that rounds FP32 add and multiply
 * correctly (tested on native Metal, MoltenVK and lavapipe):
 * <ul>
 * <li>Products of halves are exact in FP32, so each element of C is {@code acc = fl(acc + a b)} over {@code k} in
 * order, fused multiply-add or not.</li>
 * <li>The result is {@code fl(fl(alpha acc) + fl(beta c))}.</li>
 * <li>{@link #reference} computes the same bits on the CPU.</li>
 * </ul>
 * <p>
 * <b>Use.</b> Create one per device (it compiles a pipeline), then record multiplies into encoders. With
 * {@code beta == 0}, C is not read, as in BLAS. Requires {@link GpuCapabilities#FP16}; {@link #create} throws
 * {@link ComputeException.Code#UNSUPPORTED} without it.
 */
public final class GemmF16 implements AutoCloseable {
    static final KernelLibrary KERNELS = KernelLibrary.load(GemmF16.class, "kernels", "kernels.properties", "dynamisgpu.primitives.");
    private static final String KERNEL = "gemm_f16";
    /** The output tile one group computes. */
    static final int TILE = 64;

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

    private GemmF16(GpuDevice device, GpuComputePipeline pipeline) { this.device = device; this.pipeline = pipeline; }

    /** Compiles the kernel for {@code device}. */
    public static GemmF16 create(GpuDevice device) {
        var descriptor = KERNELS.pipeline(KERNEL, List.of(Access.READ, Access.READ, Access.READ_WRITE),
                Set.of(GpuCapabilities.FP32, GpuCapabilities.FP16), Set.of());
        return new GemmF16(device, device.pipeline(descriptor));
    }

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
     * The CPU result of this kernel version, bit for bit: {@code C = fl(fl(alpha acc) + fl(beta c))}, where
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
