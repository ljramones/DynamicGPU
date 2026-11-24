package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.test.compute.MockComputeProvider;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Without a GPU: the kernel's ABI, the shape arithmetic, and refusal on a device without FP16. */
class GemmF16Test {
    @Test void kernelAbiMatchesTheEncoder() {
        for (String k : new String[] {"gemm_f16", "gemm_f16_h", "gemm_skinny", "gemm_skinny_h", "gemm_skinny_v", "gemm_skinny_vh"}) {
            assertEquals(java.util.List.of("a", "b", "c", "epBias", "epGate", "epResidual"), GemmF16.KERNELS.buffers(k));
            assertEquals(68, GemmF16.KERNELS.parameterBytes(k));
            assertEquals(new org.dynamisengine.gpu.api.compute.Dimensions(k.startsWith("gemm_f16") ? 16 : 256, k.startsWith("gemm_f16") ? 16 : 1, 1),
                    GemmF16.KERNELS.localSize(k));
            assertEquals(java.util.List.of("m", "n", "k", "lda", "ldb", "ldc", "alpha", "beta", "transB", "strideA", "strideB", "strideC",
                    "transA", "bias", "act", "gated", "residual"), java.util.List.copyOf(GemmF16.KERNELS.parameterOffsets(k).keySet()));
        }
    }

    @Test void shapeSizes() {
        var s = new GemmF16.Shape(3, 5, 7, 8, 6, 9, 1, 0);
        assertEquals(2L * (2 * 8 + 7), s.aBytes()); assertEquals(2L * (6 * 6 + 5), s.bBytes()); assertEquals(4L * (2 * 9 + 5), s.cBytes());
        assertEquals(2L * (2 * 9 + 5), s.cBytes(Precision.FP16));
        assertEquals(2.0 * 3 * 5 * 7, s.flops());
        var t = GemmF16.Shape.ofTransposedB(3, 5, 7, 1, 0); // B is 5 x 7
        assertEquals(7, t.ldb()); assertEquals(2L * (4 * 7 + 7), t.bBytes()); assertEquals(2 * 7 + 4, t.bIndex(0, 4, 2));
        var batch = GemmF16.Shape.of(3, 5, 7, 1, 0).batched(4);
        assertEquals(3 * 7, batch.strideA()); assertEquals(7 * 5, batch.strideB()); assertEquals(3 * 5, batch.strideC());
        assertEquals(4L * 3 * 7, batch.aElements()); assertEquals(4L * 3 * 5, batch.cElements()); assertEquals(4 * 2.0 * 3 * 5 * 7, batch.flops());
        var shared = GemmF16.Shape.of(3, 5, 7, 1, 0).withBatch(4, 21, 0, 15);
        assertEquals(7L * 5, shared.bElements(), "a stride of 0 shares B");
    }

    @Test void transposedAShapes() {
        var t = GemmF16.Shape.ofTransposedA(3, 5, 7, 1, 0); // A is 7 x 3
        assertEquals(3, t.lda()); assertEquals(2L * (6 * 3 + 3), t.aBytes()); assertEquals(4 * 3 + 2, t.aIndex(0, 2, 4));
        assertEquals(7 * 3, t.batched(2).strideA());
        assertThrows(IllegalArgumentException.class, () -> new GemmF16.Shape(4, 2, 8, 3, 2, 2, 1, 0, false, 1, 0, 0, 0, true)); // lda < m
    }

    /**
     * A column-major product through Shape.columnMajor (C^T = op(B)^T op(A)^T, A and B swapped) equals the product
     * computed directly in column-major order, for every op(A) and op(B).
     */
    @Test void columnMajorProductsMapOntoRowMajorShapes() {
        var r = new java.util.SplittableRandom(5);
        int m = 5, n = 4, k = 6;
        for (boolean ta : new boolean[] {false, true}) {
            for (boolean tb : new boolean[] {false, true}) {
                int lda = (ta ? k : m) + 1, ldb = (tb ? n : k) + 2, ldc = m + 3;   // column-major: ld >= rows stored
                short[] a = new short[lda * (ta ? m : k)], b = new short[ldb * (tb ? k : n)];
                for (int i = 0; i < a.length; i++) a[i] = Float.floatToFloat16((float) r.nextDouble(-1, 1));
                for (int i = 0; i < b.length; i++) b[i] = Float.floatToFloat16((float) r.nextDouble(-1, 1));
                float[] c = new float[ldc * n];
                var s = GemmF16.Shape.columnMajor(m, n, k, lda, ldb, ldc, 1, 0, ta, tb);
                GemmF16.reference(b, a, c, s);   // A and B swapped
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0;
                        for (int q = 0; q < k; q++) {
                            float av = Float.float16ToFloat(ta ? a[i * lda + q] : a[q * lda + i]);   // op(A)[i][q]
                            float bv = Float.float16ToFloat(tb ? b[q * ldb + j] : b[j * ldb + q]);   // op(B)[q][j]
                            sum += (double) av * bv;
                        }
                        assertEquals(sum, c[j * ldc + i], 1e-5, "C[" + i + "][" + j + "] with op(A)=" + (ta ? "A^T" : "A") + " op(B)=" + (tb ? "B^T" : "B"));
                    }
                }
            }
        }
    }

    /** The epilogue's CPU steps, as the kernels apply them. */
    @Test void epilogueSteps() {
        var e = GemmF16.Epilogue.NONE.withBias(GemmF16.Epilogue.Bias.ROW).withActivation(GemmF16.Epilogue.Activation.RELU).withResidual();
        GemmF16.Operand none = i -> { throw new AssertionError(); };
        assertEquals(1.5f, GemmF16.epilogue(-2f, 1, 0, 0, e, i -> i == 1 ? 1f : 99f, none, i -> 1.5f));   // max(-2 + 1, 0) + 1.5
        assertTrue(Float.isNaN(GemmF16.epilogue(Float.NaN, 0, 0, 0, GemmF16.Epilogue.NONE.withActivation(GemmF16.Epilogue.Activation.RELU), none, none, none)));
        assertEquals(0f, GemmF16.epilogue(-0f, 0, 0, 0, GemmF16.Epilogue.NONE.withActivation(GemmF16.Epilogue.Activation.RELU), none, none, none));
        var silu = GemmF16.Epilogue.swiGlu();
        assertEquals(2f / (1f + (float) Math.exp(-2)) * 0.5f, GemmF16.epilogue(2f, 0, 0, 0, silu, none, i -> 0.5f, none));
        var gelu = GemmF16.Epilogue.NONE.withActivation(GemmF16.Epilogue.Activation.GELU);
        assertEquals(1.9545977f, GemmF16.epilogue(2f, 0, 0, 0, gelu, none, none, none), 1e-6f);
        assertEquals(0f, GemmF16.epilogue(-100f, 0, 0, 0, gelu, none, none, none), 0f, "exp overflow gives -1, not NaN");
        assertEquals(100f, GemmF16.epilogue(100f, 0, 0, 0, gelu, none, none, none));
        assertTrue(GemmF16.Epilogue.NONE.withActivation(GemmF16.Epilogue.Activation.RELU).withGate().exact());
        assertFalse(gelu.exact());
    }

    @Test void refusesDevicesWithoutFp16() {
        try (var device = new MockComputeProvider().adapters().getFirst().open()) {
            assertEquals(ComputeException.Code.UNSUPPORTED, assertThrows(ComputeException.class, () -> GemmF16.create(device)).code());
            for (var algorithm : GemmF16.Algorithm.values()) {
                assertFalse(GemmF16.supports(device, algorithm));
                assertEquals(ComputeException.Code.UNSUPPORTED, assertThrows(ComputeException.class, () -> GemmF16.create(device, algorithm)).code());
            }
        }
    }
}
