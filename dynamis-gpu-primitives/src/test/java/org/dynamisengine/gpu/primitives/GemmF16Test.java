package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.test.compute.MockComputeProvider;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Without a GPU: the kernel's ABI, the shape arithmetic, and refusal on a device without FP16. */
class GemmF16Test {
    @Test void kernelAbiMatchesTheEncoder() {
        var kernels = new java.util.ArrayList<String>();
        for (var family : new String[] {"gemm", "gemm_skinny", "gemm_skinny_v", "gemm_small"})
            for (var in : GemmF16.Input.values()) for (var out : Precision.values()) kernels.add(GemmF16.kernel(family, in, out));
        assertTrue(kernels.containsAll(java.util.List.of("gemm_f16", "gemm_f16_h", "gemm_bf16", "gemm_f32_h", "gemm_skinny", "gemm_skinny_bf16_h",
                "gemm_skinny_v_f32", "gemm_small_h")), kernels.toString());
        for (String k : kernels) {
            assertEquals(java.util.List.of("a", "b", "c", "epBias", "epGate", "epResidual"), GemmF16.KERNELS.buffers(k));
            assertEquals(68, GemmF16.KERNELS.parameterBytes(k));
            boolean wide = k.startsWith("gemm_skinny");
            assertEquals(new org.dynamisengine.gpu.api.compute.Dimensions(wide ? 256 : 16, wide ? 1 : 16, 1), GemmF16.KERNELS.localSize(k), k);
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

    /** The grouped kernels add the problem table and the error buffer, and the bounds and lengths to the block. */
    @Test void groupedKernelAbi() {
        for (var in : GemmF16.Input.values()) {
            for (var out : Precision.values()) {
                String k = GemmF16.kernel("gemm_grouped", in, out);
                assertEquals(java.util.List.of("a", "b", "c", "epBias", "epGate", "epResidual", "problems", "errors"), GemmF16.KERNELS.buffers(k));
                assertEquals(68 + 20, GemmF16.KERNELS.parameterBytes(k));
            }
        }
    }

    /** BF16 conversions: exact widening, round to nearest even, NaN kept. */
    @Test void bf16Conversions() {
        assertEquals(1f, GemmF16.bf16ToFloat(GemmF16.floatToBf16(1f)));
        assertEquals((short) 0x3F80, GemmF16.floatToBf16(1f));
        assertEquals((short) 0x3F80, GemmF16.floatToBf16(Float.intBitsToFloat(0x3F808000)), "a tie rounds to even (down)");
        assertEquals((short) 0x3F82, GemmF16.floatToBf16(Float.intBitsToFloat(0x3F818000)), "a tie rounds to even (up)");
        assertEquals((short) 0x3F81, GemmF16.floatToBf16(Float.intBitsToFloat(0x3F808001)), "above the tie rounds up");
        assertEquals((short) 0x7F80, GemmF16.floatToBf16(Float.MAX_VALUE), "the largest float rounds to infinity");
        assertTrue(Float.isNaN(GemmF16.bf16ToFloat(GemmF16.floatToBf16(Float.NaN))));
        assertEquals((short) 0x8000, GemmF16.floatToBf16(-0f));
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

    /**
     * The registry's cost model against the measured shape sweep (Apple M4 Max, native Metal; times relative to TILED,
     * -1 where the algorithm does not apply): with every algorithm (Metal) and without the matrix kernel (Vulkan), the
     * model's choice must be the fastest measured or within 1.2 times of it.
     */
    @Test void costModelPicksTheMeasuredFastest() {
        // m, n, k, batch, B^T, then SIMDGROUP_MATRIX, SKINNY, SMALL relative to TILED
        double[][] measured = {
            {1, 4096, 4096, 1, 0, 1.44, .22, .57}, {1, 4096, 4096, 1, 1, .68, .14, .48}, {16, 4096, 4096, 1, 0, 1.63, 1.63, .59},
            {16, 4096, 4096, 1, 1, .86, .50, .49}, {32, 4096, 4096, 1, 0, 1.88, -1, .99}, {32, 4096, 4096, 1, 1, 1.06, -1, .84},
            {64, 4096, 4096, 1, 0, .54, -1, 1.76}, {64, 4096, 4096, 1, 1, .40, -1, 1.50}, {128, 128, 128, 1, 0, .76, -1, .29},
            {128, 128, 128, 1, 1, .73, -1, .27}, {512, 512, 512, 1, 0, .52, -1, 1.68}, {512, 512, 512, 1, 1, .56, -1, 1.67},
            {2048, 2048, 128, 1, 0, .57, -1, 2.01}, {2048, 2048, 128, 1, 1, .57, -1, 1.99}, {8, 8, 4096, 1, 0, .98, .55, .28},
            {8, 8, 4096, 1, 1, .87, .08, .25}, {16, 16, 16, 4096, 0, 2.25, 17.81, .26}, {16, 16, 16, 4096, 1, 2.11, 11.87, .26},
            {32, 32, 32, 4096, 0, 1.67, -1, .62}, {32, 32, 32, 4096, 1, 1.39, -1, .65}, {16, 16, 256, 512, 0, 1.06, 2.12, .17},
            {16, 16, 256, 512, 1, .76, .79, .17}, {32, 32, 128, 1024, 0, 1.31, -1, .58}, {32, 32, 128, 1024, 1, .96, -1, .59},
            {8, 32, 64, 2048, 0, 1.12, 2.58, .31}, {8, 32, 64, 2048, 1, .80, 2.95, .31}, {64, 64, 64, 512, 0, .75, -1, 1.57},
            {64, 64, 64, 512, 1, .72, -1, 1.57}, {48, 48, 48, 1024, 0, 1.55, -1, 1.09}, {48, 48, 48, 1024, 1, 1.35, -1, 1.14},
            {32, 64, 64, 512, 0, 1.28, -1, .99}, {32, 64, 64, 512, 1, 1.02, -1, 1.01}, {8, 300, 200, 1, 0, 1.15, 1.15, .33},
            {8, 300, 200, 1, 1, .99, .59, .30}, {1, 1024, 1024, 1, 0, 1.08, .18, .28}, {1, 1024, 1024, 1, 1, .84, .04, .26},
            {4, 2048, 512, 1, 0, 1.09, .47, .41}, {4, 2048, 512, 1, 1, .87, .35, .43}, {16, 1024, 1024, 1, 0, 1.18, 1.23, .29},
            {16, 1024, 1024, 1, 1, .95, .34, .29}, {256, 256, 256, 1, 0, .66, -1, .60}, {256, 256, 256, 1, 1, .58, -1, .58},
            {96, 160, 96, 1, 0, 1.50, -1, .28}, {96, 160, 96, 1, 1, 1.27, -1, .28},
            // the 1.14.1 decode grid (docs/evidence/benchmarks-1.14.1/results.csv, Metal): m = 1 to 16 against the projection
            // shapes, both layouts; guards the transposed-B estimate from 32 groups up, which a first 1.15 change broke
            {1, 4096, 4096, 1, 1, .69, .14, .47},
            {1, 4096, 4096, 1, 0, 1.46, .22, .57},
            {2, 4096, 4096, 1, 1, .68, .15, .47},
            {2, 4096, 4096, 1, 0, 1.43, .31, .56},
            {4, 4096, 4096, 1, 1, .68, .20, .47},
            {4, 4096, 4096, 1, 0, 1.43, .46, .57},
            {8, 4096, 4096, 1, 1, .71, .30, .47},
            {8, 4096, 4096, 1, 0, 1.45, .84, .58},
            {16, 4096, 4096, 1, 1, .84, .49, .49},
            {16, 4096, 4096, 1, 0, 1.59, 1.58, .59},
            {1, 11008, 4096, 1, 1, .61, .18, .49},
            {1, 11008, 4096, 1, 0, 1.25, .25, .53},
            {2, 11008, 4096, 1, 1, .62, .18, .49},
            {2, 11008, 4096, 1, 0, 1.23, .32, .53},
            {4, 11008, 4096, 1, 1, .63, .22, .50},
            {4, 11008, 4096, 1, 0, 1.24, .45, .53},
            {8, 11008, 4096, 1, 1, .67, .31, .50},
            {8, 11008, 4096, 1, 0, 1.28, .77, .55},
            {16, 11008, 4096, 1, 1, .80, .49, .52},
            {16, 11008, 4096, 1, 0, 1.38, 1.51, .56},
            {1, 4096, 11008, 1, 1, .66, .13, .49},
            {1, 4096, 11008, 1, 0, 1.49, .24, .55},
            {2, 4096, 11008, 1, 1, .66, .13, .49},
            {2, 4096, 11008, 1, 0, 1.49, .32, .55},
            {4, 4096, 11008, 1, 1, .67, .19, .49},
            {4, 4096, 11008, 1, 0, 1.49, .49, .55},
            {8, 4096, 11008, 1, 1, .75, .30, .49},
            {8, 4096, 11008, 1, 0, 1.52, .88, .56},
            {16, 4096, 11008, 1, 1, .89, .50, .52},
            {16, 4096, 11008, 1, 0, 1.70, 1.62, .58},
            // 1.15 Phase 0 probe (seeded inputs, median of 10 paired rounds, Metal): small n and m = 1 products, where
            // the registry used to withhold SKINNY (docs/evidence/benchmarks-1.15-phase0/probe.csv)
            {4, 8, 2048, 1, 1, .83, .06, .25},
            {4, 8, 2048, 1, 0, .97, .36, .28},
            {4, 8, 4096, 1, 1, .82, .04, .24},
            {4, 8, 4096, 1, 0, .95, .34, .28},
            {4, 8, 8192, 1, 1, .83, .04, .24},
            {4, 8, 8192, 1, 0, .95, .32, .27},
            {4, 16, 2048, 1, 1, .84, .06, .24},
            {4, 16, 2048, 1, 0, 1.07, .35, .27},
            {4, 16, 4096, 1, 1, .86, .05, .25},
            {4, 16, 4096, 1, 0, 1.16, .33, .26},
            {4, 16, 8192, 1, 1, .81, .04, .23},
            {4, 16, 8192, 1, 0, 1.12, .32, .27},
            {4, 32, 2048, 1, 1, .84, .06, .24},
            {4, 32, 2048, 1, 0, 1.07, .37, .27},
            {4, 32, 4096, 1, 1, .84, .05, .24},
            {4, 32, 4096, 1, 0, 1.08, .35, .27},
            {4, 32, 8192, 1, 1, .83, .04, .23},
            {4, 32, 8192, 1, 0, 1.12, .34, .27},
            {8, 8, 2048, 1, 1, .87, .11, .25},
            {8, 8, 2048, 1, 0, 1.00, .64, .28},
            {8, 8, 4096, 1, 1, .89, .08, .25},
            {8, 8, 4096, 1, 0, 1.04, .62, .29},
            {8, 8, 8192, 1, 1, .86, .07, .24},
            {8, 8, 8192, 1, 0, 1.01, .58, .28},
            {8, 16, 2048, 1, 1, .87, .11, .25},
            {8, 16, 2048, 1, 0, 1.10, .62, .27},
            {8, 16, 4096, 1, 1, .87, .08, .24},
            {8, 16, 4096, 1, 0, 1.14, .58, .27},
            {8, 16, 8192, 1, 1, .86, .07, .24},
            {8, 16, 8192, 1, 0, 1.15, .56, .27},
            {8, 32, 2048, 1, 1, .88, .11, .25},
            {8, 32, 2048, 1, 0, 1.12, .63, .28},
            {8, 32, 4096, 1, 1, .87, .08, .24},
            {8, 32, 4096, 1, 0, 1.13, .60, .28},
            {8, 32, 8192, 1, 1, .87, .07, .24},
            {8, 32, 8192, 1, 0, 1.15, .58, .27},
            {16, 8, 2048, 1, 1, .92, .21, .27},
            {16, 8, 2048, 1, 0, 1.01, 1.10, .28},
            {16, 8, 4096, 1, 1, .93, .16, .27},
            {16, 8, 4096, 1, 0, 1.03, 1.05, .28},
            {16, 8, 8192, 1, 1, .92, .13, .26},
            {16, 8, 8192, 1, 0, 1.04, 1.03, .28},
            {16, 16, 2048, 1, 1, .92, .21, .27},
            {16, 16, 2048, 1, 0, 1.17, 1.10, .28},
            {16, 16, 4096, 1, 1, .92, .16, .27},
            {16, 16, 4096, 1, 0, 1.18, 1.04, .27},
            {16, 16, 8192, 1, 1, .92, .13, .26},
            {16, 16, 8192, 1, 0, 1.22, 1.04, .27},
            {16, 32, 2048, 1, 1, .93, .21, .27},
            {16, 32, 2048, 1, 0, 1.17, 1.12, .28},
            {16, 32, 4096, 1, 1, .92, .16, .27},
            {16, 32, 4096, 1, 0, 1.19, 1.07, .28},
            {16, 32, 8192, 1, 1, .92, .13, .26},
            {16, 32, 8192, 1, 0, 1.23, 1.06, .27},
            {1, 512, 512, 1, 0, 1.09, .22, .29},
            {1, 512, 512, 1, 1, .85, .07, .25},
            {1, 512, 1024, 1, 0, 1.09, .18, .28},
            {1, 512, 1024, 1, 1, .84, .04, .24},
            {1, 512, 2048, 1, 0, 1.08, .17, .28},
            {1, 512, 2048, 1, 1, .84, .03, .25},
            {1, 1024, 512, 1, 0, 1.08, .22, .30},
            {1, 1024, 512, 1, 1, .85, .07, .29},
            {1, 1024, 1024, 1, 0, 1.11, .19, .29},
            {1, 1024, 1024, 1, 1, .83, .04, .26},
            {1, 1024, 2048, 1, 0, 1.12, .17, .28},
            {1, 1024, 2048, 1, 1, .83, .03, .25},
            {1, 2048, 512, 1, 0, 1.10, .22, .41},
            {1, 2048, 512, 1, 1, .85, .14, .43},
            {1, 2048, 1024, 1, 0, 1.08, .19, .40},
            {1, 2048, 1024, 1, 1, .84, .09, .42},
            {1, 2048, 2048, 1, 0, 1.11, .17, .40},
            {1, 2048, 2048, 1, 1, .77, .07, .38},
        };
        var algorithms = new GemmF16.Algorithm[] {GemmF16.Algorithm.TILED, GemmF16.Algorithm.SIMDGROUP_MATRIX, GemmF16.Algorithm.SKINNY, GemmF16.Algorithm.SMALL};
        for (boolean matrix : new boolean[] {true, false}) {
            var misses = new java.util.ArrayList<String>();
            for (double[] row : measured) {
                var shape = row[4] == 1 ? GemmF16.Shape.ofTransposedB((int) row[0], (int) row[1], (int) row[2], 1, 0)
                        : GemmF16.Shape.of((int) row[0], (int) row[1], (int) row[2], 1, 0);
                if (row[3] > 1) shape = shape.batched((int) row[3]);
                // the fastest is taken over every kernel that ran, offered or not, so a kernel the registry withholds
                // where it wins counts as a miss
                double[] time = {1, matrix ? row[5] : -1, row[6], row[7]};
                boolean[] offered = {true, matrix, GemmF16.Costs.skinnyApplies(shape), true};
                double[] cost = {GemmF16.Costs.tiled(shape), GemmF16.Costs.simdgroup(shape), GemmF16.Costs.skinny(shape), GemmF16.Costs.small(shape)};
                int pick = -1, best = -1;
                for (int i = 0; i < 4; i++) {
                    if (time[i] < 0) continue;
                    if (offered[i] && (pick < 0 || cost[i] < cost[pick])) pick = i;
                    if (best < 0 || time[i] < time[best]) best = i;
                }
                if (time[pick] > 1.2 * time[best])
                    misses.add(shape.m() + "x" + shape.n() + "x" + shape.k() + "x" + shape.batch() + (shape.transposeB() ? " B^T" : " B")
                            + ": picks " + algorithms[pick] + " (" + time[pick] + "), fastest " + algorithms[best] + " (" + time[best] + ")");
            }
            System.out.println("gemm cost model, " + (matrix ? "with" : "without") + " the matrix kernel: " + misses.size() + " misses " + misses);
            assertEquals(java.util.List.of(), misses);
        }
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
