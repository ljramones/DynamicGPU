package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.Grid;
import org.dynamisengine.gpu.api.compute.kit.ParameterLayout;
import org.dynamisengine.gpu.api.compute.kit.PipelineSet;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Scaled dot-product attention for every head at once: {@code O = softmax(scale Q K^T + mask) V}. Q is
 * {@code [heads][queries][headDim]}, K and V {@code [kvHeads][cacheKeys][headDim]} (of which the first {@code keys}
 * are attended), all FP16. O is {@code [heads][queries][headDim]} in FP16 or FP32.
 * <p>
 * <b>Fused (1.11.0).</b> {@link #encode(GpuCommandEncoder, GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer, Shape)} runs
 * flash attention: one pass over the keys with an online softmax, the scores never in memory, so memory is linear in
 * the sequence, and with the causal mask key blocks no query sees are skipped. It takes grouped-query attention
 * ({@code kvHeads} dividing {@code heads}: query head {@code h} reads KV head {@code h / (heads / kvHeads)}, and each
 * KV block is read once for all of its query heads) and a KV cache longer than the keys ({@code cacheKeys}). With
 * few rows (decoding), the keys are split into ranges across groups and a second dispatch combines them. Its bound is
 * {@link #fusedContract()}.
 * <p>
 * <b>Composed.</b> The overload with scratch buffers records three dispatches (it needs {@code kvHeads == heads} and
 * {@code cacheKeys == keys}).
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
     * @param kvHeads the heads of K and V, dividing {@code heads} (grouped-query attention; 1.11.0)
     * @param cacheKeys the keys K and V hold per head, at least {@code keys} (a KV cache; 1.11.0)
     */
    public record Shape(int heads, int queries, int keys, int headDim, float scale, boolean causal, int kvHeads, int cacheKeys) {
        public Shape {
            if (heads <= 0 || queries <= 0 || keys <= 0 || headDim <= 0)
                throw new IllegalArgumentException("dimensions must be positive: " + heads + "x" + queries + "x" + keys + "x" + headDim);
            if (causal && keys < queries) throw new IllegalArgumentException("causal attention needs keys >= queries: " + keys + " < " + queries);
            if (!Float.isFinite(scale)) throw new IllegalArgumentException("scale must be finite: " + scale);
            if (kvHeads <= 0 || heads % kvHeads != 0) throw new IllegalArgumentException("kvHeads must divide heads: " + kvHeads + ", " + heads);
            if (cacheKeys < keys) throw new IllegalArgumentException("cacheKeys must be at least keys: " + cacheKeys + " < " + keys);
        }
        /** A shape with {@code kvHeads = heads} and {@code cacheKeys = keys}, as before 1.11.0. */
        public Shape(int heads, int queries, int keys, int headDim, float scale, boolean causal) {
            this(heads, queries, keys, headDim, scale, causal, heads, keys);
        }
        /** No mask, {@code scale = 1 / sqrt(headDim)}. */
        public static Shape of(int heads, int queries, int keys, int headDim) {
            return new Shape(heads, queries, keys, headDim, (float) (1 / Math.sqrt(headDim)), false);
        }
        /** This shape with the causal mask. */
        public Shape asCausal() { return new Shape(heads, queries, keys, headDim, scale, true, kvHeads, cacheKeys); }
        /** This shape with {@code kvHeads} key and value heads (grouped-query attention). @since 1.11.0 */
        public Shape withKvHeads(int kvHeads) { return new Shape(heads, queries, keys, headDim, scale, causal, kvHeads, cacheKeys); }
        /** This shape over K and V buffers of {@code cacheKeys} keys per head, of which {@code keys} are attended. @since 1.11.0 */
        public Shape withCacheKeys(int cacheKeys) { return new Shape(heads, queries, keys, headDim, scale, causal, kvHeads, cacheKeys); }
        public long qBytes() { return 2L * heads * queries * headDim; }
        public long kBytes() { return 2L * kvHeads * cacheKeys * headDim; }
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

    private final GpuDevice device;
    private final GemmF16 scores;
    private final Softmax softmax;
    private final GemmF16 values;
    private final Map<Integer, PipelineSet> fused = new HashMap<>();
    private boolean closed;

    private AttentionF16(GpuDevice device, GemmF16 scores, Softmax softmax, GemmF16 values) {
        this.device = device; this.scores = scores; this.softmax = softmax; this.values = values;
    }

    /** Compiles the steps with the fastest GEMM the device has, writing O in {@code output}. Needs {@code FP16}. */
    public static AttentionF16 create(GpuDevice device, Precision output) {
        GemmF16 s = null; Softmax p = null;
        try {
            s = GemmF16.create(device, Precision.FP32);
            p = Softmax.create(device, Precision.FP16);
            return new AttentionF16(device, s, p, GemmF16.create(device, s.algorithm(), output));
        } catch (RuntimeException e) {
            if (p != null) p.close();
            if (s != null) s.close();
            throw e;
        }
    }

    /**
     * The accuracy contract: bounded, the composition of the steps' contracts (the scores' GemmF16, Softmax, the
     * rounding of P to half, the value GemmF16, then the rounding of O). @since 1.7.0
     */
    public AccuracyContract contract() {
        return new AccuracyContract("AttentionF16 " + algorithm() + ", " + output() + " O", List.of("FP16", "FP16", "FP16"),
                "FP32", output() == Precision.FP32 ? "FP32" : "FP32 rounded once to half", AccuracyContract.Order.DEVICE,
                AccuracyContract.Bound.formula("|O - exact| <= 2 P max|dS| |V| + (Softmax bound + half rounding of P) |V| "
                        + "+ (keys + 2) 2^-24 sum P |V|, plus the rounding of O; dS is the scores' GemmF16 bound"), "");
    }

    /** The GEMM algorithm both products use. */
    public GemmF16.Algorithm algorithm() { return scores.algorithm(); }

    /** The element type of O. */
    public Precision output() { return values.output(); }

    /**
     * Records the composed attention; {@code scores} and {@code probs} are scratch (see {@link Shape#scoresBytes()}).
     * It needs {@code kvHeads == heads} and {@code cacheKeys == keys}.
     */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer o,
            GpuBuffer scoresScratch, GpuBuffer probsScratch, Shape s) {
        if (s.kvHeads() != s.heads() || s.cacheKeys() != s.keys())
            throw new IllegalArgumentException("the composed attention needs kvHeads == heads and cacheKeys == keys; use the fused encode: " + s);
        scores.encode(encoder, q, k, scoresScratch, s.scoresShape());
        softmax.encode(encoder, scoresScratch, probsScratch, s.softmaxShape());
        return values.encode(encoder, probsScratch, v, o, s.outputShape());
    }

    /** The largest head dimension the fused attention takes. @since 1.11.0 */
    public static final int MAX_FUSED_HEAD_DIM = 256;
    static final int BR = 16, MATRIX_ROWS = 32;
    /** Rows of groups a dispatch should have before the keys are split into ranges (a heuristic). */
    static final int TARGET_GROUPS = 256;

    /** The kernel variant for a head dimension: its largest head dimension. */
    static int variant(int headDim) { return headDim <= 64 ? 64 : headDim <= 128 ? 128 : 256; }
    /** Keys per block of a variant. */
    static int keyBlock(int variant) { return variant == 256 ? 16 : 32; }
    /**
     * The group memory a variant's kernels declare, in bytes, read from their SPIR-V: above the 16 KiB every device
     * has (up to 26 KiB), so devices with less refuse the fused attention.
     */
    static long groupBytes(int variant) {
        long most = 0;
        for (String k : List.of("attn_flash_" + variant, "attn_flash_" + variant + "_f32", "attn_decode_" + variant, "attn_decode_" + variant + "_f32"))
            most = Math.max(most, GemmF16.KERNELS.groupSharedBytes(k));
        return most;
    }

    /**
     * Whether {@link #encode(GpuCommandEncoder, GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer, Shape) the fused attention}
     * takes {@code s} on this device: an even head dimension of at most 256, and group memory for its kernel. @since 1.11.0
     */
    public boolean fusedSupports(Shape s) {
        return s.headDim() % 2 == 0 && s.headDim() <= MAX_FUSED_HEAD_DIM
                && groupBytes(variant(s.headDim())) <= device.capabilities().maxGroupSharedBytes();
    }

    /** The most rows per KV head the decode kernel takes, and the scores its group keeps. */
    static final int DECODE_ROWS = 8, DECODE_SCORES = 1024, MAX_SPLITS = 1024;

    /** Whether {@code s} runs the decode kernel: at most 8 rows per KV head. */
    static boolean decodes(Shape s) { return (long) s.heads() / s.kvHeads() * s.queries() <= DECODE_ROWS; }

    /**
     * The key ranges a fused attention of {@code s} uses, {@code {splits, splitKeys}}: more than one when its rows
     * give few groups, and for the decode kernel at most {@code 4096 / rows} keys each.
     */
    static int[] split(Shape s) {
        if (decodes(s)) {
            int rows = s.heads() / s.kvHeads() * s.queries();
            int cap = DECODE_SCORES / rows / 128 * 128;
            int want = Math.ceilDiv(TARGET_GROUPS, s.kvHeads());
            int splitKeys = Math.min(cap, Math.max(128, Math.ceilDiv(Math.ceilDiv(s.keys(), want), 128) * 128));
            // the combine takes at most 1024 ranges
            splitKeys = Math.max(splitKeys, Math.ceilDiv(Math.ceilDiv(s.keys(), MAX_SPLITS), 128) * 128);
            if (splitKeys > cap) throw new IllegalArgumentException("too many keys for the decode kernel: " + s);
            return new int[] {Math.ceilDiv(s.keys(), splitKeys), splitKeys};
        }
        int bc = keyBlock(variant(s.headDim()));
        long groups = (long) Math.ceilDiv(s.heads() / s.kvHeads() * s.queries(), BR) * s.kvHeads();
        int splits = groups >= TARGET_GROUPS ? 1 : (int) Math.min(Math.ceilDiv(TARGET_GROUPS, groups), Math.ceilDiv(s.keys(), 256));
        splits = Math.max(1, splits);
        int splitKeys = Math.ceilDiv(Math.ceilDiv(s.keys(), splits), bc) * bc;
        return new int[] {Math.ceilDiv(s.keys(), splitKeys), splitKeys};   // splits <= 256: within the combine's 1024
    }

    private static final List<Access> FLASH_ACCESSES = List.of(Access.READ, Access.READ, Access.READ, Access.WRITE, Access.READ_WRITE, Access.READ_WRITE, Access.READ);

    /** Whether the SIMD-group matrix flash kernel serves this device and head dimension: native Metal, 64 or 128. */
    boolean matrixPath(int headDim) {
        return (headDim == 64 || headDim == 128) && device.adapter().backend() == Backend.METAL
                && device.capabilities().shaderFeatures().contains(GpuCapabilities.SUBGROUP_MATRIX_F16);
    }

    /** The kernel {@link #encode(GpuCommandEncoder, GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer, Shape)} runs for {@code s}. */
    String fusedKernel(Shape s) { return decodes(s) ? "decode" : matrixPath(s.headDim()) ? "flash.simdgroup" : "flash"; }

    private PipelineSet fusedPipelines(int variant) {
        if (closed) throw new IllegalStateException("closed");
        return fused.computeIfAbsent(variant, d -> {
            String suffix = output() == Precision.FP32 ? "_f32" : "";
            var types = Set.of(GpuCapabilities.FP32);
            var list = new java.util.ArrayList<>(List.of(
                    GemmF16.KERNELS.pipeline("attn_flash_" + d + suffix, FLASH_ACCESSES, types, Set.of()),
                    GemmF16.KERNELS.pipeline("attn_combine" + suffix, List.of(Access.READ, Access.READ, Access.WRITE), types, Set.of()),
                    GemmF16.KERNELS.pipeline("attn_decode_" + d + suffix, FLASH_ACCESSES, types, Set.of())));
            if (matrixPath(d)) {
                var b = new java.util.ArrayList<BindingLayout.BufferBinding>();
                for (int i = 0; i < FLASH_ACCESSES.size(); i++) b.add(new BindingLayout.BufferBinding(i, FLASH_ACCESSES.get(i), 4));
                list.add(new PipelineDescriptor(matrixBundle(), "attn_flash_" + d + suffix + "_simdgroup",
                        new BindingLayout(b, GemmF16.KERNELS.parameterBytes("attn_flash_" + d + suffix)), new Dimensions(128, 1, 1), Map.of(),
                        types, Set.of(GpuCapabilities.SUBGROUP_MATRIX_F16)));
            }
            return PipelineSet.compile(device, list);
        });
    }

    private static KernelBundle matrixBundle() {
        try (var in = AttentionF16.class.getResourceAsStream("metal/attn_flash_simdgroup.metal")) {
            if (in == null) throw new IllegalStateException("missing metal/attn_flash_simdgroup.metal");
            return new KernelBundle("dynamisgpu.primitives.attn_flash_simdgroup", "1",
                    Map.of(Backend.METAL, new KernelBundle.Executable(KernelBundle.Format.MSL_SOURCE, in.readAllBytes())));
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    private GpuBuffer scratch(long bytes, String label) {
        return device.allocate(new BufferDescriptor(Math.max(16, bytes), EnumSet.of(BufferUsage.STORAGE), Placement.PRIVATE).withLabel(label));
    }

    /**
     * Records the fused (flash) attention of {@code s} (1.11.0): O from Q, K and V with no scores in memory. Q
     * ({@link Shape#qBytes()}), K and V ({@link Shape#kBytes()}) are FP16; O ({@link Shape#oBytes}) is this
     * attention's output type. It needs {@link #fusedSupports}.
     */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer o, Shape s) {
        requireFused(s);
        Checks.range(k, 0, s.kBytes() / 2, 2, "k");
        Checks.range(v, 0, s.vBytes() / 2, 2, "v");
        try (var table = scratch(16, "attention-no-pages")) {
            return fused(encoder, q, k, v, table, 0, o, s);
        }
    }

    /**
     * Records the fused attention of {@code s} over layer {@code layer} of a paged KV cache (1.11.0): the keys and
     * values are the first {@code s.keys()} tokens of {@code sequence}, through its page table. {@code s.kvHeads()} and
     * {@code s.headDim()} must be the cache's; {@code s.cacheKeys()} is not used.
     */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer q, PagedKvCache cache, int layer, PagedKvCache.Sequence sequence,
            GpuBuffer o, Shape s) {
        Checks.notNull(cache, "cache"); Checks.notNull(sequence, "sequence");
        requireFused(s);
        var c = cache.config();
        if (s.kvHeads() != c.kvHeads() || s.headDim() != c.headDim())
            throw new IllegalArgumentException("the shape's kvHeads and headDim must be the cache's: " + s + ", " + c);
        if (s.keys() > sequence.tokens()) throw new IllegalArgumentException("the sequence has " + sequence.tokens() + " tokens, not " + s.keys());
        try (var table = cache.table(encoder, sequence)) {
            return fused(encoder, q, cache.keys(layer), cache.values(layer), table, c.pageSize(), o, s);
        }
    }

    private void requireFused(Shape s) {
        if (!fusedSupports(s))
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, "the fused attention needs an even head dimension of at most "
                    + MAX_FUSED_HEAD_DIM + " and " + groupBytes(variant(s.headDim())) + " bytes of group memory: " + s);
    }

    private GpuCommandEncoder fused(GpuCommandEncoder encoder, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer table, int pageSize,
            GpuBuffer o, Shape s) {
        requireFused(s);
        Checks.range(q, 0, s.qBytes() / 2, 2, "q");
        Checks.range(o, 0, s.oBytes(output()) / output().bytes(), output().bytes(), "o");
        int variant = variant(s.headDim());
        var set = fusedPipelines(variant);
        int[] sp = split(s);
        int splits = sp[0], rowBlocks = Math.ceilDiv(s.heads() / s.kvHeads() * s.queries(), BR);
        long rows = (long) s.heads() * s.queries();
        var layout = ParameterLayout.of(GemmF16.KERNELS, "attn_flash_" + variant + (output() == Precision.FP32 ? "_f32" : ""));
        var combineGrid = Grid.linear(device.capabilities(), rows);   // a group per row
        byte[] params = layout.block().u32("heads", s.heads()).u32("kvHeads", s.kvHeads()).u32("queries", s.queries()).u32("keys", s.keys())
                .u32("headDim", s.headDim()).u32("cacheKeys", s.cacheKeys()).f32("scale", s.scale()).u32("causal", s.causal() ? 1 : 0)
                .u32("splits", splits).u32("splitKeys", sp[1]).u32("rowBlocks", splits > 1 ? combineGrid.gridX() : rowBlocks).u32("pageSize", pageSize).bytes();
        // per dispatch: closing the handles keeps them alive for the recorded commands
        try (var ml = scratch(splits > 1 ? 8L * splits * rows : 16, "attention-partial-ml");
             var acc = scratch(splits > 1 ? 4L * splits * rows * s.headDim() : 16, "attention-partial-acc")) {
            if (decodes(s))
                encoder.dispatch(set.get(2), Map.of(0, q, 1, k, 2, v, 3, o, 4, ml, 5, acc, 6, table), params, new Dimensions(1, s.kvHeads(), splits), "attention.decode");
            else if (matrixPath(s.headDim()))
                encoder.dispatch(set.get(3), Map.of(0, q, 1, k, 2, v, 3, o, 4, ml, 5, acc, 6, table), params,
                        new Dimensions(Math.ceilDiv(s.heads() / s.kvHeads() * s.queries(), MATRIX_ROWS), s.kvHeads(), splits), "attention.flash.simdgroup");
            else
                encoder.dispatch(set.get(0), Map.of(0, q, 1, k, 2, v, 3, o, 4, ml, 5, acc, 6, table), params, new Dimensions(rowBlocks, s.kvHeads(), splits), "attention.flash");
            if (splits > 1)
                encoder.dispatch(set.get(1), Map.of(0, ml, 1, acc, 2, o), params, combineGrid.groups(), "attention.combine");
        }
        return encoder;
    }

    /**
     * The contract of the fused attention: bounded. Each weight {@code exp(s - max)} carries the scores' GemmF16
     * error, the rounding of its exponent's argument, one device exp per key block it is rescaled through (and one
     * per combined key range), and a rounding per rescale; the sums of P and P V add {@code (keys + 2 + 2 blocks) 2^-24}
     * of their magnitudes, and {@code acc / l} rounds once, then O. @since 1.11.0
     */
    public AccuracyContract fusedContract() {
        return new AccuracyContract("AttentionF16 fused, " + output() + " O", List.of("FP16", "FP16", "FP16"), "FP32",
                output() == Precision.FP32 ? "FP32" : "FP32 rounded once to half", AccuracyContract.Order.DEVICE,
                AccuracyContract.Bound.formula("|O - exact| <= sum_j (2 eps_max + eps_j) P_j |V_j| + (keys + 2 + 2 B) 2^-24 sum (P + dP) |V| "
                        + "+ rounding of acc / l and of O; eps_j = 2 max|dS| + (max - s_j + 2 max|dS|) 2^-24 + (B + 2) exp error, B the key blocks"), "");
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        try { for (var set : fused.values()) set.close(); } finally {
            try { values.close(); } finally { try { softmax.close(); } finally { scores.close(); } }
        }
    }
}
