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
 * {@link Shape#probsBytes()}). The memory grows with the square of the sequence length. The fused overloads keep scores on chip.
 * <p>
 * <b>Causal.</b> Query {@code i} attends keys {@code 0 .. i + keys - queries}, so new queries at the end of a KV
 * cache see the whole cache and themselves. That needs {@code keys >= queries}.
 * <p>
 * <b>Valid key lengths (1.14.1).</b> The overloads ending in {@code lengths, lengthGroup} read a device buffer
 * of unsigned 32-bit lengths. Query head h in sequence b attends only key columns strictly below
 * {@code lengths[(b * heads + h) / lengthGroup]}. A single sequence uses b = 0. A length above the physical key
 * count admits all available keys. This intersects the causal mask at its original offset and applies to key
 * columns only. A zero length gives zero output, negative-infinity log-sum-exp and zero gradient contributions.
 * Pass null for ordinary attention. The group size must be positive. Training backward uses the same lengths
 * that the forward pass used.
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
    private record DecodeVariant(int headDim, int rowTier) {}
    private final Map<DecodeVariant, GpuComputePipeline> decodePipelines = new HashMap<>();
    private GpuComputePipeline shortDecode;
    private boolean shortDecodeUnsupported;
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
    // Reflection reads immutable packaged SPIR-V. Share its result across devices and output precisions.
    private static final Map<Integer, Long> GROUP_BYTES = new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * The group memory a variant's kernels declare, in bytes, read from their SPIR-V: above the 16 KiB every device
     * has (up to 26 KiB), so devices with less refuse the fused attention.
     */
    static long groupBytes(int variant) {
        return GROUP_BYTES.computeIfAbsent(variant, AttentionF16::readGroupBytes);
    }

    private static long readGroupBytes(int variant) {
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
     * give few groups, and for the decode kernel at most {@code 1024 / rows} keys each.
     */
    static int[] split(Shape s) { return split(s, decodes(s)); }

    /** The key ranges for the decode kernel ({@code decode}) or the flash kernels. */
    static int[] split(Shape s, boolean decode) {
        if (decode) {
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

    private static final List<Access> DECODE_ACCESSES = List.of(Access.READ, Access.READ, Access.READ, Access.WRITE, Access.READ_WRITE, Access.READ, Access.READ);
    /** Page table and LSE precede the optional length buffer. */
    private static final List<Access> FLASH_ACCESSES = List.of(Access.READ, Access.READ, Access.READ, Access.WRITE, Access.READ_WRITE, Access.READ,
            Access.READ_WRITE, Access.READ);
    private static final List<Access> COMBINE_ACCESSES = List.of(Access.READ, Access.WRITE, Access.READ_WRITE);

    /** Whether the SIMD-group matrix flash kernel serves this device and head dimension: native Metal, 64 or 128. */
    boolean matrixPath(int headDim) {
        return (headDim == 64 || headDim == 128) && device.adapter().backend() == Backend.METAL
                && device.capabilities().shaderFeatures().contains(GpuCapabilities.SUBGROUP_MATRIX_F16);
    }

    /** The kernel {@link #encode(GpuCommandEncoder, GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer, Shape)} runs for {@code s}. */
    String fusedKernel(Shape s) { return shortDecode(s) ? "decode.short" : decodes(s) ? "decode" : matrixPath(s.headDim()) ? "flash.simdgroup" : "flash"; }

    private PipelineSet fusedPipelines(int variant) {
        if (closed) throw new IllegalStateException("closed");
        return fused.computeIfAbsent(variant, d -> {
            String suffix = output() == Precision.FP32 ? "_f32" : "";
            var types = Set.of(GpuCapabilities.FP32);
            var caps = device.capabilities();
            boolean wide = d == 128 && device.adapter().backend() == Backend.VULKAN
                    && caps.maxGroupThreads() >= 256 && caps.maxGroupSize().x() >= 256;
            String flashName = "attn_flash_" + d + (wide ? "_wide" : "") + suffix;
            var list = new java.util.ArrayList<>(List.of(
                    GemmF16.KERNELS.pipeline(flashName, FLASH_ACCESSES, types, Set.of()),
                    GemmF16.KERNELS.pipeline("attn_combine" + suffix, COMBINE_ACCESSES, types, Set.of())));
            if (matrixPath(d)) {
                var b = new java.util.ArrayList<BindingLayout.BufferBinding>();
                for (int i = 0; i < FLASH_ACCESSES.size(); i++) b.add(new BindingLayout.BufferBinding(i, FLASH_ACCESSES.get(i), 4));
                list.add(new PipelineDescriptor(metalBundle("attn_flash_simdgroup"), "attn_flash_" + d + suffix + "_simdgroup",
                        new BindingLayout(b, GemmF16.KERNELS.parameterBytes("attn_flash_" + d + suffix)), new Dimensions(128, 1, 1), Map.of(),
                        types, Set.of(GpuCapabilities.SUBGROUP_MATRIX_F16)));
            }
            return PipelineSet.compile(device, list);
        });
    }

    private boolean subgroups = true;

    /**
     * Tests only: compile the decode kernels without subgroup reductions (the fixed-tree path devices without subgroup
     * arithmetic run), before any decode pipeline is compiled.
     */
    void useFixedTreeDecode() {
        if (!decodePipelines.isEmpty()) throw new IllegalStateException("decode pipelines are already compiled");
        subgroups = false;
    }

    private GpuComputePipeline decodePipeline(int variant, int rows) {
        int tier = rows == 1 ? 1 : rows <= 4 ? 4 : DECODE_ROWS;
        return decodePipelines.computeIfAbsent(new DecodeVariant(variant, tier), this::compileDecodePipeline);
    }

    private GpuComputePipeline compileDecodePipeline(DecodeVariant variant) {
        String suffix = variant.headDim() + (output() == Precision.FP32 ? "_f32" : "");
        var constants = Map.<String, Object>of("ROW_TIER", variant.rowTier());
        var types = Set.of(GpuCapabilities.FP32);
        if (subgroups && device.capabilities().shaderFeatures().contains(GpuCapabilities.SUBGROUP_ARITHMETIC)) {
            try {
                return device.pipeline(GemmF16.KERNELS.pipeline("attn_decode_wave_" + suffix, DECODE_ACCESSES, types,
                        Set.of(GpuCapabilities.SUBGROUP_ARITHMETIC), constants));
            } catch (ComputeException e) {
                // Metal checks the 128-thread limit on the compiled pipeline, independently for each row tier.
                if (e.code() != ComputeException.Code.UNSUPPORTED || device.adapter().backend() != Backend.METAL) throw e;
            }
        }
        return device.pipeline(GemmF16.KERNELS.pipeline("attn_decode_" + suffix, DECODE_ACCESSES, types, Set.of(), constants));
    }

    private boolean shortDecode(Shape s) {
        var caps = device.capabilities();
        return !shortDecodeUnsupported && device.adapter().backend() == Backend.METAL && s.headDim() == 128 && decodes(s) && s.keys() <= 512
                && caps.minSubgroupSize() == 32 && caps.maxSubgroupSize() == 32
                && caps.maxGroupThreads() >= 512 && caps.maxGroupSize().x() >= 512
                && caps.shaderFeatures().containsAll(SHORT_FEATURES);
    }

    private static final Set<String> SHORT_FEATURES = Set.of(GpuCapabilities.SUBGROUP_ARITHMETIC, GpuCapabilities.SUBGROUP_SHUFFLE);

    private GpuComputePipeline shortDecodePipeline(int parameterBytes) {
        if (shortDecode == null) {
            var bindings = new java.util.ArrayList<BindingLayout.BufferBinding>();
            for (int i = 0; i < 6; i++) bindings.add(new BindingLayout.BufferBinding(i, i == 3 ? Access.WRITE : Access.READ, 4));
            String suffix = output() == Precision.FP32 ? "_f32" : "";
            try {
                shortDecode = device.pipeline(new PipelineDescriptor(metalBundle("attn_decode_short"), "attn_decode_128" + suffix + "_direct",
                        new BindingLayout(bindings, parameterBytes), new Dimensions(512, 1, 1), Map.of(),
                        Set.of(GpuCapabilities.FP16, GpuCapabilities.FP32), SHORT_FEATURES));
            } catch (ComputeException e) {
                if (e.code() != ComputeException.Code.UNSUPPORTED) throw e;
                // Metal can give this compiled kernel a lower thread limit than the device maximum.
                shortDecodeUnsupported = true;
                return null;
            }
            if (shortDecode.limits().executionWidth() != 32) {
                shortDecode.close();
                shortDecode = null;
                shortDecodeUnsupported = true;
            }
        }
        return shortDecode;
    }

    private static KernelBundle metalBundle(String name) {
        String path = "metal/" + name + ".metal";
        try (var in = AttentionF16.class.getResourceAsStream(path)) {
            if (in == null) throw new IllegalStateException("missing " + path);
            return new KernelBundle("dynamisgpu.primitives." + name, "1",
                    Map.of(Backend.METAL, new KernelBundle.Executable(KernelBundle.Format.MSL_SOURCE, in.readAllBytes())));
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    private GpuBuffer scratch(long bytes, String label) {
        return device.allocate(new BufferDescriptor(Math.max(16, bytes), EnumSet.of(BufferUsage.STORAGE), Placement.PRIVATE).withLabel(label));
    }

    /** The split maxima and sums precede the output accumulators in a single FP32 storage buffer. */
    private GpuBuffer partials(int splits, long rows, int headDim) {
        long words = splits == 1 ? 4 : (long) splits * rows * (headDim + 2);
        Checks.u32(words - 1, "attention partial index");
        return scratch(4 * words, "attention-partials");
    }

    /**
     * Records the fused (flash) attention of {@code s} (1.11.0): O from Q, K and V with no scores in memory. Q
     * ({@link Shape#qBytes()}), K and V ({@link Shape#kBytes()}) are FP16; O ({@link Shape#oBytes}) is this
     * attention's output type. It needs {@link #fusedSupports}.
     */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer o, Shape s) {
        return encode(encoder, q, k, v, o, s, null, 1);
    }

    /**
     * Fused attention with valid key lengths, grouped by query head as described on this class.
     * @since 1.14.1
     */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer o,
            Shape s, GpuBuffer lengths, int lengthGroup) {
        requireFused(s);
        Checks.range(k, 0, s.kBytes() / 2, 2, "k");
        Checks.range(v, 0, s.vBytes() / 2, 2, "v");
        try (var table = scratch(16, "attention-no-pages")) {
            return fused(encoder, q, k, v, table, 0, o, null, s, lengths, lengthGroup);
        }
    }

    /**
     * Records the fused attention of {@code s}, as {@link #encode(GpuCommandEncoder, GpuBuffer, GpuBuffer, GpuBuffer,
     * GpuBuffer, Shape)}, and also each row's log-sum-exp of the scaled scores into {@code lse} (FP32,
     * {@code [heads][queries]}): the forward pass of training, whose L the backward pass ({@link #backward}) takes
     * (1.12.0). It always runs a flash kernel.
     */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer o, GpuBuffer lse, Shape s) {
        return encode(encoder, q, k, v, o, lse, s, null, 1);
    }

    /**
     * Training forward with valid key lengths. An empty row writes zero O and negative-infinity LSE.
     * @since 1.14.1
     */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer o, GpuBuffer lse,
            Shape s, GpuBuffer lengths, int lengthGroup) {
        requireFused(s);
        Checks.notNull(lse, "lse");
        Checks.range(k, 0, s.kBytes() / 2, 2, "k");
        Checks.range(v, 0, s.vBytes() / 2, 2, "v");
        Checks.range(lse, 0, (long) s.heads() * s.queries(), 4, "lse");
        try (var table = scratch(16, "attention-no-pages")) {
            return fused(encoder, q, k, v, table, 0, o, lse, s, lengths, lengthGroup);
        }
    }

    /**
     * Records the fused attention of {@code s} over layer {@code layer} of a paged KV cache (1.11.0): the keys and
     * values are the first {@code s.keys()} tokens of {@code sequence}, through its page table. {@code s.kvHeads()} and
     * {@code s.headDim()} must be the cache's; {@code s.cacheKeys()} is not used.
     */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer q, PagedKvCache cache, int layer, PagedKvCache.Sequence sequence,
            GpuBuffer o, Shape s) {
        return encode(encoder, q, cache, layer, sequence, o, s, null, 1);
    }

    /**
     * Paged attention with the same key-length mask as the contiguous pass.
     * @since 1.14.1
     */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer q, PagedKvCache cache, int layer, PagedKvCache.Sequence sequence,
            GpuBuffer o, Shape s, GpuBuffer lengths, int lengthGroup) {
        Checks.notNull(cache, "cache"); Checks.notNull(sequence, "sequence");
        requireFused(s);
        var c = cache.config();
        if (s.kvHeads() != c.kvHeads() || s.headDim() != c.headDim())
            throw new IllegalArgumentException("the shape's kvHeads and headDim must be the cache's: " + s + ", " + c);
        if (s.keys() > sequence.tokens()) throw new IllegalArgumentException("the sequence has " + sequence.tokens() + " tokens, not " + s.keys());
        try (var table = cache.table(sequence)) {
            return fused(encoder, q, cache.keys(layer), cache.values(layer), table, c.pageSize(), o, null, s, lengths, lengthGroup);
        }
    }

    private static void checkLengths(GpuBuffer lengths, int lengthGroup, long heads) {
        if (lengthGroup < 1) throw new IllegalArgumentException("lengthGroup must be positive: " + lengthGroup);
        if (lengths != null) Checks.range(lengths, 0, Math.ceilDiv(heads, (long) lengthGroup), 4, "lengths");
    }

    private void requireFused(Shape s) {
        if (closed) throw new IllegalStateException("closed");
        if (!fusedSupports(s))
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, "the fused attention needs an even head dimension of at most "
                    + MAX_FUSED_HEAD_DIM + " and " + groupBytes(variant(s.headDim())) + " bytes of group memory: " + s);
    }

    private GpuCommandEncoder fused(GpuCommandEncoder encoder, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer table, int pageSize,
            GpuBuffer o, GpuBuffer lse, Shape s, GpuBuffer lengths, int lengthGroup) {
        requireFused(s);
        checkLengths(lengths, lengthGroup, s.heads());
        var lens = lengths == null ? table : lengths;
        Checks.range(q, 0, s.qBytes() / 2, 2, "q");
        Checks.range(o, 0, s.oBytes(output()) / output().bytes(), output().bytes(), "o");
        int variant = variant(s.headDim());
        var layout = ParameterLayout.of(GemmF16.KERNELS, "attn_flash_" + variant + (output() == Precision.FP32 ? "_f32" : ""));
        boolean decode = lse == null && decodes(s);   // the log-sum-exp comes from the flash kernels
        boolean direct = decode && shortDecode(s) && shortDecodePipeline(layout.bytes()) != null;
        int[] sp = direct ? new int[]{1, s.keys()} : split(s, decode);
        int splits = sp[0], rowBlocks = Math.ceilDiv(s.heads() / s.kvHeads() * s.queries(), BR);
        long rows = (long) s.heads() * s.queries();
        var combineGrid = Grid.linear(device.capabilities(), rows);   // a group per row
        byte[] params = layout.block().u32("heads", s.heads()).u32("kvHeads", s.kvHeads()).u32("queries", s.queries()).u32("keys", s.keys())
                .u32("headDim", s.headDim()).u32("cacheKeys", s.cacheKeys()).f32("scale", s.scale()).u32("causal", s.causal() ? 1 : 0)
                .u32("splits", splits).u32("splitKeys", sp[1]).u32("rowBlocks", direct || splits > 1 ? combineGrid.gridX() : rowBlocks).u32("pageSize", pageSize).u32("batch", 0)
                .bool("writeLse", lse != null).u32("lengthGroup", lengths == null ? 0 : lengthGroup).bytes();
        if (direct)
            return encoder.dispatch(shortDecode, Map.of(0, q, 1, k, 2, v, 3, o, 4, table, 5, lens), params, combineGrid.groups(), "attention.decode.short");
        var set = fusedPipelines(variant);
        // per dispatch: closing the handles keeps them alive for the recorded commands
        try (var partial = partials(splits, rows, s.headDim());
             var noLse = lse == null ? scratch(16, "attention-no-lse") : null) {
            var lseBinding = lse != null ? lse : noLse;
            if (decode)
                encoder.dispatch(decodePipeline(variant, s.heads() / s.kvHeads() * s.queries()), Map.of(0, q, 1, k, 2, v, 3, o, 4, partial, 5, table, 6, lens), params, new Dimensions(1, s.kvHeads(), splits), "attention.decode");
            else if (matrixPath(s.headDim()))
                encoder.dispatch(set.get(2), Map.of(0, q, 1, k, 2, v, 3, o, 4, partial, 5, table, 6, lseBinding, 7, lens), params,
                        new Dimensions(Math.ceilDiv(s.heads() / s.kvHeads() * s.queries(), MATRIX_ROWS), s.kvHeads(), splits), "attention.flash.simdgroup");
            else
                encoder.dispatch(set.get(0), Map.of(0, q, 1, k, 2, v, 3, o, 4, partial, 5, table, 6, lseBinding, 7, lens), params, new Dimensions(rowBlocks, s.kvHeads(), splits), "attention.flash");
            if (splits > 1)
                encoder.dispatch(set.get(1), Map.of(0, partial, 1, o, 2, lseBinding), params, combineGrid.groups(), "attention.combine");
        }
        return encoder;
    }

    /**
     * A batched decode step's dimensions (1.11.1): every sequence of the batch has {@code queries} new queries on each
     * of {@code heads} query heads, and attends all of its tokens in the cache. Q and O are
     * {@code [batch][heads][queries][headDim]}.
     *
     * @param kvHeads the cache's KV heads, dividing {@code heads}; {@code heads / kvHeads * queries} at most 8
     * @param causal whether query {@code i} of a sequence of n tokens sees keys {@code 0 .. i + n - queries}
     */
    public record DecodeBatch(int heads, int kvHeads, int queries, int headDim, float scale, boolean causal) {
        public DecodeBatch {
            if (heads <= 0 || kvHeads <= 0 || queries <= 0 || headDim <= 0)
                throw new IllegalArgumentException("dimensions must be positive: " + heads + ", " + kvHeads + ", " + queries + ", " + headDim);
            if (heads % kvHeads != 0) throw new IllegalArgumentException("kvHeads must divide heads: " + kvHeads + ", " + heads);
            if ((long) heads / kvHeads * queries > DECODE_ROWS)
                throw new IllegalArgumentException("batched decode takes at most " + DECODE_ROWS + " rows per KV head (heads / kvHeads * queries): " + this);
            if (!Float.isFinite(scale)) throw new IllegalArgumentException("scale must be finite: " + scale);
        }
        /** One causal query per sequence, {@code scale = 1 / sqrt(headDim)}. */
        public static DecodeBatch of(int heads, int kvHeads, int headDim) {
            return new DecodeBatch(heads, kvHeads, 1, headDim, (float) (1 / Math.sqrt(headDim)), true);
        }
        /** The bytes Q must hold for {@code batch} sequences. */
        public long qBytes(int batch) { return 2L * batch * heads * queries * headDim; }
        /** The bytes O must hold for {@code batch} sequences in {@code output}. */
        public long oBytes(Precision output, int batch) { return (long) output.bytes() * batch * heads * queries * headDim; }
        /** The shape of one sequence of {@code keys} tokens. */
        Shape shape(int keys) { return new Shape(heads, queries, keys, headDim, scale, causal, kvHeads, keys); }
    }

    /** The key ranges of a batched decode, {@code {splits, splitKeys}}: enough ranges for the GPU, chunked in the kernel. */
    static int[] batchSplit(DecodeBatch b, int batch, int maxKeys) {
        int want = Math.max(1, Math.ceilDiv(TARGET_GROUPS, b.kvHeads() * batch));
        int splitKeys = Math.max(128, Math.ceilDiv(Math.ceilDiv(maxKeys, want), 128) * 128);
        splitKeys = Math.max(splitKeys, Math.ceilDiv(Math.ceilDiv(maxKeys, MAX_SPLITS), 128) * 128);   // the combine's limit
        return new int[] {Math.ceilDiv(maxKeys, splitKeys), splitKeys};
    }

    /**
     * Records one decode step for a batch of sequences of a paged KV cache (1.11.1): for sequence {@code b} of
     * {@code sequences}, its queries (rows {@code b} of Q) attend all of its tokens in layer {@code layer}, through its
     * page table, in one dispatch for the batch (and a combine when its keys are split into ranges). Sequences may have
     * different lengths, and may share pages. The fused bound ({@link #fusedContract}) holds for each sequence.
     */
    public GpuCommandEncoder encodeBatch(GpuCommandEncoder encoder, GpuBuffer q, PagedKvCache cache, int layer,
            List<PagedKvCache.Sequence> sequences, GpuBuffer o, DecodeBatch b) {
        return encodeBatch(encoder, q, cache, layer, sequences, o, b, null, 1);
    }

    /**
     * Batched paged decode with lengths indexed by the flattened sequence and query-head index.
     * @since 1.14.1
     */
    public GpuCommandEncoder encodeBatch(GpuCommandEncoder encoder, GpuBuffer q, PagedKvCache cache, int layer,
            List<PagedKvCache.Sequence> sequences, GpuBuffer o, DecodeBatch b, GpuBuffer lengths, int lengthGroup) {
        Checks.notNull(cache, "cache"); Checks.notNull(sequences, "sequences"); Checks.notNull(b, "batch");
        if (closed) throw new IllegalStateException("closed");
        var c = cache.config();
        if (b.kvHeads() != c.kvHeads() || b.headDim() != c.headDim())
            throw new IllegalArgumentException("the batch's kvHeads and headDim must be the cache's: " + b + ", " + c);
        int batch = sequences.size(), maxKeys = 0;
        checkLengths(lengths, lengthGroup, (long) batch * b.heads());
        if (batch == 0) return encoder;
        for (var seq : sequences) {
            int n = Checks.notNull(seq, "sequence").tokens();
            if (n < (b.causal() ? b.queries() : 1))
                throw new IllegalArgumentException("a sequence has " + n + " tokens, fewer than its " + b.queries() + " queries need");
            maxKeys = Math.max(maxKeys, n);
        }
        var s = b.shape(maxKeys);
        requireFused(s);
        Checks.range(q, 0, b.qBytes(batch) / 2, 2, "q");
        Checks.range(o, 0, b.oBytes(output(), batch) / output().bytes(), output().bytes(), "o");
        int variant = variant(b.headDim());
        var layout = ParameterLayout.of(GemmF16.KERNELS, "attn_flash_" + variant + (output() == Precision.FP32 ? "_f32" : ""));
        int[] sp = batchSplit(b, batch, maxKeys);
        int splits = sp[0];
        long rows = (long) batch * b.heads() * b.queries();
        var combineGrid = Grid.linear(device.capabilities(), rows);
        if (batch > device.capabilities().maxGroupCount().x())
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, batch + " sequences exceed the device's group-count limit");
        byte[] params = layout.block().u32("heads", b.heads()).u32("kvHeads", b.kvHeads()).u32("queries", b.queries()).u32("keys", maxKeys)
                .u32("headDim", b.headDim()).u32("cacheKeys", maxKeys).f32("scale", b.scale()).u32("causal", b.causal() ? 1 : 0)
                .u32("splits", splits).u32("splitKeys", sp[1]).u32("rowBlocks", combineGrid.gridX()).u32("pageSize", c.pageSize())
                .u32("batch", batch).bool("writeLse", false).u32("lengthGroup", lengths == null ? 0 : lengthGroup).bytes();
        var set = fusedPipelines(variant);
        try (var table = cache.batchTable(sequences);
             var partial = partials(splits, rows, b.headDim());
             var noLse = scratch(16, "attention-no-lse")) {
            var lens = lengths == null ? table : lengths;
            encoder.dispatch(decodePipeline(variant, b.heads() / b.kvHeads() * b.queries()),
                    Map.of(0, q, 1, cache.keys(layer), 2, cache.values(layer), 3, o, 4, partial, 5, table, 6, lens), params,
                    new Dimensions(batch, b.kvHeads(), splits), "attention.decode.batch");
            if (splits > 1)
                encoder.dispatch(set.get(1), Map.of(0, partial, 1, o, 2, noLse), params, combineGrid.groups(), "attention.combine");
        }
        return encoder;
    }

    private final Map<Integer, PipelineSet> backwardPipelines = new HashMap<>();

    private static final List<Access> BWD_Q_ACCESSES = List.of(Access.READ, Access.READ, Access.READ, Access.READ,
            Access.READ, Access.READ, Access.READ_WRITE);
    private static final List<Access> BWD_KV_ACCESSES = List.of(Access.READ, Access.READ, Access.READ, Access.READ,
            Access.READ, Access.READ, Access.READ_WRITE, Access.READ_WRITE);

    private PipelineSet backwardPipelines(int variant) {
        if (closed) throw new IllegalStateException("closed");
        return backwardPipelines.computeIfAbsent(variant, d -> {
            var types = Set.of(GpuCapabilities.FP32);
            return PipelineSet.compile(device, List.of(
                    GemmF16.KERNELS.pipeline(output() == Precision.FP32 ? "attn_bwd_dot" : "attn_bwd_dot_h", List.of(Access.READ, Access.READ, Access.READ_WRITE, Access.READ), types, Set.of()),
                    GemmF16.KERNELS.pipeline("attn_bwd_kv_" + d, BWD_KV_ACCESSES, types, Set.of()),
                    GemmF16.KERNELS.pipeline("attn_bwd_q_" + d, BWD_Q_ACCESSES, types, Set.of())));
        });
    }

    private boolean scalarBackward;
    private final Map<Integer, GpuComputePipeline> matrixBackwardQ = new HashMap<>();

    /** Tests only: run the portable backward kernels even where the matrix ones would serve (1.12.1). */
    void useScalarBackward() { scalarBackward = true; }

    /**
     * Whether {@link #backward} runs the Metal matrix kernels for {@code s} (1.12.1): native Metal with
     * SIMD-group matrices, head dimension 64 or 128, and not reproducible mode.
     */
    boolean matrixBackward(Shape s) {
        return !scalarBackward && matrixPath(s.headDim()) && !org.dynamisengine.gpu.api.compute.kit.KitOptions.current().reproducible();
    }

    private final Map<Integer, GpuComputePipeline> matrixBackwardKV = new HashMap<>();

    private GpuComputePipeline matrixBackwardKV(int d) {
        return matrixBackwardKV.computeIfAbsent(d, x -> {
            var acc = BWD_KV_ACCESSES;
            var b = new java.util.ArrayList<BindingLayout.BufferBinding>();
            for (int i = 0; i < acc.size(); i++) b.add(new BindingLayout.BufferBinding(i, acc.get(i), 4));
            return device.pipeline(new PipelineDescriptor(metalBundle("attn_bwd_simdgroup"), "attn_bwd_kv_" + d + "_simdgroup",
                    new BindingLayout(b, GemmF16.KERNELS.parameterBytes("attn_bwd_kv_" + d)), new Dimensions(128, 1, 1), Map.of(),
                    Set.of(GpuCapabilities.FP32), Set.of(GpuCapabilities.SUBGROUP_MATRIX_F16)));
        });
    }

    private GpuComputePipeline matrixBackwardQ(int d) {
        return matrixBackwardQ.computeIfAbsent(d, x -> {
            var acc = BWD_Q_ACCESSES;
            var b = new java.util.ArrayList<BindingLayout.BufferBinding>();
            for (int i = 0; i < acc.size(); i++) b.add(new BindingLayout.BufferBinding(i, acc.get(i), 4));
            return device.pipeline(new PipelineDescriptor(metalBundle("attn_bwd_simdgroup"), "attn_bwd_q_" + d + "_simdgroup",
                    new BindingLayout(b, GemmF16.KERNELS.parameterBytes("attn_bwd_q_" + d)), new Dimensions(128, 1, 1), Map.of(),
                    Set.of(GpuCapabilities.FP32), Set.of(GpuCapabilities.SUBGROUP_MATRIX_F16)));
        });
    }

    /** Query rows per tile of the backward kernels for a variant. */
    static int backwardRows(int variant) { return variant == 256 ? 8 : 16; }

    /**
     * Whether {@link #backward} takes {@code s} on this device: an even head dimension of at most 256, and group memory
     * for its kernels. @since 1.12.0
     */
    public boolean backwardSupports(Shape s) {
        if (s.headDim() % 2 != 0 || s.headDim() > MAX_FUSED_HEAD_DIM) return false;
        int d = variant(s.headDim());
        long bytes = Math.max(GemmF16.KERNELS.groupSharedBytes("attn_bwd_kv_" + d), GemmF16.KERNELS.groupSharedBytes("attn_bwd_q_" + d));
        return bytes <= device.capabilities().maxGroupSharedBytes();
    }

    /**
     * Records the backward pass of the fused attention of {@code s} (1.12.0), from the training forward pass's O (this
     * instance's output type) and log-sum-exp {@code lse} ({@link #encode(GpuCommandEncoder, GpuBuffer, GpuBuffer,
     * GpuBuffer, GpuBuffer, GpuBuffer, Shape)}) and the output gradient {@code dO} (FP32, the layout of O):
     * {@code dQ} (FP32, the layout of Q) and {@code dK}, {@code dV} (FP32, the layout of K and V; keys past
     * {@code s.keys()} are not written) receive the gradients.
     * <p>
     * P is recomputed from Q, K and L, in FP32. Two kernels, one walking each key block's queries for dK and dV and one
     * walking each query block's keys for dQ, keep every sum in a fixed order (no atomics), so a device's result does
     * not depend on scheduling. On native Metal at head dimensions 64 and 128 (not in reproducible mode) they run on the
     * SIMD-group matrix hardware (1.12.1), whose products take FP16 operands: dO, P and dS are rounded to FP16 for them,
     * as the forward matrix path rounds P. Bounded ({@link #backwardContract}).
     */
    public GpuCommandEncoder backward(GpuCommandEncoder encoder, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer o, GpuBuffer lse,
            GpuBuffer dO, GpuBuffer dQ, GpuBuffer dK, GpuBuffer dV, Shape s) {
        return backward(encoder, q, k, v, o, lse, dO, dQ, dK, dV, s, null, 1);
    }

    /**
     * Backward with the same key lengths and grouping as the training forward pass.
     * @since 1.14.1
     */
    public GpuCommandEncoder backward(GpuCommandEncoder encoder, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer o, GpuBuffer lse,
            GpuBuffer dO, GpuBuffer dQ, GpuBuffer dK, GpuBuffer dV, Shape s, GpuBuffer lengths, int lengthGroup) {
        if (closed) throw new IllegalStateException("closed");
        if (!backwardSupports(s))
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, "the attention backward pass needs an even head dimension of at most "
                    + MAX_FUSED_HEAD_DIM + " and group memory for its kernels: " + s);
        long rows = (long) s.heads() * s.queries(), qElems = rows * s.headDim(), kElems = (long) s.kvHeads() * s.cacheKeys() * s.headDim();
        Checks.range(q, 0, qElems, 2, "q"); Checks.range(k, 0, kElems, 2, "k"); Checks.range(v, 0, kElems, 2, "v");
        Checks.range(o, 0, qElems, output().bytes(), "o"); Checks.range(lse, 0, rows, 4, "lse"); Checks.range(dO, 0, qElems, 4, "dO");
        Checks.range(dQ, 0, qElems, 4, "dQ"); Checks.range(dK, 0, kElems, 4, "dK"); Checks.range(dV, 0, kElems, 4, "dV");
        checkLengths(lengths, lengthGroup, s.heads());
        var lens = lengths == null ? q : lengths;
        int variant = variant(s.headDim()), bq = backwardRows(variant);
        var set = backwardPipelines(variant);
        var dotGrid = Grid.forThreads(device.capabilities(), rows, 256);
        byte[] params = ParameterLayout.of(GemmF16.KERNELS, "attn_bwd_kv_" + variant).block().u32("heads", s.heads()).u32("kvHeads", s.kvHeads())
                .u32("queries", s.queries()).u32("keys", s.keys()).u32("headDim", s.headDim()).u32("cacheKeys", s.cacheKeys()).f32("scale", s.scale())
                .bool("causal", s.causal()).u32("gridX", dotGrid.gridX()).u32("lengthGroup", lengths == null ? 0 : lengthGroup).u32("pad1", 0).u32("pad2", 0).bytes();
        int rowsKv = s.heads() / s.kvHeads() * s.queries();
        long lengthWords = lengths == null ? 0 : Math.ceilDiv(s.heads(), lengthGroup);
        Checks.u32(rows + lengthWords - 1, "attention backward scratch index");
        try (var dD = scratch(4 * (rows + lengthWords), "attention-backward-d")) {
            encoder.dispatch(set.get(0), Map.of(0, o, 1, dO, 2, dD, 3, lens), params, dotGrid.groups(), "attention.backward.dot");
            if (matrixBackward(s))
                encoder.dispatch(matrixBackwardKV(variant), Map.of(0, q, 1, k, 2, v, 3, dO, 4, lse, 5, dD, 6, dK, 7, dV), params,
                        new Dimensions(Math.ceilDiv(s.keys(), 32), s.kvHeads(), 1), "attention.backward.kv.simdgroup");
            else encoder.dispatch(set.get(1), Map.of(0, q, 1, k, 2, v, 3, dO, 4, lse, 5, dD, 6, dK, 7, dV), params,
                    new Dimensions(Math.ceilDiv(s.keys(), 16), s.kvHeads(), 1), "attention.backward.kv");
            if (matrixBackward(s))
                encoder.dispatch(matrixBackwardQ(variant), Map.of(0, q, 1, k, 2, v, 3, dO, 4, lse, 5, dD, 6, dQ), params,
                        new Dimensions(Math.ceilDiv(rowsKv, 32), s.kvHeads(), 1), "attention.backward.q.simdgroup");
            else
                encoder.dispatch(set.get(2), Map.of(0, q, 1, k, 2, v, 3, dO, 4, lse, 5, dD, 6, dQ), params,
                        new Dimensions(Math.ceilDiv(rowsKv, bq), s.kvHeads(), 1), "attention.backward.q");
        }
        return encoder;
    }

    /** The backward pass's contract: bounded (the device's exp; sums in fixed orders). @since 1.12.0 */
    public static AccuracyContract backwardContract() {
        return new AccuracyContract("AttentionF16 backward", List.of("FP16 Q, K, V", "O", "FP32 L", "FP32 dO"), "FP32, fixed orders", "FP32",
                AccuracyContract.Order.DEVICE, AccuracyContract.Bound.formula("P carries the scores' error, L's, the rounding of its exponent and "
                        + "exp's; dS = P (dP - D) adds dP's (headDim + 2) 2^-24 sum|dO V| and D's error (O's own error through dO); dV, dK and "
                        + "dQ sum (P or dS) times (dO, Q or K) over their rows or keys with (length + 2) 2^-24 of the magnitudes; the Metal "
                        + "matrix path (1.12.1) adds the FP16 rounding of dO (in dP and dV), of P (in dV) and of dS (in dK and dQ); only visible keys contribute, and an empty row contributes exactly zero"), "");
    }

    /**
     * The contract of the fused attention: bounded. Each weight {@code exp(s - max)} carries the scores' GemmF16
     * error, the rounding of its exponent's argument, one device exp per key block it is rescaled through (and one
     * per combined key range), and a rounding per rescale; the sums of P and P V add {@code (keys + 2 + 2 blocks) 2^-24}
     * of their magnitudes, and {@code acc / l} rounds once, then O. Subgroup decode changes the order of the FP32
     * probability sum. Direct short decode also changes the FP32 dot-product and value summation orders.
     * These orders satisfy the same bound. @since 1.11.0
     */
    public AccuracyContract fusedContract() {
        return new AccuracyContract("AttentionF16 fused, " + output() + " O", List.of("FP16", "FP16", "FP16"), "FP32",
                output() == Precision.FP32 ? "FP32" : "FP32 rounded once to half", AccuracyContract.Order.DEVICE,
                AccuracyContract.Bound.formula("|O - exact| <= sum_j (2 eps_max + eps_j) P_j |V_j| + (keys + 2 + 2 B) 2^-24 sum (P + dP) |V| "
                        + "+ rounding of acc / l and of O; eps_j = 2 max|dS| + (max - s_j + 2 max|dS|) 2^-24 + (B + 2) exp error, B the key blocks. "
                        + "Subgroup decode sums P in subgroup order. Direct short decode reorders the FP32 dot products, sums of P and sums of P V. "
                        + "An empty row writes exactly zero."), "");
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        try {
            var resources = new java.util.ArrayList<AutoCloseable>(decodePipelines.values());
            resources.addAll(fused.values());
            resources.addAll(backwardPipelines.values());
            resources.addAll(matrixBackwardQ.values());
            resources.addAll(matrixBackwardKV.values());
            resources.add(shortDecode);
            PipelineSet.closeAll(resources.toArray(new AutoCloseable[0]));
        } finally {
            try { values.close(); } finally { try { softmax.close(); } finally { scores.close(); } }
        }
    }
}
