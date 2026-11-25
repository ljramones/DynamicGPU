package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.Grid;
import org.dynamisengine.gpu.api.compute.kit.ParameterLayout;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Token sampling on the GPU (1.10.0): a token from each row of FP32 logits into a uint32 buffer, greedy or with
 * temperature, top-k ({@code k <= 1024}) and top-p, from a seeded counter-based generator. The logits never leave the
 * GPU, and the token buffer feeds the next step's {@link Embedding} directly; reading it back is optional.
 * <p>
 * <b>Algorithm</b> ({@link #reference} computes the same on the CPU). Two dispatches: one group per chunk of 1024
 * logits (its max, first index and weight sum; with top-k or top-p, the chunk sorted), then one group per row
 * combining them. NaN counts as -infinity and -0 as +0.
 * <ul>
 * <li>greedy (temperature 0, or k = 1): the first index of the largest logit;</li>
 * <li>weights {@code exp((l - max) / T)} (the reciprocal of T from the host);</li>
 * <li>without top-k and top-p: chunk g weighs {@code s_g exp((m_g - max) / T)} ({@code s_g} its weights against its
 * own max {@code m_g}, summed in a fixed tree); {@code u = rand * total} picks the chunk, then u's place in it
 * ({@code (u - before) / w_g}) picks the token among the chunk's weights against the row's max;</li>
 * <li>with them: the k largest logits (ties at the k-th taken in index order: merged from the chunks' sorted
 * lists), sorted by logit descending then index, cut at the first whose running sum reaches p times
 * their total, and {@code u = rand * total} drawn among them;</li>
 * <li>rand has 24 bits, from a hash of (seed, row, step).</li>
 * </ul>
 * With top-k or top-p, stage 1 sorts its chunk into runs of 256, and merge rounds between the stages each merge up to
 * 2048 candidates of a power of two of runs in a group (a pairwise merge tree, one binary search per level) and keep
 * the first k, until a row has one sorted list. One row of a 128256-token vocabulary takes stage 1's
 * 126 groups across the GPU, where one group per row took 0.53 ms for top-k with top-p on an M4 Max (1.10.0).
 * The sums run in fixed orders, so a device gives the same token for the same seed and step every time; greedy is
 * the same on every device. exp is the device's, so two devices agree unless u falls within their exp error of a
 * boundary.
 */
public final class Sampler implements AutoCloseable {
    /** The most candidates top-k and top-p consider. */
    public static final int MAX_K = 1024;
    static final int THREADS = 1024;

    /**
     * How to sample.
     *
     * @param temperature 0 for greedy
     * @param topK 0 for none, else 1 .. {@link #MAX_K}
     * @param topP 1 for none, else (0, 1); with top-p but no top-k, the candidates are the 1024 largest
     * @param seed the generator's seed; each row and step draw from it independently
     */
    public record Settings(float temperature, int topK, float topP, long seed) {
        public Settings {
            if (!(temperature >= 0) || Float.isInfinite(temperature)) throw new IllegalArgumentException("temperature must be finite and not negative");
            if (topK < 0 || topK > MAX_K) throw new IllegalArgumentException("topK must be 0 .. " + MAX_K + ": " + topK);
            if (!(topP > 0) || topP > 1) throw new IllegalArgumentException("topP must be in (0, 1]: " + topP);
        }
        /** Greedy decoding. */
        public static Settings greedy() { return new Settings(0f, 0, 1f, 0); }
        boolean isGreedy() { return temperature == 0f || topK == 1; }
        float invT() { return temperature == 0f ? 0f : 1f / temperature; }
    }

    /** The largest vocabulary: 1024 chunks of 1024. */
    public static final int MAX_VOCAB = 1 << 20;

    /** {@code rows} rows of {@code vocab} logits, row r at {@code r * ldl}. */
    public record Shape(int rows, int vocab, int ldl) {
        public Shape {
            if (rows <= 0 || vocab <= 0) throw new IllegalArgumentException("dimensions must be positive");
            if (vocab > MAX_VOCAB || rows > 65535) throw new IllegalArgumentException("at most " + MAX_VOCAB + " logits and 65535 rows");
            if (ldl < vocab) throw new IllegalArgumentException("ldl smaller than a row");
        }
        public static Shape of(int rows, int vocab) { return new Shape(rows, vocab, vocab); }
    }

    private final GpuDevice device;
    private final org.dynamisengine.gpu.api.compute.kit.PipelineSet pipelines;
    private final ParameterLayout layout;

    // Subgroup sorting is optional on devices with narrower per-kernel limits.
    private final boolean shuffle;

    private Sampler(GpuDevice device, org.dynamisengine.gpu.api.compute.kit.PipelineSet pipelines, boolean shuffle) {
        this.shuffle = shuffle;
        this.device = device; this.pipelines = pipelines;
        this.layout = ParameterLayout.of(GemmF16.KERNELS, "sample_final");
    }

    /** Compiles the sampler; {@code UNSUPPORTED} where the device runs fewer than 1024 threads per group. */
    public static Sampler create(GpuDevice device) {
        Checks.notNull(device, "device");
        var types = Set.of(GpuCapabilities.FP32);
        var descriptors = new java.util.ArrayList<>(List.of(
                GemmF16.KERNELS.pipeline("sample_chunks", List.of(Access.READ, Access.READ_WRITE, Access.READ_WRITE, Access.READ_WRITE), types, Set.of()),
                GemmF16.KERNELS.pipeline("sample_final", List.of(Access.READ, Access.READ, Access.READ, Access.READ, Access.READ_WRITE), types, Set.of()),
                GemmF16.KERNELS.pipeline("sample_merge", List.of(Access.READ, Access.READ, Access.READ_WRITE, Access.READ_WRITE), types, Set.of())));
        var features = Set.of(GpuCapabilities.SUBGROUP_SHUFFLE, GpuCapabilities.SUBGROUP_ARITHMETIC);
        boolean shuffle = device.capabilities().shaderFeatures().containsAll(features);
        if (shuffle) descriptors.add(GemmF16.KERNELS.pipeline("sample_shuffle",
                List.of(Access.READ, Access.READ_WRITE, Access.READ_WRITE, Access.READ_WRITE), types, features));
        var set = org.dynamisengine.gpu.api.compute.kit.PipelineSet.compile(device, descriptors);
        if (set.get(0).limits().maxThreads() < THREADS || set.get(1).limits().maxThreads() < THREADS || set.get(2).limits().maxThreads() < THREADS) {
            set.close();
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, "the sampler needs groups of " + THREADS + " threads");
        }
        return new Sampler(device, set, shuffle && set.get(3).limits().maxThreads() >= THREADS);
    }

    private GpuBuffer scratch(long bytes, String label) {
        return device.allocate(new BufferDescriptor(Math.max(16, bytes), java.util.EnumSet.of(BufferUsage.STORAGE), Placement.PRIVATE).withLabel(label));
    }

    /** Records a draw for each row of {@code logits} into {@code tokens} (uint32 per row), for decoding step {@code step}. */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer logits, GpuBuffer tokens, Shape s, Settings settings, int step) {
        Checks.notNull(s, "shape"); Checks.notNull(settings, "settings");
        Checks.range(logits, 0, (long) (s.rows() - 1) * s.ldl() + s.vocab(), 4, "logits");
        Checks.range(tokens, 0, s.rows(), 4, "tokens");
        int chunks = Math.ceilDiv(s.vocab(), CHUNK);
        boolean restricted = !settings.isGreedy() && (settings.topK() != 0 || settings.topP() < 1f);
        int k = settings.topK() == 0 ? Math.min(s.vocab(), MAX_K) : Math.min(settings.topK(), s.vocab());
        long sorted = restricted ? 4L * s.rows() * chunks * CHUNK : 16;
        // per dispatch: closing the handles keeps them alive for the recorded commands
        try (var av = scratch(sorted, "sampler-values"); var ai = scratch(sorted, "sampler-indices");
             var bv = scratch(restricted ? 4L * s.rows() * chunks * CHUNK : 16, "sampler-merged-values");
             var bi = scratch(restricted ? 4L * s.rows() * chunks * CHUNK : 16, "sampler-merged-indices");
             var info = scratch(16L * s.rows() * chunks, "sampler-chunks")) {
            encoder.dispatch(pipelines.get(restricted && shuffle && chunks >= 64 ? 3 : 0), Map.of(0, logits, 1, av, 2, ai, 3, info), params(s, settings, step, chunks, restricted, chunks, CHUNK, k, 1),
                    new Dimensions(chunks, s.rows(), 1), "sample.chunks");
            // merge rounds: each group keeps the first k of up to 2048 candidates, until one list a row remains
            GpuBuffer curV = av, curI = ai, nextV = bv, nextI = bi;
            int lists = chunks * (CHUNK / RUN), stride = RUN;   // stage 1's sorted runs
            // rounds until the final stage can merge the rest in its 1024 slots
            while (restricted && Integer.highestOneBit(lists * 2 - 1) * Math.min(k, stride) > CHUNK) {
                // a power of two of lists whose first min(k, stride) entries fit the group's 2048 slots
                int perGroup = Integer.highestOneBit(Math.max(2, MERGE_SLOTS / Math.min(k, stride))), out = Math.ceilDiv(lists, perGroup);
                encoder.dispatch(pipelines.get(2), Map.of(0, curV, 1, curI, 2, nextV, 3, nextI), params(s, settings, step, chunks, true, lists, stride, k, perGroup),
                        new Dimensions(out, s.rows(), 1), "sample.merge");
                lists = out; stride = k;
                var tv = curV; curV = nextV; nextV = tv;
                var ti = curI; curI = nextI; nextI = ti;
            }
            encoder.dispatch(pipelines.get(1), Map.of(0, logits, 1, curV, 2, curI, 3, info, 4, tokens), params(s, settings, step, chunks, restricted, lists, stride, k, 1),
                    new Dimensions(s.rows(), 1, 1), "sample.final");
        }
        return encoder;
    }

    /** The candidate slots a merge group sorts. */
    static final int MERGE_SLOTS = 2048;

    private byte[] params(Shape s, Settings st, int step, int chunks, boolean restricted, int lists, int stride, int head, int perGroup) {
        return layout.block().u32("rows", s.rows()).u32("vocab", s.vocab()).u32("ldl", s.ldl()).f32("invT", st.invT())
                .u32("topK", st.topK()).f32("topP", st.topP()).u32("seedLo", (int) st.seed() & 0xFFFFFFFFL)
                .u32("seedHi", (st.seed() >>> 32)).u32("step", step & 0xFFFFFFFFL).u32("chunks", chunks).bool("restricted", restricted)
                .u32("lists", lists).u32("stride", stride).u32("head", head).u32("perGroup", perGroup).bytes();
    }

    static final int CHUNK = 1024, RUN = 256;

    // --- the CPU reference -----------------------------------------------------------------------------------------

    static int fmix(int h) { h ^= h >>> 16; h *= 0x85EBCA6B; h ^= h >>> 13; h *= 0xC2B2AE35; h ^= h >>> 16; return h; }

    /** The generator's uniform in [0, 1) for (seed, row, step), as the kernel computes it: 24 bits. */
    public static float rand(long seed, int row, int step) {
        int h = fmix((int) seed ^ fmix((int) (seed >>> 32) ^ fmix(row * 0x9E3779B9 ^ fmix(step + 0x632BE5AB))));
        return (h >>> 8) * 0x1p-24f;
    }

    /** The kit's Blelloch exclusive scan of {@code buf} (a power-of-two length), in its combination order. */
    static void exclusiveScan(float[] buf) {
        int n = buf.length;
        for (int d = 1; d < n; d *= 2) for (int i = 2 * d - 1; i < n; i += 2 * d) buf[i] = buf[i - d] + buf[i];
        buf[n - 1] = 0;
        for (int d = n / 2; d > 0; d /= 2) {
            for (int i = 2 * d - 1; i < n; i += 2 * d) { float sum = buf[i] + buf[i - d]; buf[i - d] = buf[i]; buf[i] = sum; }
        }
    }

    /** NaN as -infinity, -0 as +0, as the kernels sanitize logits. */
    static float sanitize(float l) { return Float.isNaN(l) ? Float.NEGATIVE_INFINITY : l == 0f ? 0f : l; }

    /** The kit's fixed-tree sum of {@code buf} (a power-of-two length), in its combination order. */
    static float treeSum(float[] buf) {
        float[] b = buf.clone();
        for (int w = b.length / 2; w > 0; w /= 2) for (int t = 0; t < w; t++) b[t] = b[t] + b[t + w];
        return b[0];
    }

    /**
     * The kernels' token for row {@code row} of {@code logits} on the CPU, with {@code Math.exp} for the device's exp:
     * the same orders of combination and the same draw, so it agrees with a device unless the draw falls within the
     * difference of their exp (and division) at a boundary; greedy always agrees.
     */
    public static int reference(float[] logits, int offset, int vocab, Settings st, int row, int step) {
        float[] v = new float[vocab];
        for (int j = 0; j < vocab; j++) v[j] = sanitize(logits[offset + j]);
        int argmax = 0; float mx = v[0];
        for (int j = 1; j < vocab; j++) if (v[j] > mx) { mx = v[j]; argmax = j; }
        if (st.isGreedy() || mx == Float.NEGATIVE_INFINITY) return argmax;
        float invT = st.invT(), rnd = rand(st.seed(), row, step);
        int chunks = Math.ceilDiv(vocab, CHUNK);
        if (st.topK() == 0 && st.topP() >= 1f) {
            float[] w = new float[THREADS];
            for (int g = 0; g < chunks; g++) {
                int j0 = g * CHUNK, j1 = Math.min(vocab, j0 + CHUNK);
                float mg = Float.NEGATIVE_INFINITY;
                for (int j = j0; j < j1; j++) mg = Math.max(mg, v[j]);
                float[] e = new float[THREADS];
                if (mg != Float.NEGATIVE_INFINITY) for (int j = j0; j < j1; j++) e[j - j0] = (float) Math.exp((v[j] - mg) * invT);
                float sg = treeSum(e);
                w[g] = sg > 0 ? sg * (float) Math.exp((mg - mx) * invT) : 0f;
            }
            float[] excl = w.clone();
            exclusiveScan(excl);
            float total = excl[THREADS - 1] + w[THREADS - 1];
            float u = rnd * total;
            int g = -1, last = 0;
            for (int t = 0; t < THREADS; t++) {
                if (w[t] > 0) last = t;
                if (g < 0 && w[t] > 0 && u >= excl[t] && u < excl[t] + w[t]) g = t;
            }
            if (g < 0) g = last;
            float frac = (u - excl[g]) / w[g];
            float[] e = new float[THREADS];
            for (int t = 0; t < THREADS; t++) { int j = g * CHUNK + t; e[t] = j < vocab ? (float) Math.exp((v[j] - mx) * invT) : 0f; }
            float[] ex = e.clone();
            exclusiveScan(ex);
            float target = frac * (ex[THREADS - 1] + e[THREADS - 1]);
            int pick = -1; last = 0;
            for (int t = 0; t < THREADS; t++) {
                if (e[t] > 0) last = t;
                if (pick < 0 && e[t] > 0 && ex[t] + e[t] > target && ex[t] <= target) pick = t;
            }
            return g * CHUNK + (pick < 0 ? last : pick);
        }
        int k = st.topK() == 0 ? Math.min(vocab, MAX_K) : Math.min(st.topK(), vocab);
        Integer[] order = new Integer[vocab];
        for (int j = 0; j < vocab; j++) order[j] = j;
        Arrays.sort(order, (x, y) -> v[x] > v[y] ? -1 : v[x] < v[y] ? 1 : Integer.compare(x, y));
        float[] w = new float[THREADS];
        for (int i = 0; i < k; i++) w[i] = (float) Math.exp((v[order[i]] - mx) * invT);
        float[] excl = w.clone();
        exclusiveScan(excl);
        float total = excl[THREADS - 1] + w[THREADS - 1];
        int last = k - 1;
        if (st.topP() < 1f) {
            float cut = st.topP() * total;
            for (int i = 0; i < k; i++) if (excl[i] + w[i] >= cut && (i == 0 || excl[i] < cut)) { last = i; break; }
        }
        float u = rnd * (excl[last] + w[last]);
        for (int i = 0; i <= last; i++) if (excl[i] + w[i] > u && (i == 0 || excl[i] <= u)) return order[i];
        return order[last];
    }

    /** The contract: greedy exact; sampling repeatable per device, bounded across devices by exp. */
    public static AccuracyContract contract() {
        return new AccuracyContract("Sampler", List.of("FP32 logits"), "FP32 weights in fixed orders (index order, or the sorted candidates)", "uint32 token",
                AccuracyContract.Order.DEVICE, AccuracyContract.Bound.formula("greedy: exactly the first largest logit; sampling: the same token on a "
                        + "device for the same seed and step, and Sampler.reference's unless the draw falls within exp's error of a boundary"), "Sampler.reference");
    }

    @Override public void close() { pipelines.close(); }
}
