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
 * <b>Algorithm</b> (one group of 1024 threads per row; {@link #reference} computes the same on the CPU):
 * <ul>
 * <li>greedy (temperature 0, or k = 1): the first index of the largest logit (NaN counts as -infinity);</li>
 * <li>weights {@code exp((l - max) / T)} (the reciprocal of T from the host);</li>
 * <li>without top-k and top-p, the weights in index order; with them, the k largest logits (all of a vocabulary of
 * at most 1024; ties at the k-th taken in index order), sorted by logit descending then index, cut at the first whose
 * running sum reaches p times their total;</li>
 * <li>{@code u = rand * total}, rand with 24 bits from a hash of (seed, row, step): the first token whose running sum
 * exceeds u.</li>
 * </ul>
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

    /** {@code rows} rows of {@code vocab} logits, row r at {@code r * ldl}. */
    public record Shape(int rows, int vocab, int ldl) {
        public Shape {
            if (rows <= 0 || vocab <= 0) throw new IllegalArgumentException("dimensions must be positive");
            if (ldl < vocab) throw new IllegalArgumentException("ldl smaller than a row");
        }
        public static Shape of(int rows, int vocab) { return new Shape(rows, vocab, vocab); }
    }

    private final GpuDevice device;
    private final GpuComputePipeline pipeline;
    private final ParameterLayout layout;

    private Sampler(GpuDevice device, GpuComputePipeline pipeline) {
        this.device = device; this.pipeline = pipeline;
        this.layout = ParameterLayout.of(GemmF16.KERNELS, "sample_rows");
    }

    /** Compiles the sampler; {@code UNSUPPORTED} where the device runs fewer than 1024 threads per group. */
    public static Sampler create(GpuDevice device) {
        Checks.notNull(device, "device");
        var pipeline = device.pipeline(GemmF16.KERNELS.pipeline("sample_rows", List.of(Access.READ, Access.READ_WRITE), Set.of(GpuCapabilities.FP32), Set.of()));
        if (pipeline.limits().maxThreads() < THREADS) {
            pipeline.close();
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, "the sampler needs groups of " + THREADS + " threads");
        }
        return new Sampler(device, pipeline);
    }

    /** Records a draw for each row of {@code logits} into {@code tokens} (uint32 per row), for decoding step {@code step}. */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer logits, GpuBuffer tokens, Shape s, Settings settings, int step) {
        Checks.notNull(s, "shape"); Checks.notNull(settings, "settings");
        Checks.range(logits, 0, (long) (s.rows() - 1) * s.ldl() + s.vocab(), 4, "logits");
        Checks.range(tokens, 0, s.rows(), 4, "tokens");
        var grid = Grid.linear(device.capabilities(), s.rows());
        byte[] params = layout.block().u32("rows", s.rows()).u32("vocab", s.vocab()).u32("ldl", s.ldl()).f32("invT", settings.invT())
                .u32("topK", settings.topK()).f32("topP", settings.topP()).u32("seedLo", (int) settings.seed() & 0xFFFFFFFFL)
                .u32("seedHi", (settings.seed() >>> 32)).u32("step", step & 0xFFFFFFFFL).u32("gridX", grid.gridX()).bytes();
        return encoder.dispatch(pipeline, Map.of(0, logits, 1, tokens), params, grid.groups(), "sample");
    }

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

    static int key(float l) {
        int f = Float.floatToRawIntBits(l);
        return Float.isNaN(l) ? 0 : (f & 0x80000000) != 0 ? ~f : f | 0x80000000;
    }

    /**
     * The kernel's token for row {@code r} of {@code logits} on the CPU, with {@code Math.exp} for the device's exp:
     * the same orders of combination and the same draw, so it agrees with a device unless the draw falls within the
     * difference of their exp at a boundary (and greedy always agrees).
     */
    public static int reference(float[] logits, int offset, int vocab, Settings st, int row, int step) {
        int argmax = 0; float mx = Float.NEGATIVE_INFINITY; boolean any = false;
        for (int j = 0; j < vocab; j++) {
            float l = logits[offset + j];
            if (!Float.isNaN(l) && (!any || l > mx)) { mx = l; argmax = j; any = true; }
        }
        if (st.isGreedy() || mx == Float.NEGATIVE_INFINITY) return argmax;
        float invT = st.invT();
        boolean restricted = st.topK() != 0 || st.topP() < 1f;
        if (!restricted) {
            int chunk = (vocab + THREADS - 1) / THREADS;
            float[] part = new float[THREADS];
            for (int t = 0; t < THREADS; t++)
                for (int j = t * chunk; j < Math.min(vocab, (t + 1) * chunk); j++) part[t] = part[t] + (float) Math.exp((logits[offset + j] - mx) * invT);
            float[] excl = part.clone();
            exclusiveScan(excl);
            float total = excl[THREADS - 1] + part[THREADS - 1];
            float u = rand(st.seed(), row, step) * total;
            for (int t = 0; t < THREADS; t++) {
                if (part[t] > 0 && u < excl[t] + part[t]) {
                    float run = excl[t];
                    for (int j = t * chunk; j < Math.min(vocab, (t + 1) * chunk); j++) {
                        run = run + (float) Math.exp((logits[offset + j] - mx) * invT);
                        if (run > u) return j;
                    }
                }
            }
            return argmax;
        }
        int k = st.topK() == 0 ? Math.min(vocab, MAX_K) : Math.min(st.topK(), vocab);
        Integer[] order = new Integer[vocab];
        for (int j = 0; j < vocab; j++) order[j] = j;
        // the k candidates: the k largest keys, ties in index order (the radix select); then the kernel's sort by logit
        // descending (NaN as -infinity, -0 equal to +0), then index
        Arrays.sort(order, (x, y) -> {
            int kx = key(logits[offset + x]), ky = key(logits[offset + y]);
            return kx != ky ? Integer.compareUnsigned(ky, kx) : Integer.compare(x, y);
        });
        Arrays.sort(order, 0, k, (x, y) -> {
            float lx = logits[offset + x], ly = logits[offset + y];
            if (Float.isNaN(lx)) lx = Float.NEGATIVE_INFINITY;
            if (Float.isNaN(ly)) ly = Float.NEGATIVE_INFINITY;
            return lx > ly ? -1 : lx < ly ? 1 : Integer.compare(x, y);
        });
        float[] w = new float[THREADS];
        for (int i = 0; i < k; i++) {
            float l = logits[offset + order[i]];
            w[i] = (float) Math.exp(((Float.isNaN(l) ? Float.NEGATIVE_INFINITY : l) - mx) * invT);
        }
        float[] excl = w.clone();
        exclusiveScan(excl);
        float total = excl[THREADS - 1] + w[THREADS - 1];
        int last = k - 1;
        if (st.topP() < 1f) {
            float cut = st.topP() * total;
            for (int i = 0; i < k; i++) if (excl[i] + w[i] >= cut && (i == 0 || excl[i] < cut)) { last = i; break; }
        }
        float u = rand(st.seed(), row, step) * (excl[last] + w[last]);
        for (int i = 0; i <= last; i++) if (excl[i] + w[i] > u && (i == 0 || excl[i] <= u)) return order[i];
        return order[last];
    }

    /** The contract: greedy exact; sampling repeatable per device, bounded across devices by exp. */
    public static AccuracyContract contract() {
        return new AccuracyContract("Sampler", List.of("FP32 logits"), "FP32 weights in fixed orders (index order, or the sorted candidates)", "uint32 token",
                AccuracyContract.Order.DEVICE, AccuracyContract.Bound.formula("greedy: exactly the first largest logit; sampling: the same token on a "
                        + "device for the same seed and step, and Sampler.reference's unless the draw falls within exp's error of a boundary"), "Sampler.reference");
    }

    @Override public void close() { pipeline.close(); }
}
