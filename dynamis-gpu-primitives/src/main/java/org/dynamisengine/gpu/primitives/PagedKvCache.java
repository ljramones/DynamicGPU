package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.Grid;
import org.dynamisengine.gpu.api.compute.kit.ParameterLayout;
import org.dynamisengine.gpu.api.compute.kit.PipelineSet;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.EnumSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A paged KV cache (1.11.0): keys and values of many sequences in fixed-size pages of a pool, one K and one V pool per
 * layer, attended through each sequence's page table ({@link AttentionF16#encode(GpuCommandEncoder, GpuBuffer,
 * PagedKvCache, int, Sequence, GpuBuffer, AttentionF16.Shape)}).
 * <p>
 * <b>Pages.</b> A page holds {@code pageSize} tokens of every KV head: {@code [kvHeads][pageSize][headDim]} FP16, in
 * every layer's pools at the same index, so a sequence's page list serves all layers. {@code pageSize} is a multiple
 * of 32 (an attention key block never crosses a page).
 * <p>
 * <b>Sharing.</b> {@link #fork} gives a sequence that shares all of another's pages (reference counts), the mechanism
 * prefix caching needs; the prefix index itself (hashing token IDs, a radix tree) belongs to the serving layer. A
 * page is written only while its sequence holds it alone: {@link #reserve} copies a shared, partly filled last page
 * first (copy on write, in every layer). Full shared pages are never written again.
 * <p>
 * <b>Writing a step.</b> {@link #reserve} grows a sequence by n tokens (pages from the pool, copies recorded on the
 * encoder), then {@link #write} stores each layer's new keys and values at their positions.
 * <p>
 * <b>Facts without a sync.</b> {@link #stats()} and a sequence's {@link Sequence#tokens()} and {@link Sequence#pages()}
 * are host-side bookkeeping, readable at any time without waiting for the GPU. Pages referenced by a pinned (active)
 * sequence are pinned; pages referenced only by unpinned ones (a serving layer's cached prefixes) are evictable,
 * reclaimed by {@link Sequence#close()}. What recomputing them would cost depends on the model and the hardware, so
 * the serving layer's cost model derives it from these facts.
 * <p>
 * Methods are thread-safe (synchronized); the GPU work they record must be submitted in the order recorded.
 */
public final class PagedKvCache implements AutoCloseable {
    /**
     * The cache's dimensions.
     *
     * @param layers the layers, each with its own K and V pools
     * @param kvHeads the key and value heads
     * @param headDim the head dimension, even
     * @param pageSize tokens per page, a positive multiple of 32
     * @param pages the pool's pages
     */
    public record Config(int layers, int kvHeads, int headDim, int pageSize, int pages) {
        public Config {
            if (layers <= 0 || kvHeads <= 0 || headDim <= 0 || pages <= 0)
                throw new IllegalArgumentException("dimensions must be positive: " + this);
            if (headDim % 2 != 0) throw new IllegalArgumentException("headDim must be even: " + headDim);
            if (pageSize <= 0 || pageSize % 32 != 0) throw new IllegalArgumentException("pageSize must be a positive multiple of 32: " + pageSize);
        }
        /** Bytes of one page in one layer's K (or V) pool. */
        public long pageBytes() { return 2L * kvHeads * pageSize * headDim; }
        /** Bytes of one layer's K (or V) pool. */
        public long poolBytes() { return pageBytes() * pages; }
    }

    /**
     * The pool's state, host-side and current.
     *
     * @param pages the pool's pages
     * @param free pages no sequence references
     * @param pinned pages a pinned sequence references
     * @param evictable pages referenced only by unpinned sequences
     * @param shared pages more than one sequence references
     * @param sequences open sequences
     */
    public record Stats(int pages, int free, int pinned, int evictable, int shared, int sequences) { }

    /** A sequence: its tokens' pages, in order. Closing it releases its references. */
    public final class Sequence implements AutoCloseable {
        private final List<Integer> pages = new ArrayList<>();
        private int tokens;
        private boolean pinned = true, closed;

        private Sequence() { }

        /** Its tokens. */
        public int tokens() { synchronized (PagedKvCache.this) { return tokens; } }
        /** Its pages, in token order. */
        public int[] pages() { synchronized (PagedKvCache.this) { return pages.stream().mapToInt(Integer::intValue).toArray(); } }
        /** Whether it is pinned (active): new sequences are. */
        public boolean pinned() { synchronized (PagedKvCache.this) { return pinned; } }
        /** Pins or unpins it: an unpinned sequence's pages count as evictable unless a pinned one shares them. */
        public void pin(boolean pin) { synchronized (PagedKvCache.this) { live(this); pinned = pin; } }
        /** The cache it belongs to. */
        public PagedKvCache cache() { return PagedKvCache.this; }

        /** Releases its pages (a page shared with another sequence stays). */
        @Override public void close() {
            synchronized (PagedKvCache.this) {
                if (closed) return;
                closed = true;
                for (int page : pages) unref(page);
                pages.clear();
                sequences.remove(this);
            }
        }
    }

    private final GpuDevice device;
    private final Config config;
    private final GpuBuffer[] keys, values;
    private final int[] refs;
    private final ArrayDeque<Integer> freePages = new ArrayDeque<>();
    private final Map<Sequence, Boolean> sequences = new IdentityHashMap<>();
    private final PipelineSet pipelines;
    private final ParameterLayout layout;
    private boolean closed;

    private PagedKvCache(GpuDevice device, Config config) {
        this.device = device;
        this.config = config;
        this.keys = new GpuBuffer[config.layers()];
        this.values = new GpuBuffer[config.layers()];
        this.refs = new int[config.pages()];
        for (int i = 0; i < config.pages(); i++) freePages.add(i);
        try {
            for (int l = 0; l < config.layers(); l++) {
                keys[l] = pool("kv-keys-" + l);
                values[l] = pool("kv-values-" + l);
            }
            this.pipelines = PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("kv_write",
                    List.of(Access.READ, Access.READ, Access.READ_WRITE, Access.READ_WRITE, Access.READ), Set.of(GpuCapabilities.FP32), Set.of())));
            this.layout = ParameterLayout.of(GemmF16.KERNELS, "kv_write");
        } catch (RuntimeException e) {
            for (var b : keys) if (b != null) b.close();
            for (var b : values) if (b != null) b.close();
            throw e;
        }
    }

    private GpuBuffer pool(String label) {
        return device.allocate(new BufferDescriptor(config.poolBytes(),
                EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE, BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE).withLabel(label));
    }

    /** A cache of {@code config} on {@code device}: its pools are allocated now. */
    public static PagedKvCache create(GpuDevice device, Config config) {
        return new PagedKvCache(Checks.notNull(device, "device"), Checks.notNull(config, "config"));
    }

    /** Its dimensions. */
    public Config config() { return config; }
    /** Layer {@code layer}'s K pool: {@code pages x [kvHeads][pageSize][headDim]} FP16. */
    public GpuBuffer keys(int layer) { return keys[layerIndex(layer)]; }
    /** Layer {@code layer}'s V pool. */
    public GpuBuffer values(int layer) { return values[layerIndex(layer)]; }

    /** A new, empty, pinned sequence. */
    public synchronized Sequence newSequence() {
        open();
        var s = new Sequence();
        sequences.put(s, Boolean.TRUE);
        return s;
    }

    /** A new pinned sequence sharing all of {@code s}'s pages and tokens (a prefix to continue). */
    public synchronized Sequence fork(Sequence s) {
        live(s);
        var f = newSequence();
        for (int page : s.pages) { refs[page]++; f.pages.add(page); }
        f.tokens = s.tokens;
        return f;
    }

    /** Shortens {@code s} to {@code tokens} (a rejected speculation, say), releasing the pages past them. */
    public synchronized void truncate(Sequence s, int tokens) {
        live(s);
        if (tokens < 0 || tokens > s.tokens) throw new IllegalArgumentException("tokens must be in [0, " + s.tokens + "]: " + tokens);
        int keep = Math.ceilDiv(tokens, config.pageSize());
        while (s.pages.size() > keep) unref(s.pages.removeLast());
        s.tokens = tokens;
    }

    /**
     * Grows {@code s} by {@code tokens}, positions {@code s.tokens() ..} of the next {@link #write}s: pages from the
     * pool, and if its last page is shared and partly filled, a private copy of it in every layer, recorded on
     * {@code encoder}. Throws {@link ComputeException.Code#RESOURCE_EXHAUSTED} if the pool has too few free pages (nothing
     * changes then).
     */
    public synchronized GpuCommandEncoder reserve(GpuCommandEncoder encoder, Sequence s, int tokens) {
        live(s);
        if (tokens < 0) throw new IllegalArgumentException("tokens must not be negative: " + tokens);
        int ps = config.pageSize();
        boolean cow = s.tokens % ps != 0 && tokens > 0 && refs[s.pages.getLast()] > 1;
        int need = Math.ceilDiv(s.tokens + tokens, ps) - s.pages.size() + (cow ? 1 : 0);
        if (need > freePages.size())
            throw new ComputeException(ComputeException.Code.RESOURCE_EXHAUSTED, "the KV cache needs " + need + " pages and has " + freePages.size() + " free");
        if (cow) {
            int old = s.pages.removeLast(), fresh = freePages.poll();
            refs[fresh] = 1;
            long pb = config.pageBytes();
            for (int l = 0; l < config.layers(); l++) {
                encoder.copy(keys[l].slice(old * pb, pb), keys[l].slice(fresh * pb, pb));
                encoder.copy(values[l].slice(old * pb, pb), values[l].slice(fresh * pb, pb));
            }
            unref(old);
            s.pages.add(fresh);
        }
        while (s.pages.size() < Math.ceilDiv(s.tokens + tokens, ps)) {
            int page = freePages.poll();
            refs[page] = 1;
            s.pages.add(page);
        }
        s.tokens += tokens;
        return encoder;
    }

    /**
     * Records layer {@code layer}'s keys {@code k} and values {@code v} for {@code tokens} tokens at positions
     * {@code position ..} of {@code s} (already {@link #reserve reserved}), FP16. {@code headStride} and
     * {@code tokenStride} are the sources' strides in halves (even): {@code tokens * headDim} and {@code headDim} for
     * {@code [kvHeads][tokens][headDim]}, {@code headDim} and {@code kvHeads * headDim} for {@code [tokens][kvHeads][headDim]}.
     */
    public synchronized GpuCommandEncoder write(GpuCommandEncoder encoder, int layer, Sequence s, int position, GpuBuffer k, GpuBuffer v,
            int tokens, int headStride, int tokenStride) {
        live(s);
        layerIndex(layer);
        if (position < 0 || tokens < 0 || (long) position + tokens > s.tokens)
            throw new IllegalArgumentException("positions " + position + " + " + tokens + " are not reserved (" + s.tokens + " tokens)");
        if (headStride < 0 || tokenStride < 0 || headStride % 2 != 0 || tokenStride % 2 != 0)
            throw new IllegalArgumentException("strides must be even and non-negative: " + headStride + ", " + tokenStride);
        if (tokens == 0) return encoder;
        long last = (long) (config.kvHeads() - 1) * headStride + (long) (tokens - 1) * tokenStride + config.headDim();
        Checks.range(k, 0, last, 2, "k");
        Checks.range(v, 0, last, 2, "v");
        long words = (long) tokens * config.kvHeads() * (config.headDim() / 2);
        var grid = Grid.forThreads(device.capabilities(), words, 256);
        byte[] params = layout.block().u32("kvHeads", config.kvHeads()).u32("headDim", config.headDim()).u32("tokens", tokens)
                .u32("position", position).u32("pageSize", config.pageSize()).u32("headStride", headStride).u32("tokenStride", tokenStride)
                .u32("gridX", grid.gridX()).bytes();
        try (var table = table(encoder, s)) {
            encoder.dispatch(pipelines.get(0), Map.of(0, k, 1, v, 2, keys[layer], 3, values[layer], 4, table), params, grid.groups(), "kv.write");
        }
        return encoder;
    }

    /** The sequence's page table in a new GPU buffer, uploaded on {@code encoder} (the caller closes the handle). */
    synchronized GpuBuffer table(GpuCommandEncoder encoder, Sequence s) {
        live(s);
        var b = ByteBuffer.allocate(Math.max(16, 4 * s.pages.size())).order(ByteOrder.nativeOrder());
        for (int page : s.pages) b.putInt(page);
        var buffer = device.allocate(new BufferDescriptor(b.capacity(), EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_DESTINATION),
                Placement.PRIVATE).withLabel("kv-page-table"));
        encoder.upload(buffer, b.array());
        return buffer;
    }

    /** The pool's state now (no GPU sync). */
    public synchronized Stats stats() {
        boolean[] pinned = new boolean[config.pages()];
        for (var s : sequences.keySet()) if (s.pinned) for (int page : s.pages) pinned[page] = true;
        int free = 0, pin = 0, evict = 0, shared = 0;
        for (int i = 0; i < config.pages(); i++) {
            if (refs[i] == 0) { free++; continue; }
            if (pinned[i]) pin++; else evict++;
            if (refs[i] > 1) shared++;
        }
        return new Stats(config.pages(), free, pin, evict, shared, sequences.size());
    }

    private int layerIndex(int layer) {
        if (layer < 0 || layer >= config.layers()) throw new IndexOutOfBoundsException("layer " + layer + " of " + config.layers());
        return layer;
    }

    private void unref(int page) {
        if (--refs[page] == 0) freePages.add(page);
    }

    private void open() { if (closed) throw new IllegalStateException("closed"); }

    private void live(Sequence s) {
        open();
        Checks.notNull(s, "sequence");
        if (s.cache() != this) throw new IllegalArgumentException("the sequence belongs to another cache");
        if (s.closed) throw new IllegalStateException("the sequence is closed");
    }

    /** Releases the pools; its sequences become unusable. */
    @Override public synchronized void close() {
        if (closed) return;
        closed = true;
        try { pipelines.close(); } finally {
            for (var b : keys) b.close();
            for (var b : values) b.close();
        }
    }
}
