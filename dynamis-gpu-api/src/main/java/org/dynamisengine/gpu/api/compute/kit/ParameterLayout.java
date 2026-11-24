package org.dynamisengine.gpu.api.compute.kit;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.dynamisengine.gpu.api.compute.KernelLibrary;

/**
 * A kernel's parameter block as its manifest records it: field names, byte offsets and scalar types. Blocks are
 * written by field name ({@link #block()}), so the Java side never restates the field order, and a field written with
 * the wrong type, written twice or left unwritten is an error rather than a silently shifted block.
 * <p>
 * Make the layout once per kernel (it parses the manifest) and a block per dispatch.
 *
 * @since 1.7.0
 */
public final class ParameterLayout {
    private record Field(String name, int index, int offset, String type) { }

    private final String kernel;
    private final Map<String, Field> fields = new LinkedHashMap<>();
    private final int bytes;

    private ParameterLayout(String kernel, Map<String, Integer> offsets, Map<String, String> types) {
        this.kernel = kernel;
        int i = 0, end = 0;
        for (var e : offsets.entrySet()) {
            fields.put(e.getKey(), new Field(e.getKey(), i++, e.getValue(), types.get(e.getKey())));
            end = Math.max(end, e.getValue() + 4);
        }
        this.bytes = end;
    }

    /** The layout of {@code kernel}'s parameter block in {@code library}. */
    public static ParameterLayout of(KernelLibrary library, String kernel) {
        return new ParameterLayout(kernel, library.parameterOffsets(kernel), library.parameterTypes(kernel));
    }

    /** The kernel this layout belongs to. */
    public String kernel() { return kernel; }

    /** The block's size in bytes. */
    public int bytes() { return bytes; }

    /** The field names in declaration order. */
    public List<String> fields() { return List.copyOf(fields.keySet()); }

    /** Whether the block has field {@code name}. */
    public boolean has(String name) { return fields.containsKey(name); }

    /** A new block to fill. */
    public Block block() { return new Block(); }

    /** One parameter block: every field written exactly once, then {@link #bytes()}. */
    public final class Block {
        private final ByteBuffer b = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
        private final boolean[] set = new boolean[fields.size()];

        private Block() { }

        private int slot(String name, String type) {
            Field f = fields.get(name);
            if (f == null) throw new IllegalArgumentException(kernel + " has no parameter " + name + "; it has " + fields.keySet());
            if (!f.type().equals(type)) throw new IllegalArgumentException(kernel + "." + name + " is " + f.type() + ", not " + type);
            if (set[f.index()]) throw new IllegalArgumentException(kernel + "." + name + " written twice");
            set[f.index()] = true;
            return f.offset();
        }

        /** An unsigned 32-bit field; {@code value} must fit. */
        public Block u32(String name, long value) {
            b.putInt(slot(name, "uint32"), Checks.u32(value, kernel + "." + name));
            return this;
        }

        /** A boolean as an unsigned 32-bit field (1 or 0). */
        public Block bool(String name, boolean value) { return u32(name, value ? 1 : 0); }

        /** A signed 32-bit field. */
        public Block i32(String name, int value) {
            b.putInt(slot(name, "int32"), value);
            return this;
        }

        /** A 32-bit float field. */
        public Block f32(String name, float value) {
            b.putFloat(slot(name, "float32"), value);
            return this;
        }

        /**
         * A view's fields: {@code <prefix>Offset}, and per dimension {@code d} that the block has,
         * {@code <prefix>Extent<d>} and {@code <prefix>Stride<d>} (all uint32). A dimension the block lacks must have
         * extent 1; the view is padded with leading unit dimensions to the block's rank.
         */
        public Block view(String prefix, StridedView view) {
            int rank = 0;
            while (rank < StridedView.MAX_RANK && has(prefix + "Extent" + rank)) rank++;
            if (rank == 0) throw new IllegalArgumentException(kernel + " has no view parameters " + prefix + "Extent0...");
            if (view.rank() > rank) {
                for (int d = 0; d < view.rank() - rank; d++) {
                    if (view.extent(d) != 1) throw new IllegalArgumentException(kernel + "." + prefix + " takes rank " + rank + " views, got " + view);
                }
                while (view.rank() > rank) view = view.select(0, 0);   // leading unit dimensions
            }
            StridedView v = view.withRank(rank);
            u32(prefix + "Offset", v.offset());
            for (int d = 0; d < rank; d++) {
                u32(prefix + "Extent" + d, v.extent(d));
                u32(prefix + "Stride" + d, v.stride(d));
            }
            return this;
        }

        /** The block; every field must have been written. */
        public byte[] bytes() {
            for (var f : fields.values()) {
                if (!set[f.index()]) throw new IllegalStateException(kernel + "." + f.name() + " was not written");
            }
            return b.array().clone();
        }
    }
}
