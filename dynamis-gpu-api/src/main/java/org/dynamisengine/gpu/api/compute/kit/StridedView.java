package org.dynamisengine.gpu.api.compute.kit;

import java.util.Arrays;
import org.dynamisengine.gpu.api.compute.GpuBuffer;

/**
 * An element view of a buffer: a base offset, an extent per dimension and a stride per dimension, all in elements.
 * Element {@code (i0, i1, ...)} is at {@code offset + i0 * stride0 + i1 * stride1 + ...}. Dimension 0 is the
 * outermost.
 * <p>
 * One type covers the layouts kernels meet: dense and sliced matrices (stride = leading dimension), transposes
 * (swapped strides), batches (an outer dimension), broadcasts (stride 0) and interleaved or structure-of-arrays game
 * data (a field's offset, with the record or plane size as its stride). {@link #check} bounds the largest index the
 * view can reach, once, so a kernel reading through it never leaves the buffer.
 * <p>
 * Ranks 1 to 4. Strides are non-negative; a stride of 0 repeats elements, which a kernel may read but must not write
 * ({@link #isWritable}). Views are immutable; {@code equals} compares the layout.
 *
 * @since 1.7.0
 */
public final class StridedView {
    /** The largest rank. */
    public static final int MAX_RANK = 4;

    private final long offset;
    private final long[] extents;
    private final long[] strides;

    private StridedView(long offset, long[] extents, long[] strides) {
        if (extents.length == 0 || extents.length > MAX_RANK || strides.length != extents.length) {
            throw new IllegalArgumentException("a view has 1 to " + MAX_RANK + " dimensions with one stride each");
        }
        if (offset < 0) throw new IllegalArgumentException("negative offset " + offset);
        for (int d = 0; d < extents.length; d++) {
            if (extents[d] < 0 || strides[d] < 0) {
                throw new IllegalArgumentException("negative extent or stride in dimension " + d + ": "
                        + Arrays.toString(extents) + " " + Arrays.toString(strides));
            }
        }
        this.offset = offset;
        this.extents = extents;
        this.strides = strides;
    }

    /** A view with explicit offset, extents and strides (copied). */
    public static StridedView of(long offset, long[] extents, long[] strides) {
        return new StridedView(offset, extents.clone(), strides.clone());
    }

    /** {@code n} consecutive elements from element 0. */
    public static StridedView contiguous(long n) { return new StridedView(0, new long[] {n}, new long[] {1}); }

    /** {@code n} consecutive elements from {@code offset}. */
    public static StridedView contiguous(long offset, long n) {
        return new StridedView(offset, new long[] {n}, new long[] {1});
    }

    /** A dense row-major {@code rows x cols} matrix from element 0. */
    public static StridedView matrix(long rows, long cols) { return matrix(0, rows, cols, cols); }

    /** A row-major {@code rows x cols} matrix from {@code offset} with leading dimension {@code ld} (at least cols). */
    public static StridedView matrix(long offset, long rows, long cols, long ld) {
        if (ld < cols) throw new IllegalArgumentException("leading dimension " + ld + " is less than " + cols + " columns");
        return new StridedView(offset, new long[] {rows, cols}, new long[] {ld, 1});
    }

    /**
     * Field {@code field} of {@code count} records of {@code recordElements} elements each (array of structures,
     * interleaved), from {@code offset}: {@code count} elements, {@code recordElements} apart.
     */
    public static StridedView interleaved(long offset, long count, long recordElements, long field) {
        if (field < 0 || field >= recordElements) throw new IllegalArgumentException("field " + field + " outside a record of " + recordElements);
        return new StridedView(offset + field, new long[] {count}, new long[] {recordElements});
    }

    /**
     * {@code fields} planes of {@code count} elements (structure of arrays), {@code planeStride} elements apart, from
     * {@code offset}: a {@code fields x count} view.
     */
    public static StridedView planes(long offset, long fields, long count, long planeStride) {
        if (fields > 1 && planeStride < count) throw new IllegalArgumentException("plane stride " + planeStride + " is less than " + count);
        return new StridedView(offset, new long[] {fields, count}, new long[] {planeStride, 1});
    }

    public long offset() { return offset; }
    public int rank() { return extents.length; }
    public long extent(int d) { return extents[d]; }
    public long stride(int d) { return strides[d]; }
    public long[] extents() { return extents.clone(); }
    public long[] strides() { return strides.clone(); }

    /** The number of elements the view covers (product of extents). */
    public long elements() {
        long n = 1;
        for (long e : extents) n = Math.multiplyExact(n, e);
        return n;
    }

    /** Whether the view covers no element. */
    public boolean isEmpty() {
        for (long e : extents) if (e == 0) return true;
        return false;
    }

    /** The largest element index the view reaches; -1 if it is empty. */
    public long maxIndex() {
        if (isEmpty()) return -1;
        long m = offset;
        for (int d = 0; d < extents.length; d++) m = Math.addExact(m, Math.multiplyExact(extents[d] - 1, strides[d]));
        return m;
    }

    /** The element index of {@code (i0, i1, ...)}. */
    public long index(long... at) {
        if (at.length != extents.length) throw new IllegalArgumentException("need " + extents.length + " indices");
        long i = offset;
        for (int d = 0; d < at.length; d++) {
            if (at[d] < 0 || at[d] >= extents[d]) throw new IndexOutOfBoundsException("index " + at[d] + " in dimension " + d);
            i += at[d] * strides[d];
        }
        return i;
    }

    /** Whether the elements are consecutive in row-major order from the offset. */
    public boolean isContiguous() {
        long expected = 1;
        for (int d = extents.length - 1; d >= 0; d--) {
            if (extents[d] != 1 && strides[d] != expected) return false;
            expected *= extents[d];
        }
        return true;
    }

    /**
     * Whether no two in-range indices reach the same element, so a kernel may write through the view. Checked
     * conservatively: every stride (of a dimension with more than one element) must exceed the span of the inner
     * dimensions sorted by stride.
     */
    public boolean isWritable() {
        Integer[] order = new Integer[extents.length];
        for (int d = 0; d < order.length; d++) order[d] = d;
        Arrays.sort(order, (a, b) -> Long.compare(strides[a], strides[b]));
        long span = 0; // the largest offset reachable by the dimensions seen so far
        for (int d : order) {
            if (extents[d] <= 1) continue;
            if (strides[d] <= span) return false;
            span += (extents[d] - 1) * strides[d];
        }
        return true;
    }

    /** Swaps dimensions {@code a} and {@code b} (a transpose for a matrix). */
    public StridedView transpose(int a, int b) {
        long[] e = extents.clone(), s = strides.clone();
        long t = e[a]; e[a] = e[b]; e[b] = t;
        t = s[a]; s[a] = s[b]; s[b] = t;
        return new StridedView(offset, e, s);
    }

    /** The {@code count} elements of dimension {@code d} from {@code start}. */
    public StridedView slice(int d, long start, long count) {
        if (start < 0 || count < 0 || start > extents[d] - count) {
            throw new IllegalArgumentException("slice " + start + "+" + count + " outside extent " + extents[d]);
        }
        long[] e = extents.clone(); e[d] = count;
        return new StridedView(offset + start * strides[d], e, strides.clone());
    }

    /** Dimension {@code d} fixed at {@code i}: one rank less (a rank-1 view cannot drop its last dimension). */
    public StridedView select(int d, long i) {
        if (extents.length == 1) throw new IllegalArgumentException("a rank-1 view has no dimension to drop");
        if (i < 0 || i >= extents[d]) throw new IndexOutOfBoundsException("index " + i + " in dimension " + d);
        long[] e = new long[extents.length - 1], s = new long[extents.length - 1];
        for (int k = 0, j = 0; k < extents.length; k++) {
            if (k == d) continue;
            e[j] = extents[k]; s[j++] = strides[k];
        }
        return new StridedView(offset + i * strides[d], e, s);
    }

    /** A new outer dimension: {@code count} copies of this view, {@code stride} elements apart (0 broadcasts). */
    public StridedView batch(long count, long stride) {
        if (extents.length == MAX_RANK) throw new IllegalArgumentException("rank " + MAX_RANK + " views cannot be batched");
        long[] e = new long[extents.length + 1], s = new long[extents.length + 1];
        e[0] = count; s[0] = stride;
        System.arraycopy(extents, 0, e, 1, extents.length);
        System.arraycopy(strides, 0, s, 1, strides.length);
        return new StridedView(offset, e, s);
    }

    /** This view padded with leading dimensions of extent 1 to {@code rank}. */
    public StridedView withRank(int rank) {
        if (rank < extents.length || rank > MAX_RANK) throw new IllegalArgumentException("cannot pad rank " + extents.length + " to " + rank);
        StridedView v = this;
        while (v.rank() < rank) v = v.batch(1, 0);
        return v;
    }

    /**
     * Checks that every element the view reaches lies in {@code buffer} (of {@code elementBytes}-byte elements) and
     * has a 32-bit index. Returns this view.
     */
    public StridedView check(GpuBuffer buffer, int elementBytes, String name) {
        Checks.notNull(buffer, name);
        long max = maxIndex();
        if (max < 0) return this;
        long size = Checks.elements(buffer, elementBytes);
        if (max >= size || max > Checks.MAX_INDEX) {
            throw new IllegalArgumentException(name + ": " + this + " reaches element " + max + ", but the buffer holds "
                    + size + (max > Checks.MAX_INDEX ? " (and kernels index with 32 bits)" : ""));
        }
        return this;
    }

    /** As {@link #check}, and also that a kernel may write through the view. */
    public StridedView checkWritable(GpuBuffer buffer, int elementBytes, String name) {
        if (!isWritable()) throw new IllegalArgumentException(name + ": " + this + " reaches some element twice; it cannot be written");
        return check(buffer, elementBytes, name);
    }

    @Override public boolean equals(Object o) {
        return o instanceof StridedView v && v.offset == offset && Arrays.equals(v.extents, extents) && Arrays.equals(v.strides, strides);
    }

    @Override public int hashCode() { return Long.hashCode(offset) * 31 + Arrays.hashCode(extents) * 17 + Arrays.hashCode(strides); }

    @Override public String toString() {
        return "view(offset " + offset + ", extents " + Arrays.toString(extents) + ", strides " + Arrays.toString(strides) + ")";
    }
}
