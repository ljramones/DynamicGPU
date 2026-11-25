package org.dynamisengine.gpu.primitives;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.dynamisengine.gpu.api.compute.Access;
import org.dynamisengine.gpu.api.compute.GpuBuffer;
import org.dynamisengine.gpu.api.compute.GpuCapabilities;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuComputePipeline;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract.Bound;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract.Order;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.Grid;
import org.dynamisengine.gpu.api.compute.kit.ParameterLayout;
import org.dynamisengine.gpu.api.compute.kit.PipelineSet;
import org.dynamisengine.gpu.api.compute.kit.StridedView;

/**
 * Element type conversions (1.14.0) among FP32, FP16, BF16, INT32 and BOOL, over strided views of rank up to 4 (the
 * input broadcasting to the output's shape). Computed in integer arithmetic on the bits, so every conversion is bit for
 * bit {@link #reference} on every device:
 * <ul>
 * <li>FP32 to FP16 or BF16: round to nearest even ({@link Float#floatToFloat16}, {@link GemmF16#floatToBf16}),
 * overflow to infinity, NaN stays NaN;</li>
 * <li>FP16 and BF16 to FP32: exact;</li>
 * <li>FP32 to INT32: Java's {@code (int)} cast (toward zero, saturating, NaN to 0); INT32 to FP32: round to nearest
 * even;</li>
 * <li>other pairs through FP32: FP16 and BF16 convert into each other with one rounding, and INT32 to FP16 or BF16 rounds
 * to FP32 first;</li>
 * <li>to BOOL: 1 for a nonzero value (NaN included), else 0; from BOOL: a nonzero word is 1 (1.0).</li>
 * </ul>
 * <b>Layout.</b> Views count elements of their own type. FP16 and BF16 elements are 16 bits, two to a 32-bit word
 * (element e in half {@code e & 1} of word {@code e >> 1}, little-endian); the others are 32-bit words, BOOL a uint32 0
 * or 1. A 16-bit output must be contiguous (any element offset); the words it shares with neighbouring data keep their
 * other half.
 *
 * @since 1.14.0
 */
public final class Cast implements AutoCloseable {
    /** Element types. */
    public enum Type {
        FP32(4), FP16(2), BF16(2), INT32(4),
        /** A uint32 word, 0 or 1 (read: nonzero is true). */
        BOOL(4);

        private final int bytes;
        Type(int bytes) { this.bytes = bytes; }
        /** The bytes of an element. */
        public int bytes() { return bytes; }
    }

    private static final int THREADS = 256;
    private final GpuDevice device;
    private final Map<String, PipelineSet> pipelines = new HashMap<>();
    private final ParameterLayout layout = ParameterLayout.of(GemmF16.KERNELS, "cast");
    private final DebugChecks debug;
    private boolean closed;

    private Cast(GpuDevice device) { this.device = device; this.debug = new DebugChecks(device); }

    /** Conversions on {@code device}; pipelines compile on first use. */
    public static Cast create(GpuDevice device) { return new Cast(Checks.notNull(device, "device")); }

    private GpuComputePipeline pipeline(Type from, Type to) {
        if (closed) throw new IllegalStateException("closed");
        return pipelines.computeIfAbsent(from + ">" + to, k -> PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("cast",
                List.of(Access.READ, Access.READ_WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32), Set.of(),
                debug.constants(Map.of("FROM", from.ordinal(), "TO", to.ordinal())))))).get(0);
    }

    /** Records {@code y = (to) x}, x (of type {@code from}) broadcasting to y's shape. */
    public GpuCommandEncoder cast(GpuCommandEncoder encoder, Type from, Type to, GpuBuffer x, StridedView xv, GpuBuffer y, StridedView yv) {
        Checks.notNull(from, "from"); Checks.notNull(to, "to");
        var y4 = yv.withRank(4);
        var x4 = Views.broadcast(xv, y4, "x");
        xv.check(x, from.bytes(), "x");
        yv.checkWritable(y, to.bytes(), "y");
        long elements = Checks.u32(y4.elements(), "element count");
        if (elements == 0) return encoder;
        long threads = elements, firstWord = 0;
        if (to.bytes() == 2) {
            if (!yv.isContiguous()) throw new IllegalArgumentException("a 16-bit output must be contiguous: " + yv);
            firstWord = y4.offset() / 2;
            threads = (y4.offset() + elements - 1) / 2 - firstWord + 1;
        }
        var grid = Grid.forThreads(device.capabilities(), threads, THREADS);
        byte[] params = layout.block().u32("count", threads).u32("gridX", grid.gridX()).u32("xLength", DebugChecks.length(x))
                .u32("yLength", DebugChecks.length(y)).u32("elements", elements).u32("firstWord", firstWord)
                .view("x", x4).view("y", y4).bytes();
        var pipeline = pipeline(from, to);
        debug.before(encoder);
        encoder.dispatch(pipeline, Map.of(0, x, 1, y, 2, debug.buffer()), params, grid.groups(),
                "cast." + from.name().toLowerCase() + "." + to.name().toLowerCase());
        debug.after(encoder);
        return encoder;
    }

    /** An element of {@code type} as a Java float: exact for the float types and BOOL, rounded to nearest for INT32. */
    static float toFloat(Type type, int bits) {
        return switch (type) {
            case FP32 -> Float.intBitsToFloat(bits);
            case FP16 -> Float.float16ToFloat((short) bits);
            case BF16 -> Float.intBitsToFloat(bits << 16);
            case INT32 -> (float) bits;
            case BOOL -> bits != 0 ? 1f : 0f;
        };
    }

    /**
     * The GPU's conversion on the CPU: the output element's bits (a 16-bit result in the low 16 bits) for an input
     * element's bits (a 16-bit input in the low 16 bits). Bit for bit the GPU's, NaN payloads included.
     */
    public static int reference(Type from, Type to, int bits) {
        if (from.bytes() == 2) bits &= 0xFFFF;
        if (to == Type.BOOL) {
            boolean nonzero = switch (from) {
                case FP32 -> (bits & 0x7FFFFFFF) != 0;
                case FP16, BF16 -> (bits & 0x7FFF) != 0;
                default -> bits != 0;
            };
            return nonzero ? 1 : 0;
        }
        if (to == Type.INT32) {
            if (from == Type.INT32) return bits;
            if (from == Type.BOOL) return bits != 0 ? 1 : 0;
            return (int) toFloat(from, bits);
        }
        if (from == to) return bits;
        // FP32 bits of the input, computed on the bits so NaN payloads survive (the GPU's widening)
        int fb = switch (from) {
            case FP32 -> bits;
            case FP16 -> (bits & 0x7C00) == 0x7C00 ? (bits & 0x8000) << 16 | 0x7F800000 | (bits & 0x3FF) << 13
                    : Float.floatToRawIntBits(Float.float16ToFloat((short) bits));
            case BF16 -> bits << 16;
            case INT32 -> Float.floatToRawIntBits((float) bits);
            case BOOL -> bits != 0 ? 0x3F800000 : 0;
        };
        boolean nan = (fb & 0x7FFFFFFF) > 0x7F800000;
        return switch (to) {
            case FP32 -> fb;
            // a NaN keeps its sign and quiets with the top payload bits: sign | 0x7C00 | 0x200 | (payload >> 13)
            case FP16 -> nan ? (fb >>> 16 & 0x8000) | 0x7E00 | (fb & 0x7FFFFF) >>> 13
                    : Float.floatToFloat16(Float.intBitsToFloat(fb)) & 0xFFFF;
            case BF16 -> (nan ? fb >>> 16 | 0x40 : (fb + 0x7FFF + (fb >>> 16 & 1)) >>> 16) & 0xFFFF;   // GemmF16.floatToBf16, on the bits
            default -> throw new AssertionError();
        };
    }

    /** The contract: exact. */
    public static AccuracyContract contract(Type from, Type to) {
        return new AccuracyContract("Cast " + from + " to " + to, List.of(from.name()), "none", to.name(), Order.FIXED,
                new Bound(Bound.Kind.EXACT, 0, "bit for bit Cast.reference: integer arithmetic on the bits"), "Cast.reference");
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        List<AutoCloseable> all = new java.util.ArrayList<>(pipelines.values());
        all.add(debug);
        PipelineSet.closeAll(all.toArray(new AutoCloseable[0]));
    }
}
