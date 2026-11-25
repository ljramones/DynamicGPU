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
 * Comparisons, masks, select and int32 arithmetic (1.14.0) over strided views of 32-bit words, rank up to 4, inputs
 * broadcasting along their unit dimensions as in {@link Elementwise}.
 * <p>
 * <b>Masks</b> are uint32 words, 1 for true and 0 for false; a mask read as input is true when nonzero. A BOOL tensor
 * on the GPU is such a word per element ({@link Cast.Type#BOOL}).
 * <p>
 * <b>Exact everywhere.</b> Every operation works on the bits in integer arithmetic, so it is bit for bit
 * {@link #reference(Compare, Operand, int, int) its reference} on every device. FP32 comparisons follow Java's float
 * comparisons: NaN compares unequal to everything (only NOT_EQUAL holds), -0 equals +0, and subnormals compare by
 * value even on devices that flush them in arithmetic. Int32 arithmetic wraps as Java's int.
 *
 * @since 1.14.0
 */
public final class Logic implements AutoCloseable {
    /** Comparisons, giving a mask. */
    public enum Compare { EQUAL, NOT_EQUAL, LESS, LESS_EQUAL, GREATER, GREATER_EQUAL }

    /** The element type compared. */
    public enum Operand { FP32, INT32 }

    /** Logical operations on masks. */
    public enum Logical { AND, OR, XOR }

    /** FP32 classification, giving a mask. */
    public enum Classify {
        /** Neither NaN nor infinite. */
        IS_FINITE,
        IS_NAN,
        IS_INF
    }

    /** Int32 arithmetic, wrapping as Java's int. */
    public enum IntOp { ADD, SUB, MUL, MIN, MAX }

    private static final int THREADS = 256;
    private final GpuDevice device;
    private final Map<String, PipelineSet> pipelines = new HashMap<>();
    private final ParameterLayout unaryLayout = ParameterLayout.of(GemmF16.KERNELS, "logic_unary");
    private final ParameterLayout binaryLayout = ParameterLayout.of(GemmF16.KERNELS, "logic_binary");
    private final ParameterLayout selectLayout = ParameterLayout.of(GemmF16.KERNELS, "logic_select");
    private final DebugChecks debug;
    private boolean closed;

    private Logic(GpuDevice device) { this.device = device; this.debug = new DebugChecks(device); }

    /** Logic operations on {@code device}; pipelines compile on first use. */
    public static Logic create(GpuDevice device) { return new Logic(Checks.notNull(device, "device")); }

    private GpuComputePipeline pipeline(String kernel, int op, int buffers) {
        if (closed) throw new IllegalStateException("closed");
        var accesses = new java.util.ArrayList<Access>();
        for (int i = 0; i < buffers - 2; i++) accesses.add(Access.READ);
        accesses.add(Access.WRITE);
        accesses.add(Access.READ_WRITE);
        return pipelines.computeIfAbsent(kernel + ":" + op, k -> PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline(kernel,
                List.copyOf(accesses), Set.of(GpuCapabilities.FP32), Set.of(),
                debug.constants(kernel.equals("logic_select") ? Map.of() : Map.of("OP", op)))))).get(0);   // select has no OP
    }

    /** The logic_binary operation number of a comparison. */
    static int binaryOp(Compare op, Operand type) { return op.ordinal() + (type == Operand.INT32 ? 6 : 0); }
    static int binaryOp(Logical op) { return 12 + op.ordinal(); }
    static int binaryOp(IntOp op) { return 15 + op.ordinal(); }

    /** Records {@code mask = a op b}, a and b FP32 or INT32 words broadcasting to the mask's shape. */
    public GpuCommandEncoder compare(GpuCommandEncoder encoder, Compare op, Operand type, GpuBuffer a, StridedView av, GpuBuffer b,
            StridedView bv, GpuBuffer mask, StridedView mv) {
        Checks.notNull(op, "op"); Checks.notNull(type, "type");
        return binary(encoder, binaryOp(op, type), "logic." + op.name().toLowerCase(), a, av, b, bv, mask, mv);
    }

    /** Records {@code y = a op b} on masks. */
    public GpuCommandEncoder logical(GpuCommandEncoder encoder, Logical op, GpuBuffer a, StridedView av, GpuBuffer b, StridedView bv,
            GpuBuffer y, StridedView yv) {
        Checks.notNull(op, "op");
        return binary(encoder, binaryOp(op), "logic." + op.name().toLowerCase(), a, av, b, bv, y, yv);
    }

    /** Records {@code y = a op b} on int32 words. */
    public GpuCommandEncoder integer(GpuCommandEncoder encoder, IntOp op, GpuBuffer a, StridedView av, GpuBuffer b, StridedView bv,
            GpuBuffer y, StridedView yv) {
        Checks.notNull(op, "op");
        return binary(encoder, binaryOp(op), "logic.int." + op.name().toLowerCase(), a, av, b, bv, y, yv);
    }

    /** Records {@code y = !x} on masks. */
    public GpuCommandEncoder not(GpuCommandEncoder encoder, GpuBuffer x, StridedView xv, GpuBuffer y, StridedView yv) {
        return unary(encoder, 0, "logic.not", x, xv, y, yv);
    }

    /** Records {@code mask = op(x)} for FP32 x. */
    public GpuCommandEncoder classify(GpuCommandEncoder encoder, Classify op, GpuBuffer x, StridedView xv, GpuBuffer mask, StridedView mv) {
        Checks.notNull(op, "op");
        return unary(encoder, 1 + op.ordinal(), "logic." + op.name().toLowerCase(), x, xv, mask, mv);
    }

    /**
     * Records {@code y = mask != 0 ? a : b} on 32-bit words of any type (FP32, INT32 or masks), the three inputs
     * broadcasting to y's shape. A NaN in the unselected input does not reach y.
     */
    public GpuCommandEncoder select(GpuCommandEncoder encoder, GpuBuffer mask, StridedView mv, GpuBuffer a, StridedView av, GpuBuffer b,
            StridedView bv, GpuBuffer y, StridedView yv) {
        var y4 = yv.withRank(4);
        var m4 = Views.broadcast(mv, y4, "mask");
        var a4 = Views.broadcast(av, y4, "a");
        var b4 = Views.broadcast(bv, y4, "b");
        mv.check(mask, 4, "mask"); av.check(a, 4, "a"); bv.check(b, 4, "b");
        yv.checkWritable(y, 4, "y");
        long count = Checks.u32(y4.elements(), "element count");
        if (count == 0) return encoder;
        var grid = Grid.forThreads(device.capabilities(), count, THREADS);
        var block = selectLayout.block().u32("count", count).u32("gridX", grid.gridX()).u32("mLength", DebugChecks.length(mask))
                .u32("aLength", DebugChecks.length(a)).u32("bLength", DebugChecks.length(b)).u32("yLength", DebugChecks.length(y));
        Views.broadcastStrides(block, "m", m4);
        Views.broadcastStrides(block, "a", a4);
        Views.broadcastStrides(block, "b", b4);
        byte[] params = block.view("y", y4).bytes();
        var pipeline = pipeline("logic_select", 0, 5);
        debug.before(encoder);
        encoder.dispatch(pipeline, Map.of(0, mask, 1, a, 2, b, 3, y, 4, debug.buffer()), params, grid.groups(), "logic.select");
        debug.after(encoder);
        return encoder;
    }

    private GpuCommandEncoder unary(GpuCommandEncoder encoder, int op, String label, GpuBuffer x, StridedView xv, GpuBuffer y, StridedView yv) {
        var y4 = yv.withRank(4);
        var x4 = Views.broadcast(xv, y4, "x");
        xv.check(x, 4, "x");
        yv.checkWritable(y, 4, "y");
        long count = Checks.u32(y4.elements(), "element count");
        if (count == 0) return encoder;
        var grid = Grid.forThreads(device.capabilities(), count, THREADS);
        byte[] params = unaryLayout.block().u32("count", count).u32("gridX", grid.gridX()).u32("xLength", DebugChecks.length(x))
                .u32("yLength", DebugChecks.length(y)).view("x", x4).view("y", y4).bytes();
        var pipeline = pipeline("logic_unary", op, 3);
        debug.before(encoder);
        encoder.dispatch(pipeline, Map.of(0, x, 1, y, 2, debug.buffer()), params, grid.groups(), label);
        debug.after(encoder);
        return encoder;
    }

    private GpuCommandEncoder binary(GpuCommandEncoder encoder, int op, String label, GpuBuffer a, StridedView av, GpuBuffer b,
            StridedView bv, GpuBuffer y, StridedView yv) {
        var y4 = yv.withRank(4);
        var a4 = Views.broadcast(av, y4, "a");
        var b4 = Views.broadcast(bv, y4, "b");
        av.check(a, 4, "a");
        bv.check(b, 4, "b");
        yv.checkWritable(y, 4, "y");
        long count = Checks.u32(y4.elements(), "element count");
        if (count == 0) return encoder;
        var grid = Grid.forThreads(device.capabilities(), count, THREADS);
        byte[] params = binaryLayout.block().u32("count", count).u32("gridX", grid.gridX()).u32("aLength", DebugChecks.length(a))
                .u32("bLength", DebugChecks.length(b)).u32("yLength", DebugChecks.length(y)).view("a", a4).view("b", b4)
                .view("y", y4).bytes();
        var pipeline = pipeline("logic_binary", op, 4);
        debug.before(encoder);
        encoder.dispatch(pipeline, Map.of(0, a, 1, b, 2, y, 3, debug.buffer()), params, grid.groups(), label);
        debug.after(encoder);
        return encoder;
    }

    /** The GPU's comparison on the CPU: 1 or 0, for the elements' bits (Java's float or int comparison). */
    public static int reference(Compare op, Operand type, int a, int b) {
        if (type == Operand.FP32) {
            float x = Float.intBitsToFloat(a), y = Float.intBitsToFloat(b);
            boolean r = switch (op) {
                case EQUAL -> x == y;
                case NOT_EQUAL -> x != y;
                case LESS -> x < y;
                case LESS_EQUAL -> x <= y;
                case GREATER -> x > y;
                case GREATER_EQUAL -> x >= y;
            };
            return r ? 1 : 0;
        }
        boolean r = switch (op) {
            case EQUAL -> a == b;
            case NOT_EQUAL -> a != b;
            case LESS -> a < b;
            case LESS_EQUAL -> a <= b;
            case GREATER -> a > b;
            case GREATER_EQUAL -> a >= b;
        };
        return r ? 1 : 0;
    }

    /** The GPU's logical operation on the CPU. */
    public static int reference(Logical op, int a, int b) {
        boolean x = a != 0, y = b != 0;
        return switch (op) {
            case AND -> x && y;
            case OR -> x || y;
            case XOR -> x != y;
        } ? 1 : 0;
    }

    /** The GPU's int32 arithmetic on the CPU. */
    public static int reference(IntOp op, int a, int b) {
        return switch (op) {
            case ADD -> a + b;
            case SUB -> a - b;
            case MUL -> a * b;
            case MIN -> Math.min(a, b);
            case MAX -> Math.max(a, b);
        };
    }

    /** The GPU's classification on the CPU, for the bits of an FP32 value. */
    public static int reference(Classify op, int bits) {
        float x = Float.intBitsToFloat(bits);
        return switch (op) {
            case IS_FINITE -> Float.isFinite(x);
            case IS_NAN -> Float.isNaN(x);
            case IS_INF -> Float.isInfinite(x);
        } ? 1 : 0;
    }

    /** The contract of every Logic operation: exact. */
    public static AccuracyContract contract() {
        return new AccuracyContract("Logic", List.of("32-bit words: FP32, INT32 or masks"), "none", "uint32 mask, INT32 or the selected word",
                Order.FIXED, new Bound(Bound.Kind.EXACT, 0, "bit for bit Logic.reference: integer arithmetic on the bits"), "Logic.reference");
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        List<AutoCloseable> all = new java.util.ArrayList<>(pipelines.values());
        all.add(debug);
        PipelineSet.closeAll(all.toArray(new AutoCloseable[0]));
    }
}
