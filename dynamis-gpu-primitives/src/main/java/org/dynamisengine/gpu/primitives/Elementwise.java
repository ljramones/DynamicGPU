package org.dynamisengine.gpu.primitives;

import java.util.EnumMap;
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
 * Elementwise maps over FP32 strided views of rank up to 4: {@code y = op(x)} and {@code y = op(a, b)}. Inputs
 * broadcast along dimensions of extent 1 (views are aligned at their last dimension, as in NumPy). The output view
 * must not reach an element twice. Each operation compiles on first use, specialized for that operation.
 * <p>
 * <b>Accuracy</b> ({@link #contract}): the {@link Unary#exact() exact} operations and ADD, SUB, MUL, MAX and MIN match
 * {@link #apply} bit for bit on every device; MAX and MIN follow Java's {@code Math.max} and {@code Math.min} (NaN
 * propagates; max(-0, +0) = +0). The others use the device's function, within the Vulkan and Metal precision
 * requirements. Devices may flush subnormal values to zero.
 *
 * @since 1.7.0
 */
public final class Elementwise implements AutoCloseable {
    /** Unary operations. */
    public enum Unary {
        COPY, NEG, ABS, SQUARE,
        /** {@code x < 0 ? 0 : x}: NaN passes through; -0 stays -0. */
        RELU,
        SQRT, EXP, LOG, RECIPROCAL;

        /** Whether the result is the same bits on every device ({@link #apply}). */
        public boolean exact() { return ordinal() <= RELU.ordinal(); }
    }

    /** Binary operations. */
    public enum Binary {
        ADD, SUB, MUL, DIV,
        /** Java {@code Math.max}. */
        MAX,
        /** Java {@code Math.min}. */
        MIN;

        /** Whether the result is the same bits on every device ({@link #apply}). */
        public boolean exact() { return this != DIV; }
    }

    private static final int THREADS = 256;
    private final GpuDevice device;
    private final Map<Unary, PipelineSet> unary = new EnumMap<>(Unary.class);
    private final Map<Binary, PipelineSet> binary = new EnumMap<>(Binary.class);
    private final ParameterLayout unaryLayout = ParameterLayout.of(GemmF16.KERNELS, "ew_unary");
    private final ParameterLayout binaryLayout = ParameterLayout.of(GemmF16.KERNELS, "ew_binary");
    private final DebugChecks debug;
    private boolean closed;

    private Elementwise(GpuDevice device) { this.device = device; this.debug = new DebugChecks(device); }

    /** Elementwise maps on {@code device}; pipelines compile on first use. */
    public static Elementwise create(GpuDevice device) { return new Elementwise(Checks.notNull(device, "device")); }

    private GpuComputePipeline pipeline(Unary op) {
        if (closed) throw new IllegalStateException("closed");
        return unary.computeIfAbsent(op, o -> PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("ew_unary",
                List.of(Access.READ, Access.WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32), Set.of(),
                debug.constants(Map.of("OP", o.ordinal())))))).get(0);
    }

    private GpuComputePipeline pipeline(Binary op) {
        if (closed) throw new IllegalStateException("closed");
        return binary.computeIfAbsent(op, o -> PipelineSet.compile(device, List.of(GemmF16.KERNELS.pipeline("ew_binary",
                List.of(Access.READ, Access.READ, Access.WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32), Set.of(),
                debug.constants(Map.of("OP", o.ordinal())))))).get(0);
    }

    /** Records {@code y = op(x)}, x broadcasting to y's shape. */
    public GpuCommandEncoder unary(GpuCommandEncoder encoder, Unary op, GpuBuffer x, StridedView xv, GpuBuffer y, StridedView yv) {
        Checks.notNull(op, "op");
        var y4 = yv.withRank(4);
        var x4 = Views.broadcast(xv, y4, "x");
        xv.check(x, 4, "x");
        yv.checkWritable(y, 4, "y");
        long count = Checks.u32(y4.elements(), "element count");
        if (count == 0) return encoder;
        var grid = Grid.forThreads(device.capabilities(), count, THREADS);
        byte[] params = unaryLayout.block().u32("count", count).u32("gridX", grid.gridX()).u32("xLength", DebugChecks.length(x))
                .u32("yLength", DebugChecks.length(y)).view("x", x4).view("y", y4).bytes();
        var pipeline = pipeline(op);
        debug.before(encoder);
        encoder.dispatch(pipeline, Map.of(0, x, 1, y, 2, debug.buffer()), params, grid.groups(), "elementwise." + op.name().toLowerCase());
        debug.after(encoder);
        return encoder;
    }

    /** Records {@code y = op(a, b)}, a and b broadcasting to y's shape. */
    public GpuCommandEncoder binary(GpuCommandEncoder encoder, Binary op, GpuBuffer a, StridedView av, GpuBuffer b, StridedView bv,
            GpuBuffer y, StridedView yv) {
        Checks.notNull(op, "op");
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
        var pipeline = pipeline(op);
        debug.before(encoder);
        encoder.dispatch(pipeline, Map.of(0, a, 1, b, 2, y, 3, debug.buffer()), params, grid.groups(), "elementwise." + op.name().toLowerCase());
        debug.after(encoder);
        return encoder;
    }

    /** The CPU result: bit for bit the GPU's for {@link Unary#exact()} operations; Java's {@code Math} otherwise. */
    public static float apply(Unary op, float x) {
        return switch (op) {
            case COPY -> x;
            case NEG -> Float.intBitsToFloat(Float.floatToRawIntBits(x) ^ 0x80000000);
            case ABS -> Float.intBitsToFloat(Float.floatToRawIntBits(x) & 0x7FFFFFFF);
            case SQUARE -> x * x;
            case RELU -> x < 0 ? 0f : x;
            case SQRT -> (float) Math.sqrt(x);
            case EXP -> (float) Math.exp(x);
            case LOG -> (float) Math.log(x);
            case RECIPROCAL -> 1f / x;
        };
    }

    /** The CPU result: bit for bit the GPU's for {@link Binary#exact()} operations; the correctly rounded quotient for DIV. */
    public static float apply(Binary op, float a, float b) {
        return switch (op) {
            case ADD -> a + b;
            case SUB -> a - b;
            case MUL -> a * b;
            case DIV -> a / b;
            case MAX -> Math.max(a, b);
            case MIN -> Math.min(a, b);
        };
    }

    /** The accuracy contract of {@code op}. */
    public static AccuracyContract contract(Unary op) {
        String name = "Elementwise " + op;
        if (op.exact()) {
            return new AccuracyContract(name, List.of("FP32"), "none", "FP32", Order.FIXED, Bound.exact(), "Elementwise.apply");
        }
        Bound bound = switch (op) {
            case SQRT -> Bound.ulps(5, "Vulkan: 1 / inversesqrt (2 ULP) through a 2.5 ULP division; Metal: correctly rounded");
            case EXP -> Bound.formula("max(4, 3 + 2|x|) ULP: Vulkan 3 + 2|x| ULP, Metal (precise) 4 ULP");
            case LOG -> Bound.formula("4 ULP for x outside [0.5, 2]; inside, absolute 2^-21 or 4 ULP, whichever is larger");
            default -> Bound.ulps(2.5, "Vulkan: 1 / x within 2.5 ULP for |x| in [2^-126, 2^126]");
        };
        return new AccuracyContract(name, List.of("FP32"), "none", "FP32", Order.DEVICE, bound, "");
    }

    /** The accuracy contract of {@code op}. */
    public static AccuracyContract contract(Binary op) {
        String name = "Elementwise " + op;
        if (op.exact()) {
            return new AccuracyContract(name, List.of("FP32", "FP32"), "none", "FP32, correctly rounded", Order.FIXED,
                    Bound.exact(), "Elementwise.apply");
        }
        return new AccuracyContract(name, List.of("FP32", "FP32"), "none", "FP32", Order.DEVICE,
                Bound.ulps(2.5, "Vulkan: a / b within 2.5 ULP for |b| in [2^-126, 2^126]"), "");
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        List<AutoCloseable> all = new java.util.ArrayList<>(unary.values());
        all.addAll(binary.values());
        all.add(debug);
        PipelineSet.closeAll(all.toArray(new AutoCloseable[0]));
    }
}
