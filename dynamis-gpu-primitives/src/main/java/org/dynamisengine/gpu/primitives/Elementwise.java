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
        SQRT, EXP, LOG, RECIPROCAL,
        /** {@code x / (1 + exp(-x))} (1.10.0). */
        SILU,
        /** GELU in its tanh form, as {@code GemmF16}'s epilogue (1.10.0). */
        GELU,
        /** {@code 1 / (1 + exp(-x))} (1.14.0). */
        SIGMOID,
        /** tanh (1.14.0): an odd polynomial below |x| = 0.625, {@code 1 - 2 / (exp(2|x|) + 1)} with x's sign above. */
        TANH;

        /** Whether the result is the same bits on every device ({@link #apply}). */
        public boolean exact() { return ordinal() <= RELU.ordinal(); }
    }

    /** Binary operations. */
    public enum Binary {
        ADD, SUB, MUL, DIV,
        /** Java {@code Math.max}. */
        MAX,
        /** Java {@code Math.min}. */
        MIN,
        /** {@code silu(a) * b}: the gated unit of SwiGLU, a the gate projection and b the up projection (1.10.0). */
        SWIGLU,
        /**
         * ReLU's backward pass (1.12.0): {@code a} the forward input x, {@code b} the output gradient dy;
         * {@code x > 0 ? dy : 0} (a NaN x gives 0).
         */
        RELU_BACKWARD,
        /**
         * SiLU's backward pass (1.12.0): {@code dy s (1 + x (1 - s))}, {@code s = 1 / (1 + exp(-x))}, for {@code a = x},
         * {@code b = dy}. SwiGLU's gradients compose from it: {@code d(gate) = SILU_BACKWARD(gate, dy * up)} and
         * {@code d(up) = dy * silu(gate)}.
         */
        SILU_BACKWARD,
        /** The backward pass of GELU's tanh form (1.12.0), for {@code a = x}, {@code b = dy}. */
        GELU_BACKWARD,
        /**
         * Sigmoid's backward pass (1.14.0) from the forward input: {@code dy s (1 - s)} for {@code a = x}, {@code b = dy},
         * computed as {@code dy e / (1 + e)^2}, {@code e = exp(-|x|)}, with no cancellation for large |x|.
         */
        SIGMOID_BACKWARD,
        /** tanh's backward pass (1.14.0) from the forward input: {@code dy (1 - tanh^2 x)} as {@code dy 4e / (1 + e)^2}, {@code e = exp(-2|x|)}. */
        TANH_BACKWARD;

        /** Whether the result is the same bits on every device ({@link #apply}). */
        public boolean exact() { return ordinal() <= MIN.ordinal() && this != DIV || this == RELU_BACKWARD; }
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
            case SILU -> x / (1f + (float) Math.exp(-x));
            case GELU -> {
                float u = GemmF16.GELU_SCALE * (x + GemmF16.GELU_CUBIC * x * x * x);
                yield 0.5f * x * (1f + (1f - 2f / ((float) Math.exp(2f * u) + 1f)));
            }
            case SIGMOID -> 1f / (1f + (float) Math.exp(-x));
            case TANH -> (float) Math.tanh(x);
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
            case SWIGLU -> a / (1f + (float) Math.exp(-a)) * b;
            case RELU_BACKWARD -> a > 0 ? b : 0f;
            case SILU_BACKWARD -> {
                float s = 1f / (1f + (float) Math.exp(-a));
                yield b * (s * (1f + a * (1f - s)));
            }
            case GELU_BACKWARD -> {
                float u = GemmF16.GELU_SCALE * (a + GemmF16.GELU_CUBIC * a * a * a);
                float t = 1f - 2f / ((float) Math.exp(2f * u) + 1f);
                float du = GemmF16.GELU_SCALE * (1f + 3f * GemmF16.GELU_CUBIC * a * a);
                yield b * (0.5f * (1f + t) + 0.5f * a * (1f - t * t) * du);
            }
            case SIGMOID_BACKWARD -> {
                double e = Math.exp(-Math.abs((double) a));
                yield (float) (b * (e / ((1 + e) * (1 + e))));
            }
            case TANH_BACKWARD -> {
                double e = Math.exp(-2 * Math.abs((double) a));
                yield (float) (b * (4 * e / ((1 + e) * (1 + e))));
            }
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
            case SILU -> Bound.formula("(7 + 2|x|) ULP: exp's (3 + 2|x| on Vulkan) through 1 + e and a 2.5 ULP division");
            case GELU -> Bound.formula("(8 + 4|u|) ULP of y plus |x| 2^-22 absolute, u = sqrt(2/pi)(x + 0.044715 x^3): exp's error, and "
                    + "the cancellation of 1 + tanh(u) for x below about -3");
            case SIGMOID -> Bound.formula("(6 + 2|x|) ULP: exp's (3 + 2|x| on Vulkan) through 1 + e and a 2.5 ULP division");
            case TANH -> Bound.formula("(8 + 4|x|) ULP: below |x| = 0.625 the polynomial's rounding (a few ULP); above, exp(2|x|)'s "
                    + "(3 + 4|x|) through the division and a subtraction that amplifies it at most 0.8 times");
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
        if (op == Binary.SWIGLU) {
            return new AccuracyContract(name, List.of("FP32", "FP32"), "none", "FP32", Order.DEVICE,
                    Bound.formula("(8 + 2|a|) ULP: SiLU's (7 + 2|a|), then one rounded product"), "");
        }
        if (op == Binary.SILU_BACKWARD) {
            return new AccuracyContract(name, List.of("FP32", "FP32"), "none", "FP32", Order.DEVICE,
                    Bound.formula("absolute, |dy| times the error of s (1 + x (1 - s)) carried through each step from sigmoid's "
                            + "(7 + 2|x|) ULP, each step's rounding relative to the magnitude of its terms (the derivative "
                            + "crosses zero near x = -1.28, where a relative bound cannot hold)"), "");
        }
        if (op == Binary.SIGMOID_BACKWARD || op == Binary.TANH_BACKWARD) {
            String e = op == Binary.SIGMOID_BACKWARD ? "exp(-|x|)'s (3 + 2|x|)" : "exp(-2|x|)'s (3 + 4|x|)";
            return new AccuracyContract(name, List.of("FP32", "FP32"), "none", "FP32", Order.DEVICE,
                    Bound.formula("(9 + " + (op == Binary.SIGMOID_BACKWARD ? "2" : "4") + "|x|) ULP of dx: " + e + " ULP, three roundings, a 2.5 ULP "
                            + "division and the product with dy; relative everywhere (no cancellation). Where the exponential is below 2^-126 (|x| above "
                            + "about 87, or 43.6 for tanh) a device that flushes subnormals may give 0"), "");
        }
        if (op == Binary.GELU_BACKWARD) {
            return new AccuracyContract(name, List.of("FP32", "FP32"), "none", "FP32", Order.DEVICE,
                    Bound.formula("absolute, |dy| times the error of 0.5 (1 + t) + 0.5 x (1 - t^2) du carried through each step "
                            + "from exp(2u)'s (4 + 4|u|) ULP and the rounding of u, relative to the magnitudes of the terms"), "");
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
