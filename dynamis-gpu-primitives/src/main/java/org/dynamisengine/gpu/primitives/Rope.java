package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.Grid;
import org.dynamisengine.gpu.api.compute.kit.ParameterLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Rotary position embedding (1.10.0) of the heads of each token, for queries and keys: element pairs {@code (a, b)}
 * of a head rotate by the angle of the token's position and the pair,
 * {@code a' = a cos - b sin, b' = a sin + b cos}. The positions come from a GPU buffer (uint32 per token), so a
 * decode step's position never returns to the host.
 * <p>
 * <b>Table.</b> The cosines and sines come from a {@link Table} computed on the host in double precision (GPU sine is
 * coarse at large angles: Vulkan allows 2^-11 absolute), so the kernel is two correctly rounded products and an
 * unfused add per element: the same bits on every device, bit for bit {@link #reference}. Any frequency scheme
 * (linear, NTK, YaRN, llama 3) is a table: {@link Table#standard} and {@link Table#scaled} build the common ones,
 * {@link Table#of} takes any.
 * <p>
 * <b>Layouts.</b> {@link Layout#INTERLEAVED} rotates {@code (2i, 2i + 1)} (GPT-J); {@link Layout#HALVES} rotates
 * {@code (i, i + rotaryDim / 2)} (GPT-NeoX, llama). Elements from rotaryDim to headDim pass through.
 */
public final class Rope implements AutoCloseable {
    /** Which elements form a pair. */
    public enum Layout { INTERLEAVED, HALVES }

    /**
     * The cosines and sines of positions {@code 0 .. positions - 1} and pairs {@code 0 .. rotaryDim / 2 - 1}: element
     * {@code 2 (p rotaryDim / 2 + i)} the cosine and {@code + 1} the sine, as FP32.
     */
    public record Table(int positions, int rotaryDim, float[] cosSin) {
        public Table {
            if (positions <= 0 || rotaryDim <= 0 || rotaryDim % 2 != 0) throw new IllegalArgumentException("positions and an even rotaryDim");
            if (cosSin.length != (long) positions * rotaryDim) throw new IllegalArgumentException("cosSin must hold positions x rotaryDim");
            cosSin = cosSin.clone();
        }
        /** The standard table: angle {@code p theta^(-2i / rotaryDim)}, rounded from double. */
        public static Table standard(int positions, int rotaryDim, double theta) { return scaled(positions, rotaryDim, theta, 1); }
        /** Linear position interpolation: angle {@code (p / scale) theta^(-2i / rotaryDim)}. */
        public static Table scaled(int positions, int rotaryDim, double theta, double scale) {
            float[] t = new float[positions * rotaryDim];
            for (int p = 0; p < positions; p++) {
                for (int i = 0; i < rotaryDim / 2; i++) {
                    double angle = p / scale * Math.pow(theta, -2.0 * i / rotaryDim);
                    t[2 * (p * rotaryDim / 2 + i)] = (float) Math.cos(angle);
                    t[2 * (p * rotaryDim / 2 + i) + 1] = (float) Math.sin(angle);
                }
            }
            return new Table(positions, rotaryDim, t);
        }
        /** A table of the given values. */
        public static Table of(int positions, int rotaryDim, float[] cosSin) { return new Table(positions, rotaryDim, cosSin); }
        /** Its bytes for a GPU buffer (little-endian FP32). */
        public byte[] bytes() {
            var b = ByteBuffer.allocate(4 * cosSin.length).order(ByteOrder.LITTLE_ENDIAN);
            for (float v : cosSin) b.putFloat(v);
            return b.array();
        }
        float cos(int p, int i) { return cosSin[2 * (p * rotaryDim / 2 + i)]; }
        float sin(int p, int i) { return cosSin[2 * (p * rotaryDim / 2 + i) + 1]; }
    }

    /**
     * Tokens of {@code heads} heads of {@code headDim}: token t's heads start at {@code t * ldx} (in y, {@code t * ldy}),
     * head h at {@code h * strideHead} from there.
     */
    public record Shape(int tokens, int heads, int headDim, int rotaryDim, int ldx, int ldy, int strideHead, Layout layout) {
        public Shape {
            Checks.notNull(layout, "layout");
            if (tokens <= 0 || heads <= 0 || headDim <= 0 || headDim % 2 != 0) throw new IllegalArgumentException("positive sizes and an even headDim");
            if (rotaryDim <= 0 || rotaryDim % 2 != 0 || rotaryDim > headDim) throw new IllegalArgumentException("rotaryDim must be even and at most headDim: " + rotaryDim);
            if (strideHead < headDim || ldx < (long) (heads - 1) * strideHead + headDim || ldy < (long) (heads - 1) * strideHead + headDim)
                throw new IllegalArgumentException("heads overlap or exceed a token's row");
        }
        /** Densely packed: {@code tokens x heads x headDim}, the whole head rotated. */
        public static Shape of(int tokens, int heads, int headDim, Layout layout) {
            return new Shape(tokens, heads, headDim, headDim, heads * headDim, heads * headDim, headDim, layout);
        }
        long elements(int ld) { return (long) (tokens - 1) * ld + (long) (heads - 1) * strideHead + headDim; }
    }

    private final GpuDevice device;
    private final GpuComputePipeline pipeline;
    private final Precision input, output;
    private final ParameterLayout layout;
    private org.dynamisengine.gpu.api.compute.kit.PipelineSet placeholders;

    private Rope(GpuDevice device, GpuComputePipeline pipeline, Precision input, Precision output) {
        this.device = device; this.pipeline = pipeline; this.input = input; this.output = output;
        this.layout = ParameterLayout.of(GemmF16.KERNELS, "rope_f32");
    }

    static String kernel(Precision input, Precision output) {
        return "rope_" + (input == Precision.FP32 ? "f32" : "f16") + (output == Precision.FP16 ? "_h" : "");
    }

    /** Compiles the kernel for {@code input} and {@code output}; FP16 needs {@link GpuCapabilities#FP16}. */
    public static Rope create(GpuDevice device, Precision input, Precision output) {
        Checks.notNull(device, "device");
        var types = input == Precision.FP16 || output == Precision.FP16 ? Set.of(GpuCapabilities.FP32, GpuCapabilities.FP16) : Set.of(GpuCapabilities.FP32);
        var accesses = List.of(Access.READ, Access.READ, Access.READ, Access.READ_WRITE, Access.READ_WRITE);
        return new Rope(device, device.pipeline(GemmF16.KERNELS.pipeline(kernel(input, output), accesses, types, Set.of())), input, output);
    }

    /**
     * Records the rotation of {@code x} into {@code y} (which may be x when the types match), with each token's
     * position from {@code positions} (uint32) and the table's values in {@code table} ({@link Table#bytes}). A
     * position beyond the table fails the submission with {@link ComputeException.Code#KERNEL_ERROR}.
     */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer x, GpuBuffer positions, Table tab, GpuBuffer table, GpuBuffer y, Shape s) {
        Checks.notNull(tab, "table"); Checks.notNull(s, "shape");
        if (tab.rotaryDim() != s.rotaryDim()) throw new IllegalArgumentException("the table is for rotaryDim " + tab.rotaryDim());
        Checks.range(x, 0, s.elements(s.ldx()), input.bytes(), "x");
        Checks.range(y, 0, s.elements(s.ldy()), output.bytes(), "y");
        Checks.range(positions, 0, s.tokens(), 4, "positions");
        Checks.range(table, 0, (long) tab.positions() * tab.rotaryDim(), 4, "table");
        boolean inPlace = Norms.same(x, y);
        if (inPlace && (input != output || s.ldx() != s.ldy())) throw new IllegalArgumentException("in place, x and y share their type and leading dimension");
        long count = (long) s.tokens() * s.heads() * s.headDim() / 2;
        var grid = Grid.forThreads(device.capabilities(), count, 256);
        byte[] params = layout.block().u32("tokens", s.tokens()).u32("heads", s.heads()).u32("headDim", s.headDim()).u32("rotaryDim", s.rotaryDim())
                .u32("ldx", s.ldx()).u32("ldy", s.ldy()).u32("strideHead", s.strideHead()).u32("layout", s.layout().ordinal())
                .u32("positions", tab.positions()).u32("count", count).u32("gridX", grid.gridX()).bool("inPlace", inPlace).bytes();
        try (var errors = device.allocate(KernelErrors.descriptor().withLabel("rope-errors"))) {
            encoder.resetErrors(errors);
            if (placeholders == null) placeholders = org.dynamisengine.gpu.api.compute.kit.PipelineSet.compile(device, List.of());
            encoder.dispatch(pipeline, Map.of(0, inPlace ? placeholders.placeholder() : x, 1, positions, 2, table, 3, y, 4, errors), params, grid.groups(), "rope");
            encoder.checkErrors(errors);
        }
        return encoder;
    }

    /**
     * The GPU's result on the CPU, bit for bit: {@code x} and {@code y} hold the elements as floats (FP16 values
     * widened), {@code y} the output's rounding applied; y may be x.
     */
    public static void reference(float[] x, int[] positions, Table tab, float[] y, Shape s, Precision output) {
        int half = s.rotaryDim() / 2;
        for (int t = 0; t < s.tokens(); t++) {
            int p = positions[t];
            for (int h = 0; h < s.heads(); h++) {
                int bx = t * s.ldx() + h * s.strideHead(), by = t * s.ldy() + h * s.strideHead();
                float[] head = new float[s.headDim()];
                System.arraycopy(x, bx, head, 0, s.headDim());
                for (int i = 0; i < half; i++) {
                    int ia = s.layout() == Layout.INTERLEAVED ? 2 * i : i, ib = s.layout() == Layout.INTERLEAVED ? 2 * i + 1 : i + half;
                    float a = head[ia], b = head[ib], c = tab.cos(p, i), sn = tab.sin(p, i);
                    y[by + ia] = round(a * c - b * sn, output);
                    y[by + ib] = round(a * sn + b * c, output);
                }
                for (int e = s.rotaryDim(); e < s.headDim(); e++) y[by + e] = round(head[e], output);
            }
        }
    }

    private static float round(float v, Precision output) { return output == Precision.FP16 ? Float.float16ToFloat(Float.floatToFloat16(v)) : v; }

    /** The accuracy contract: exact, given the table. */
    public static AccuracyContract contract(Precision input, Precision output) {
        return new AccuracyContract("Rope, " + input + " in, " + output + " out", List.of(input.name(), "FP32 table"), "none",
                output == Precision.FP32 ? "FP32" : "FP32 rounded once to half", AccuracyContract.Order.FIXED,
                new AccuracyContract.Bound(AccuracyContract.Bound.Kind.EXACT, 0, "bit for bit Rope.reference: two correctly rounded products and an unfused add"),
                "Rope.reference");
    }

    @Override public void close() { org.dynamisengine.gpu.api.compute.kit.PipelineSet.closeAll(pipeline, placeholders); }
}
