package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.Grid;
import org.dynamisengine.gpu.api.compute.kit.ParameterLayout;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Embedding lookup (1.10.0): {@code y[t] = table[tokens[t]]}, rows of a {@code vocab x dim} table gathered by token,
 * with the tokens (uint32) read from a GPU buffer, so the {@link Sampler}'s output feeds the next step directly. The
 * table is FP32, FP16 or a packed {@link QuantizedMatrix} ({@code vocab x dim}, any format), dequantized exactly as
 * {@link QuantizedMatrix#value}: the same bits on every device. A token beyond the vocabulary fails the submission
 * with {@link ComputeException.Code#KERNEL_ERROR}.
 */
public final class Embedding implements AutoCloseable {
    /** A table's element type: FP32, FP16, or a quantized matrix. */
    public record Table(int vocab, int dim, Precision precision, QuantizedMatrix quantized) {
        public Table {
            if (quantized != null) {
                if (quantized.rows() != vocab || quantized.cols() != dim) throw new IllegalArgumentException("the quantized matrix must be vocab x dim");
            } else {
                Checks.notNull(precision, "precision");
                if (vocab <= 0 || dim <= 0) throw new IllegalArgumentException("dimensions must be positive");
            }
        }
        /** A table of FP32 or FP16 rows. */
        public static Table of(int vocab, int dim, Precision precision) { return new Table(vocab, dim, precision, null); }
        /** A quantized table. */
        public static Table quantized(QuantizedMatrix m) { return new Table(m.rows(), m.cols(), null, m); }
        /** The bytes its buffer holds. */
        public long bytes() { return quantized != null ? quantized.bytes() : (long) vocab * dim * precision.bytes(); }
    }

    private final GpuDevice device;
    private final GpuComputePipeline pipeline;
    private final Precision output;
    private final ParameterLayout layout;

    private Embedding(GpuDevice device, GpuComputePipeline pipeline, Precision output) {
        this.device = device; this.pipeline = pipeline; this.output = output;
        this.layout = ParameterLayout.of(GemmF16.KERNELS, "embed");
    }

    /** Compiles the gather writing {@code output}; FP16 output needs {@link GpuCapabilities#FP16}. */
    public static Embedding create(GpuDevice device, Precision output) {
        Checks.notNull(device, "device");
        var types = output == Precision.FP16 ? Set.of(GpuCapabilities.FP32, GpuCapabilities.FP16) : Set.of(GpuCapabilities.FP32);
        var accesses = List.of(Access.READ, Access.READ, Access.READ_WRITE, Access.READ_WRITE);
        return new Embedding(device, device.pipeline(GemmF16.KERNELS.pipeline(output == Precision.FP32 ? "embed" : "embed_h", accesses, types, Set.of())), output);
    }

    /** Records the gather of {@code count} tokens' rows into {@code y} (row t at {@code t * ldy}). */
    public GpuCommandEncoder encode(GpuCommandEncoder encoder, GpuBuffer tokens, int count, Table tab, GpuBuffer table, GpuBuffer y, int ldy) {
        Checks.notNull(tab, "table");
        if (count <= 0) throw new IllegalArgumentException("count must be positive: " + count);
        if (ldy < tab.dim()) throw new IllegalArgumentException("ldy smaller than a row: " + ldy);
        if (tab.precision() == Precision.FP16 && ((long) tab.vocab() * tab.dim()) % 2 != 0) throw new IllegalArgumentException("an FP16 table must hold whole words");
        Checks.range(tokens, 0, count, 4, "tokens");
        Checks.range(table, 0, (tab.bytes() + 3) / 4, 4, "table");
        Checks.range(y, 0, (long) (count - 1) * ldy + tab.dim(), output.bytes(), "y");
        var q = tab.quantized();
        long threads = (long) count * tab.dim();
        var grid = Grid.forThreads(device.capabilities(), threads, 256);
        byte[] params = layout.block().u32("count", threads).u32("dim", tab.dim()).u32("vocab", tab.vocab()).u32("ldy", ldy)
                .u32("kind", q != null ? 2 : tab.precision() == Precision.FP32 ? 0 : 1).u32("qformat", q == null ? 0 : q.format().ordinal())
                .u32("qblock", q == null ? 32 : q.block()).u32("qparams", q == null ? 0 : q.parameterOffset() / 4)
                .u32("qhigh", q == null ? 0 : q.highOffset() / 4).u32("gridX", grid.gridX()).bytes();
        try (var errors = device.allocate(KernelErrors.descriptor().withLabel("embedding-errors"))) {
            encoder.resetErrors(errors);
            encoder.dispatch(pipeline, Map.of(0, tokens, 1, table, 2, y, 3, errors), params, grid.groups(), "embedding");
            encoder.checkErrors(errors);
        }
        return encoder;
    }

    /** The contract: exact. */
    public static AccuracyContract contract(Precision output) {
        return new AccuracyContract("Embedding, " + output + " out", List.of("FP32, FP16 or quantized"), "none",
                output == Precision.FP32 ? "FP32" : "rounded once to half", AccuracyContract.Order.FIXED,
                new AccuracyContract.Bound(AccuracyContract.Bound.Kind.EXACT, 0, "the table's value (QuantizedMatrix.value for a quantized table)"), "");
    }

    @Override public void close() { pipeline.close(); }
}
