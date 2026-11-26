package org.dynamisengine.gpu.primitives;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.Checks;
import org.dynamisengine.gpu.api.compute.kit.Grid;
import org.dynamisengine.gpu.api.compute.kit.ParameterLayout;
import org.dynamisengine.gpu.api.compute.kit.PipelineSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Block quantization on the GPU (1.9.0): FP32 values to a {@link QuantizedMatrix} ({@link #pack}; Q8, Q4 and Q4M) and
 * back ({@link #unpack}; every format, the K formats since 1.9.1), bit for bit as {@link QuantizedMatrix#quantize} and
 * {@link QuantizedMatrix#dequantize} on every device. {@link #unpackF16} (1.15.0) writes the dequantized values rounded
 * to FP16, for an FP16 GEMM on unpacked weights. Use it to quantize weights that are already on the GPU, at model load, or to check a packed matrix.
 * <p>
 * The values must be finite and each block's scale must fit FP16: a block that breaks that is written as zeros and
 * fails the submission with {@link ComputeException.Code#KERNEL_ERROR}. Devices that flush FP32 subnormals (Apple
 * GPUs) flush subnormal values too, which the CPU encoder does not.
 */
public final class Quantize implements AutoCloseable {
    private final GpuDevice device;
    private final PipelineSet pipelines;
    private final ParameterLayout packLayout, unpackLayout, unpackF16Layout;

    private Quantize(GpuDevice device) {
        this.device = device;
        var k = GemmF16.KERNELS;
        this.pipelines = PipelineSet.compile(device, List.of(
                k.pipeline("quant_pack", List.of(Access.READ, Access.WRITE, Access.READ_WRITE), Set.of(GpuCapabilities.FP32), Set.of()),
                k.pipeline("quant_unpack", List.of(Access.READ, Access.WRITE), Set.of(GpuCapabilities.FP32), Set.of()),
                k.pipeline("quant_unpack_h", List.of(Access.READ, Access.WRITE), Set.of(GpuCapabilities.FP32), Set.of())));
        this.packLayout = ParameterLayout.of(k, "quant_pack");
        this.unpackLayout = ParameterLayout.of(k, "quant_unpack");
        this.unpackF16Layout = ParameterLayout.of(k, "quant_unpack_h");
    }

    /** Compiles the kernels for {@code device}. */
    public static Quantize create(GpuDevice device) { return new Quantize(Checks.notNull(device, "device")); }

    /**
     * Records the quantization of {@code values} ({@code m.rows() x m.cols()} FP32, row-major) into {@code packed}
     * ({@code m.bytes()}). A non-finite value, or a block scale beyond FP16, fails the submission.
     */
    public GpuCommandEncoder pack(GpuCommandEncoder encoder, GpuBuffer values, QuantizedMatrix m, GpuBuffer packed) {
        Checks.notNull(m, "matrix");
        if (m.format().kQuant())
            throw new IllegalArgumentException("there is no GPU encoder for " + m.format() + ": use QuantizedMatrix.quantize or import llama.cpp's blocks (GgufBlocks)");
        Checks.range(values, 0, (long) m.rows() * m.cols(), 4, "values");
        Checks.range(packed, 0, m.bytes() / 4, 4, "packed");
        long blocks = (long) m.rows() * m.blocksPerRow();
        var grid = Grid.forThreads(device.capabilities(), Math.ceilDiv(blocks, 2), 64);
        byte[] params = packLayout.block().u32("rows", m.rows()).u32("cols", m.cols()).u32("format", m.format().ordinal())
                .u32("block", m.block()).u32("blocks", blocks).u32("qparams", m.parameterOffset() / 4).u32("gridX", grid.gridX()).bytes();
        try (var errors = device.allocate(KernelErrors.descriptor().withLabel("quantize-errors"))) {
            encoder.resetErrors(errors);
            encoder.dispatch(pipelines.get(0), Map.of(0, values, 1, packed, 2, errors), params, grid.groups(), "quantize.pack");
            encoder.checkErrors(errors);
        }
        return encoder;
    }

    /** Records the dequantization of {@code packed} into {@code values} ({@code m.rows() x m.cols()} FP32, row-major). */
    public GpuCommandEncoder unpack(GpuCommandEncoder encoder, QuantizedMatrix m, GpuBuffer packed, GpuBuffer values) {
        Checks.notNull(m, "matrix");
        Checks.range(packed, 0, m.bytes() / 4, 4, "packed");
        long count = (long) m.rows() * m.cols();
        Checks.range(values, 0, count, 4, "values");
        var grid = Grid.forThreads(device.capabilities(), count, 256);
        byte[] params = unpackLayout.block().u32("rows", m.rows()).u32("cols", m.cols()).u32("format", m.format().ordinal())
                .u32("block", m.block()).u32("count", count).u32("qparams", m.parameterOffset() / 4).u32("gridX", grid.gridX())
                .u32("qhigh", m.highOffset() / 4).bytes();
        return encoder.dispatch(pipelines.get(1), Map.of(0, packed, 1, values), params, grid.groups(), "quantize.unpack");
    }

    /**
     * Records the dequantization of {@code packed} into {@code values} ({@code m.rows() x m.cols()} FP16, row-major),
     * each value rounded once from {@link #unpack}'s FP32 value to the nearest FP16 (ties to even; overflow to
     * infinity): bit for bit {@code Float.floatToFloat16} of {@link QuantizedMatrix#dequantize}. A quarter of
     * {@link #unpack}'s output bytes followed by a cast, for running an FP16 GEMM on unpacked weights. @since 1.15.0
     */
    public GpuCommandEncoder unpackF16(GpuCommandEncoder encoder, QuantizedMatrix m, GpuBuffer packed, GpuBuffer values) {
        Checks.notNull(m, "matrix");
        Checks.range(packed, 0, m.bytes() / 4, 4, "packed");
        long count = (long) m.rows() * m.cols();   // a multiple of 8, as every block size is
        Checks.range(values, 0, count, 2, "values");
        var grid = Grid.forThreads(device.capabilities(), count / 8, 256);   // 8 elements per thread
        byte[] params = unpackF16Layout.block().u32("rows", m.rows()).u32("cols", m.cols()).u32("format", m.format().ordinal())
                .u32("block", m.block()).u32("count", count).u32("qparams", m.parameterOffset() / 4).u32("gridX", grid.gridX())
                .u32("qhigh", m.highOffset() / 4).bytes();
        return encoder.dispatch(pipelines.get(2), Map.of(0, packed, 1, values), params, grid.groups(), "quantize.unpack.f16");
    }

    /** The contract of {@link #pack} and {@link #unpack}: bit for bit the CPU encoder and decoder. */
    public static AccuracyContract contract(QuantizedMatrix.Format format) {
        return new AccuracyContract("Quantize " + format, List.of("FP32"), "none: block statistics, one correctly rounded multiply per scale",
                format + " codes and FP16 parameters; dequantized " + (format == QuantizedMatrix.Format.Q4M ? "fl(q d + m)" : "q d"),
                AccuracyContract.Order.FIXED,
                new AccuracyContract.Bound(AccuracyContract.Bound.Kind.EXACT, 0, "bit for bit QuantizedMatrix.quantize and dequantize"),
                "QuantizedMatrix.quantize");
    }

    @Override public void close() { pipelines.close(); }
}
