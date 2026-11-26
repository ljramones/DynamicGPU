package org.dynamisengine.gpu.primitives;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;
import org.dynamisengine.gpu.api.compute.BufferDescriptor;
import org.dynamisengine.gpu.api.compute.BufferUsage;
import org.dynamisengine.gpu.api.compute.GpuBuffer;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.gpu.api.compute.KernelErrors;
import org.dynamisengine.gpu.api.compute.Placement;
import org.dynamisengine.gpu.api.compute.kit.StridedView;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * -Dkit.benchmark=true: the kit and training primitives that no other benchmark times, at LLM-training sizes.
 * Each line: {@code kitbench|primitive|operation|shape|gpuUs|encodeUs|wallUs|bytes}, bytes being the minimum
 * traffic (each input read once, each output written once).
 */
abstract class KitBenchmarkConformance {
    protected abstract GpuProvider provider();
    protected GpuDevice device;

    @BeforeEach void open() { device = provider().adapters().getFirst().open(); }
    @AfterEach void close() { device.close(); }

    GpuBuffer buffer(long bytes) {
        return device.allocate(new BufferDescriptor(Math.max(16, bytes), EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE,
                BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE));
    }

    private void init(Consumer<GpuCommandEncoder> body) {
        try (var e = device.encoder()) {
            body.accept(e);
            try (var sub = device.queue().submit(e)) {
                if (!sub.await(java.time.Duration.ofSeconds(60))) throw new IllegalStateException("init timed out");
            }
        }
    }

    private static byte[] ints(int n, java.util.function.IntUnaryOperator f) {
        var b = ByteBuffer.allocate(4 * n).order(ByteOrder.nativeOrder());
        for (int i = 0; i < n; i++) b.putInt(f.applyAsInt(i));
        return b.array();
    }

    private void time(String primitive, String op, String shape, long bytes, Consumer<GpuCommandEncoder> body) {
        var t = GpuBenchmark.measure(device, body);
        System.out.printf(Locale.ROOT, "kitbench|%s|%s|%s|%.2f|%.2f|%.2f|%d%n", primitive, op, shape, t.gpuMicros(), t.encodeMicros(), t.wallMicros(), bytes);
    }

    static final int ONE = Float.floatToRawIntBits(1f), HALF_ONES = 0x3C003C00;

    @Test void benchmark() {
        Assumptions.assumeTrue(Boolean.getBoolean("kit.benchmark"), "set -Dkit.benchmark=true");
        System.out.println("kitbench-device|" + device.adapter().name() + "|" + device.adapter().backend());
        elementwise();
        reductions();
        softmaxAndLoss();
        norms();
        rope();
        gatherScatter();
        optimizer();
        logicCastChecks();
    }

    private void elementwise() {
        int n = 1 << 24;
        var v = StridedView.contiguous(n);
        String shape = "16M fp32";
        try (var ew = Elementwise.create(device); var a = buffer(4L * n); var b = buffer(4L * n); var y = buffer(4L * n)) {
            init(e -> e.fill(a, ONE).fill(b, ONE));
            for (var op : new Elementwise.Unary[] {Elementwise.Unary.COPY, Elementwise.Unary.RELU, Elementwise.Unary.EXP,
                    Elementwise.Unary.SILU, Elementwise.Unary.GELU, Elementwise.Unary.SIGMOID, Elementwise.Unary.TANH})
                time("Elementwise", "unary " + op, shape, 8L * n, e -> ew.unary(e, op, a, v, y, v));
            for (var op : new Elementwise.Binary[] {Elementwise.Binary.ADD, Elementwise.Binary.MUL, Elementwise.Binary.SWIGLU,
                    Elementwise.Binary.SILU_BACKWARD, Elementwise.Binary.GELU_BACKWARD, Elementwise.Binary.TANH_BACKWARD})
                time("Elementwise", "binary " + op, shape, 12L * n, e -> ew.binary(e, op, a, v, b, v, y, v));
        }
    }

    private void reductions() {
        int r = 4096, c = 4096;
        var xv = StridedView.matrix(r, c);
        var yv = StridedView.contiguous(r);
        String shape = "4096x4096 fp32";
        try (var reduce = Reduce.create(device); var scan = Scan.create(device); var x = buffer(4L * r * c); var y = buffer(4L * r * c);
             var idx = buffer(4L * r); var dy = buffer(4L * r)) {
            init(e -> e.fill(x, ONE).fill(dy, ONE));
            time("Reduce", "rows SUM", shape, 4L * r * c, e -> reduce.rows(e, Reduce.Op.SUM, x, xv, y, yv));
            time("Reduce", "rows MAX", shape, 4L * r * c, e -> reduce.rows(e, Reduce.Op.MAX, x, xv, y, yv));
            time("Reduce", "axis 0 MEAN", shape, 4L * r * c, e -> reduce.axis(e, Reduce.Op.MEAN, x, xv, 0, y, StridedView.contiguous(c)));
            time("Reduce", "logSumExp rows", shape, 4L * r * c, e -> reduce.logSumExp(e, x, xv, y, yv));
            time("Reduce", "argmaxAxis 1", shape, 4L * r * c, e -> reduce.argmaxAxis(e, x, xv, 1, idx, yv));
            init(e -> reduce.argmaxAxis(e, x, xv, 1, idx, yv));
            time("Reduce", "maxBackward axis 1", shape, 4L * r * c, e -> reduce.maxBackward(e, idx, yv, dy, yv, 1, y, xv));
            time("Scan", "rows SUM", shape, 8L * r * c, e -> scan.rows(e, Reduce.Op.SUM, x, xv, y, xv));
        }
    }

    private void softmaxAndLoss() {
        try (var softmax = Softmax.create(device, Precision.FP32)) {
            // attention scores: 32 heads x 512 queries x 512 keys, causal
            var att = Softmax.Shape.of(512, 512).causal(0).batched(32);
            long attBytes = 4L * 32 * 512 * 512;
            try (var s = buffer(attBytes); var y = buffer(attBytes); var dy = buffer(attBytes); var dx = buffer(attBytes)) {
                init(e -> e.fill(s, 0).fill(dy, ONE));
                time("Softmax", "forward causal", "32x512x512", 2 * attBytes, e -> softmax.encode(e, s, y, att));
                init(e -> softmax.encode(e, s, y, att));
                time("Softmax", "backward causal", "32x512x512", 3 * attBytes, e -> softmax.backward(e, y, dy, dx, att));
            }
            // logits: 512 tokens x 32000 vocab
            var lg = Softmax.Shape.of(512, 32000);
            long lgBytes = 4L * 512 * 32000;
            try (var s = buffer(lgBytes); var y = buffer(lgBytes)) {
                init(e -> e.fill(s, 0));
                time("Softmax", "forward", "512x32000", 2 * lgBytes, e -> softmax.encode(e, s, y, lg));
            }
        }
        try (var ce = CrossEntropy.create(device); var logits = buffer(4L * 512 * 32000); var targets = buffer(4L * 512);
             var loss = buffer(4L * 512); var dl = buffer(4L * 512 * 32000)) {
            init(e -> e.fill(logits, 0).fill(targets, 5));
            long bytes = 4L * 512 * 32000;
            time("CrossEntropy", "loss", "512x32000", bytes, e -> ce.encode(e, logits, targets, loss, null, 1f, CrossEntropy.Shape.of(512, 32000)));
            time("CrossEntropy", "loss + gradient", "512x32000", 2 * bytes,
                    e -> ce.encode(e, logits, targets, loss, dl, 1f / 512, CrossEntropy.Shape.of(512, 32000)));
        }
    }

    private void norms() {
        int rows = 512, n = 4096;
        long bytes = 4L * rows * n;
        try (var norms = Norms.create(device, Precision.FP32, Precision.FP32); var x = buffer(bytes); var r = buffer(bytes); var h = buffer(bytes);
             var w = buffer(4L * n); var bias = buffer(4L * n); var y = buffer(bytes); var dy = buffer(bytes); var dv = buffer(bytes);
             var dw = buffer(4L * n); var db = buffer(4L * n)) {
            init(e -> e.fill(x, ONE).fill(r, ONE).fill(w, ONE).fill(bias, 0).fill(dy, ONE));
            var s = Norms.Shape.of(rows, n, 1e-5f);
            var bs = Norms.BackwardShape.of(rows, n, 1e-5f);
            String shape = "512x4096 fp32";
            time("Norms", "RMS forward", shape, 2 * bytes, e -> norms.encode(e, Norms.Kind.RMS, x, null, null, w, null, y, s));
            time("Norms", "RMS forward + residual", shape, 4 * bytes, e -> norms.encode(e, Norms.Kind.RMS, x, r, h, w, null, y, s));
            time("Norms", "LayerNorm forward", shape, 2 * bytes, e -> norms.encode(e, Norms.Kind.LAYER, x, null, null, w, bias, y, s));
            time("Norms", "RMS backward", shape, 3 * bytes, e -> norms.backward(e, Norms.Kind.RMS, x, w, dy, dv, dw, null, bs, false));
            time("Norms", "LayerNorm backward", shape, 3 * bytes, e -> norms.backward(e, Norms.Kind.LAYER, x, w, dy, dv, dw, db, bs, false));
        }
    }

    private void rope() {
        int tokens = 512, heads = 32, dim = 128;
        long bytes = 4L * tokens * heads * dim;
        var table = Rope.Table.standard(8192, dim, 500000);
        try (var rope = Rope.create(device, Precision.FP32, Precision.FP32); var x = buffer(bytes); var y = buffer(bytes);
             var pos = buffer(4L * tokens); var tab = buffer(table.bytes().length)) {
            init(e -> e.fill(x, ONE).upload(pos, ints(tokens, i -> i)).upload(tab, table.bytes()));
            var s = Rope.Shape.of(tokens, heads, dim, Rope.Layout.HALVES);
            time("Rope", "forward", "512x32x128 fp32", 2 * bytes, e -> rope.encode(e, x, pos, table, tab, y, s));
            time("Rope", "backward", "512x32x128 fp32", 2 * bytes, e -> rope.encodeBackward(e, x, pos, table, tab, y, s));
        }
    }

    private void gatherScatter() {
        int vocab = 8192, dim = 4096, count = 512;
        var xv = StridedView.matrix(vocab, dim);
        var yv = StridedView.matrix(count, dim);
        var iv = StridedView.contiguous(count);
        try (var gather = Gather.create(device); var x = buffer(4L * vocab * dim); var y = buffer(4L * count * dim); var idx = buffer(4L * count)) {
            init(e -> e.fill(x, ONE).fill(y, ONE).upload(idx, ints(count, i -> (i * 13) % vocab)));
            time("Gather", "gather rows", "512 of 8192x4096", 8L * count * dim, e -> gather.gather(e, x, xv, 0, idx, iv, y, yv));
            time("Gather", "scatterAdd accumulate", "512 into 8192x4096", 12L * count * dim,
                    e -> gather.scatterAdd(e, y, yv, 0, idx, iv, x, xv, true));
            time("Gather", "scatterAdd overwrite", "512 into 8192x4096", 4L * count * dim + 4L * vocab * dim,
                    e -> gather.scatterAdd(e, y, yv, 0, idx, iv, x, xv, false));
        }
    }

    private void optimizer() {
        long n = 1L << 24;
        try (var opt = Optimizer.create(device); var p = buffer(4 * n); var g = buffer(4 * n); var m = buffer(4 * n); var v = buffer(4 * n);
             var gh = buffer(2 * n); var shadow = buffer(2 * n); var sum = buffer(4)) {
            init(e -> e.fill(p, ONE).fill(g, ONE).fill(m, 0).fill(v, 0).fill(gh, HALF_ONES));
            var adam = Optimizer.AdamW.of(1e-3f, 0.01f);
            // AdamW: read p, g, m, v; write p, m, v
            time("Optimizer", "AdamW fp32 grads", "16M params", 28 * n, e -> opt.adamW(e, p, g, Precision.FP32, m, v, null, null, n, adam, 10));
            time("Optimizer", "AdamW fp16 grads + fp16 shadow", "16M params", 28 * n,
                    e -> opt.adamW(e, p, gh, Precision.FP16, m, v, null, shadow, n, adam, 10));
            var sgd = new Optimizer.Sgd(1e-2f, 0.9f, 0f, false);
            time("Optimizer", "SGD momentum", "16M params", 20 * n, e -> opt.sgd(e, p, g, Precision.FP32, m, null, null, n, sgd));
            try (var clip = GradientClip.create(device)) {
                var parts = List.of(new GradientClip.Part(g, StridedView.contiguous(n)));
                time("GradientClip", "squaredNorm", "16M fp32", 4 * n, e -> clip.squaredNorm(e, parts, sum));
                init(e -> clip.squaredNorm(e, parts, sum));
                // the norm of 16M ones is 4096: every repetition scales by 4095/4096
                time("GradientClip", "clip (scale in place)", "16M fp32", 8 * n, e -> clip.clip(e, sum, 4095f, parts));
                var many = new java.util.ArrayList<GradientClip.Part>();
                for (int i = 0; i < 64; i++) many.add(new GradientClip.Part(g, StridedView.contiguous((long) i * (n / 64), n / 64)));
                time("GradientClip", "squaredNorm, 64 tensors", "16M fp32", 4 * n, e -> clip.squaredNorm(e, many, sum));
            }
        }
    }

    private void logicCastChecks() {
        int n = 1 << 24;
        var v = StridedView.contiguous(n);
        String shape = "16M";
        try (var logic = Logic.create(device); var cast = Cast.create(device); var nan = NanCheck.create(device);
             var a = buffer(4L * n); var b = buffer(4L * n); var mask = buffer(4L * n); var y = buffer(4L * n); var h = buffer(2L * n);
             var errors = device.allocate(KernelErrors.descriptor())) {
            init(e -> e.fill(a, ONE).fill(b, 0).fill(h, HALF_ONES));
            time("Logic", "compare LESS fp32", shape, 12L * n, e -> logic.compare(e, Logic.Compare.LESS, Logic.Operand.FP32, a, v, b, v, mask, v));
            time("Logic", "select fp32, constant mask", shape, 16L * n, e -> logic.select(e, mask, v, a, v, b, v, y, v));
            // a mask that alternates every element: both inputs are read
            init(e -> e.upload(mask, ints(n, i -> i & 1)));
            time("Logic", "select fp32, alternating mask", shape, 16L * n, e -> logic.select(e, mask, v, a, v, b, v, y, v));
            time("Cast", "FP32 to FP16", shape, 6L * n, e -> cast.cast(e, Cast.Type.FP32, Cast.Type.FP16, a, v, h, v));
            time("Cast", "FP16 to FP32", shape, 6L * n, e -> cast.cast(e, Cast.Type.FP16, Cast.Type.FP32, h, v, y, v));
            time("Cast", "FP32 to BF16", shape, 6L * n, e -> cast.cast(e, Cast.Type.FP32, Cast.Type.BF16, a, v, h, v));
            time("NanCheck", "check", shape + " fp32", 4L * n, e -> nan.encode(e, a, v, errors));
        }
    }
}
