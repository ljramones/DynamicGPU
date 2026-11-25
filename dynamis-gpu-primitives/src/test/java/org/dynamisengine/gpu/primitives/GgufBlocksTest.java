package org.dynamisengine.gpu.primitives;

import org.junit.jupiter.api.Test;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GGML blocks against gguf-py's own dequantization (llama.cpp's Python package): every value of the imported matrix
 * must be the same bits, and exporting must give back the original blocks. ggml-blocks.bin holds synthetic blocks of
 * every supported type (random codes and scales); -Dgguf.samples=file runs the same checks on slices of real models.
 */
class GgufBlocksTest {
    record Sample(GgufBlocks.Type type, int rows, int cols, byte[] raw, float[] values) { }

    static List<Sample> read(InputStream in) throws IOException {
        var data = new DataInputStream(in);
        var out = new ArrayList<Sample>();
        byte[] header = new byte[16];
        while (data.read(header) == 16) {
            var h = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
            var type = GgufBlocks.Type.of(h.getInt());
            int rows = h.getInt(), cols = h.getInt(), n = h.getInt();
            byte[] raw = data.readNBytes(n);
            var v = ByteBuffer.wrap(data.readNBytes(4 * rows * cols)).order(ByteOrder.LITTLE_ENDIAN);
            float[] values = new float[rows * cols];
            for (int i = 0; i < values.length; i++) values[i] = v.getFloat();
            out.add(new Sample(type, rows, cols, raw, values));
        }
        return out;
    }

    static void check(Sample s, String source) {
        var packed = GgufBlocks.fromGgml(s.type(), s.raw(), 0, s.rows(), s.cols());
        assertEquals(s.raw().length, GgufBlocks.bytes(s.type(), s.rows(), s.cols()));
        float[] got = packed.matrix().dequantize(packed.bytes());
        for (int i = 0; i < got.length; i++)
            assertEquals(Float.floatToRawIntBits(s.values()[i]), Float.floatToRawIntBits(got[i]),
                    source + " " + s.type() + " element " + i + ": " + got[i] + ", gguf-py " + s.values()[i]);
        assertArrayEquals(s.raw(), GgufBlocks.toGgml(packed.matrix(), packed.bytes()), source + " " + s.type() + ": exported blocks");
    }

    @Test void syntheticBlocksMatchGgufPy() throws IOException {
        List<Sample> samples;
        try (var in = GgufBlocksTest.class.getResourceAsStream("ggml-blocks.bin")) { samples = read(in); }
        assertEquals(GgufBlocks.Type.values().length, samples.size());
        for (var s : samples) check(s, "synthetic");
    }

    @Test void realModelBlocksMatchGgufPy() throws IOException {
        String path = System.getProperty("gguf.samples");
        assumeTrue(path != null, "set -Dgguf.samples=<file written by the oracle script>");
        try (var in = Files.newInputStream(Path.of(path))) {
            var samples = read(in);
            for (var s : samples) check(s, "real");
            System.out.println("gguf real samples: " + samples.stream().map(s -> s.type() + " " + s.rows() + "x" + s.cols()).toList());
        }
    }

    @Test void typesAndSizes() {
        assertEquals(GgufBlocks.Type.Q4_K, GgufBlocks.Type.of(12));
        assertThrows(IllegalArgumentException.class, () -> GgufBlocks.Type.of(7), "Q5_1 is not supported");
        assertEquals(144L * 2 * 4, GgufBlocks.bytes(GgufBlocks.Type.Q4_K, 4, 512));
        assertEquals(GgufBlocks.Type.Q4_0, GgufBlocks.Type.of(new QuantizedMatrix(QuantizedMatrix.Format.Q4, 32, 1, 64)));
        assertThrows(IllegalArgumentException.class, () -> GgufBlocks.Type.of(new QuantizedMatrix(QuantizedMatrix.Format.Q4, 64, 1, 64)));
        assertThrows(IllegalArgumentException.class, () -> GgufBlocks.fromGgml(GgufBlocks.Type.Q8_0, new byte[33], 0, 1, 32));
    }
}
