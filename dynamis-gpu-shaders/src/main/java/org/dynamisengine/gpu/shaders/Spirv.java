package org.dynamisengine.gpu.shaders;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * SPIR-V post-processing: decorates every OpFAdd, OpFSub and GLSL.std.450 Fma result NoContraction, so no driver
 * fuses a product into an add or rewrites an fma (Mesa otherwise merges fma(k, a, fma(k, b, c)) into
 * fma(k, a + b, c)). Multiplies are left unmarked: SPIRV-Cross (MoltenVK) turns a NoContraction multiply into
 * fma(a, b, 0), which loses the sign of -0.
 */
public final class Spirv {
    static final int MAGIC = 0x07230203;
    private static final int OP_DECORATE = 71;
    private static final int NO_CONTRACTION = 42;
    private static final int OP_EXT_INST_IMPORT = 11;
    private static final int OP_EXT_INST = 12;
    private static final int OP_FADD = 129;
    private static final int OP_FSUB = 131;
    private static final int GLSL_FMA = 50;
    /** Capabilities, extensions, imports, memory model, entry points, execution modes, debug and annotations. */
    private static final Set<Integer> LEADING = Set.of(17, 10, 11, 14, 15, 16, 330, 331, 2, 3, 4, 5, 6, 7, 8, 12, 71, 72,
            73, 74, 75, 332, 5632, 5633);

    private Spirv() { }

    /** The module with NoContraction on every float add, subtract and fma that lacks it. */
    public static byte[] noContraction(byte[] spv) {
        IntBufferView w = new IntBufferView(spv);
        if (w.words.length < 5 || w.words[0] != MAGIC) throw new IllegalArgumentException("not little-endian SPIR-V");
        List<int[]> insts = new ArrayList<>();
        for (int i = 5; i < w.words.length; ) {
            int count = w.words[i] >>> 16;
            if (count == 0 || i + count > w.words.length) throw new IllegalArgumentException("malformed SPIR-V at word " + i);
            int[] inst = new int[count];
            System.arraycopy(w.words, i, inst, 0, count);
            insts.add(inst);
            i += count;
        }
        Set<Integer> marked = new HashSet<>();
        Set<Integer> glsl = new HashSet<>();
        for (int[] inst : insts) {
            int op = inst[0] & 0xFFFF;
            if (op == OP_DECORATE && inst[2] == NO_CONTRACTION) marked.add(inst[1]);
            if (op == OP_EXT_INST_IMPORT && "GLSL.std.450".equals(string(inst, 2))) glsl.add(inst[1]);
        }
        List<Integer> targets = new ArrayList<>();
        for (int[] inst : insts) {
            int op = inst[0] & 0xFFFF;
            if (op == OP_FADD || op == OP_FSUB) targets.add(inst[2]);
        }
        for (int[] inst : insts) {
            if ((inst[0] & 0xFFFF) == OP_EXT_INST && glsl.contains(inst[3]) && inst[4] == GLSL_FMA) targets.add(inst[2]);
        }
        targets.removeIf(marked::contains);
        int at = 0;
        while (at < insts.size() && LEADING.contains(insts.get(at)[0] & 0xFFFF)) at++;
        List<int[]> out = new ArrayList<>(insts.subList(0, at));
        for (int t : targets) out.add(new int[] {(3 << 16) | OP_DECORATE, t, NO_CONTRACTION});
        out.addAll(insts.subList(at, insts.size()));
        int total = 5;
        for (int[] inst : out) total += inst.length;
        ByteBuffer b = ByteBuffer.allocate(total * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int k = 0; k < 5; k++) b.putInt(w.words[k]);
        for (int[] inst : out) for (int v : inst) b.putInt(v);
        return b.array();
    }

    /** The nul-terminated string stored from word {@code from} of an instruction. */
    private static String string(int[] inst, int from) {
        ByteBuffer b = ByteBuffer.allocate((inst.length - from) * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int k = from; k < inst.length; k++) b.putInt(inst[k]);
        byte[] bytes = b.array();
        int n = 0;
        while (n < bytes.length && bytes[n] != 0) n++;
        return new String(bytes, 0, n, StandardCharsets.UTF_8);
    }

    private static final class IntBufferView {
        final int[] words;

        IntBufferView(byte[] bytes) {
            if (bytes.length % 4 != 0) throw new IllegalArgumentException("SPIR-V length is not a multiple of 4");
            words = new int[bytes.length / 4];
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(words);
        }
    }
}
