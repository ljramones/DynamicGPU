package org.dynamisengine.gpu.api.compute.spi;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

/**
 * Facts read from a SPIR-V module without a driver: the group-shared memory it declares.
 *
 * @since 1.7.0
 */
public final class SpirvModule {
    private static final int MAGIC = 0x07230203;
    private static final int OP_TYPE_BOOL = 20, OP_TYPE_INT = 21, OP_TYPE_FLOAT = 22, OP_TYPE_VECTOR = 23,
            OP_TYPE_MATRIX = 24, OP_TYPE_ARRAY = 28, OP_TYPE_STRUCT = 30, OP_TYPE_POINTER = 32, OP_CONSTANT = 43,
            OP_SPEC_CONSTANT = 50, OP_VARIABLE = 59;
    private static final int WORKGROUP = 4;

    private SpirvModule() { }

    /**
     * The bytes of the module's Workgroup-storage variables (Slang {@code groupshared}): scalars at their width (bools
     * as 4 bytes), vectors, matrices and arrays as their element sizes times their counts, structures as the sum of
     * their members. An array sized by a specialization constant counts at the constant's default. Padding a backend
     * adds (three-component vectors, structure alignment) is not counted.
     */
    public static long workgroupBytes(byte[] spirv) {
        if (spirv.length < 20 || spirv.length % 4 != 0) throw new IllegalArgumentException("not a SPIR-V module");
        ByteBuffer b = ByteBuffer.wrap(spirv).order(ByteOrder.LITTLE_ENDIAN);
        if (b.getInt(0) != MAGIC) {
            b.order(ByteOrder.BIG_ENDIAN);
            if (b.getInt(0) != MAGIC) throw new IllegalArgumentException("not a SPIR-V module");
        }
        Map<Integer, Long> sizes = new HashMap<>(), constants = new HashMap<>();
        Map<Integer, int[]> pending = new HashMap<>();   // arrays and structures, sized after the pass
        Map<Integer, Integer> pointee = new HashMap<>();
        long total = 0;
        java.util.List<Integer> workgroupPointers = new java.util.ArrayList<>();
        for (int at = 20; at < spirv.length;) {
            int word = b.getInt(at), count = word >>> 16, op = word & 0xFFFF;
            if (count == 0 || at + count * 4 > spirv.length) throw new IllegalArgumentException("truncated SPIR-V instruction at " + at);
            int[] w = new int[count];
            for (int i = 0; i < count; i++) w[i] = b.getInt(at + 4 * i);
            switch (op) {
                case OP_TYPE_BOOL -> sizes.put(w[1], 4L);
                case OP_TYPE_INT, OP_TYPE_FLOAT -> sizes.put(w[1], (long) w[2] / 8);
                case OP_TYPE_VECTOR, OP_TYPE_MATRIX, OP_TYPE_ARRAY, OP_TYPE_STRUCT -> pending.put(w[1], w);
                case OP_CONSTANT, OP_SPEC_CONSTANT -> constants.put(w[2], Integer.toUnsignedLong(w[3]));
                case OP_TYPE_POINTER -> { if (w[2] == WORKGROUP) pointee.put(w[1], w[3]); }
                case OP_VARIABLE -> { if (w[3] == WORKGROUP) workgroupPointers.add(w[1]); }
                default -> { }
            }
            at += count * 4;
        }
        for (int pointer : workgroupPointers) {
            Integer type = pointee.get(pointer);
            if (type == null) throw new IllegalArgumentException("Workgroup variable of unknown pointer type %" + pointer);
            total += size(type, sizes, pending, constants);
        }
        return total;
    }

    private static long size(int type, Map<Integer, Long> sizes, Map<Integer, int[]> pending, Map<Integer, Long> constants) {
        Long known = sizes.get(type);
        if (known != null) return known;
        int[] w = pending.get(type);
        if (w == null) throw new IllegalArgumentException("unsized SPIR-V type %" + type);
        long s = switch (w[0] & 0xFFFF) {
            case OP_TYPE_VECTOR, OP_TYPE_MATRIX -> size(w[2], sizes, pending, constants) * Integer.toUnsignedLong(w[3]);
            case OP_TYPE_ARRAY -> {
                Long n = constants.get(w[3]);
                if (n == null) throw new IllegalArgumentException("array length %" + w[3] + " is not a constant");
                yield size(w[2], sizes, pending, constants) * n;
            }
            default -> {   // structure
                long sum = 0;
                for (int i = 2; i < w.length; i++) sum += size(w[i], sizes, pending, constants);
                yield sum;
            }
        };
        sizes.put(type, s);
        return s;
    }
}
