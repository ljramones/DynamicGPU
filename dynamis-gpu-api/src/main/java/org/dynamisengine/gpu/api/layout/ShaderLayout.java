/*
 * The MIT License
 *
 * Copyright (c) 2024 Vectrix
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package org.dynamisengine.gpu.api.layout;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Shader-interface layout calculator for the GLSL/SPIR-V {@code std140}, {@code std430} and {@code scalar}
 * ({@code VK_EXT_scalar_block_layout}) rules, covering:
 * <ul>
 * <li>scalars ({@code float}, {@code int}, {@code uint}, {@code bool}, {@code double}) and vectors;</li>
 * <li>column- and row-major matrices;</li>
 * <li>arrays, including multidimensional ones;</li>
 * <li>nested structs.</li>
 * </ul>
 * <p>
 * A layout gives every member's byte offset, array stride and matrix stride, and the block size. The test suite
 * checks them against glslang's own layout (through SPIR-V reflection) on randomly generated blocks. A layout can
 * also emit its GLSL declaration ({@link Block#toGlsl}), so the Java writer and the shader share one definition.
 * <p>
 * This is a build-time calculator: it allocates, and is meant for setting up writers, not for per-frame use.
 *
 * @since 2.4.0
 */
public final class ShaderLayout {
    private ShaderLayout() {
    }

    /** Layout rules. */
    public enum Rules {
        /** GLSL std140: array and struct alignment rounded up to 16 bytes (uniform blocks). */
        STD140,
        /** GLSL std430: no rounding to 16 bytes (storage blocks). */
        STD430,
        /** {@code VK_EXT_scalar_block_layout}: everything aligned to its scalar component, no padding. */
        SCALAR
    }

    /** Scalar component types; {@code bool} occupies 4 bytes in interface blocks. */
    public enum Scalar {
        /** 32-bit {@code float}. */
        FLOAT(4, "float", ""),
        /** 32-bit signed {@code int}. */
        INT(4, "int", "i"),
        /** 32-bit {@code uint}. */
        UINT(4, "uint", "u"),
        /** {@code bool}, stored as 4 bytes. */
        BOOL(4, "bool", "b"),
        /** 64-bit {@code double}. */
        DOUBLE(8, "double", "d");

        final int bytes;
        final String glsl;
        final String vectorPrefix;

        Scalar(int bytes, String glsl, String vectorPrefix) {
            this.bytes = bytes;
            this.glsl = glsl;
            this.vectorPrefix = vectorPrefix;
        }

        /**
         * The size of one component.
         *
         * @return the size in bytes
         */
        public int bytes() {
            return bytes;
        }
    }

    /** A member type: {@link Basic}, {@link ArrayOf} or {@link Struct}. */
    public sealed interface Type permits Basic, ArrayOf, Struct {
    }

    /**
     * A scalar ({@code columns == rows == 1}), a vector ({@code columns == 1}, {@code rows} components) or a
     * matrix ({@code columns} × {@code rows}, float or double), optionally row-major.
     *
     * @param scalar the component type
     * @param columns 1 for scalars and vectors, else the matrix column count
     * @param rows the vector component count, or the matrix row count
     * @param rowMajor whether a matrix is stored row by row
     */
    public record Basic(Scalar scalar, int columns, int rows, boolean rowMajor) implements Type {
        /**
         * Validates the components.
         *
         * @throws IllegalArgumentException if {@code scalar} is null, {@code columns} or {@code rows} is outside
         *         {@code [1, 4]}, a matrix is not float or double or has fewer than 2 rows, or a non-matrix is
         *         row-major
         */
        public Basic {
            if (scalar == null) {
                throw new IllegalArgumentException("scalar must not be null");
            }
            if (columns < 1 || columns > 4 || rows < 1 || rows > 4) {
                throw new IllegalArgumentException("columns and rows must be 1..4");
            }
            if (columns > 1 && (rows < 2 || (scalar != Scalar.FLOAT && scalar != Scalar.DOUBLE))) {
                throw new IllegalArgumentException("matrices are float or double with at least 2 rows");
            }
            if (rowMajor && columns == 1) {
                throw new IllegalArgumentException("only matrices can be row-major");
            }
        }

        /**
         * Whether this is a matrix.
         *
         * @return {@code true} if {@code columns > 1}
         */
        public boolean isMatrix() {
            return columns > 1;
        }
    }

    /**
     * An array of {@code length} elements.
     *
     * @param element the element type
     * @param length the element count
     */
    public record ArrayOf(Type element, int length) implements Type {
        /**
         * Validates the components.
         *
         * @throws IllegalArgumentException if {@code element} is null or {@code length} is less than 1
         */
        public ArrayOf {
            if (element == null || length < 1) {
                throw new IllegalArgumentException("arrays need an element type and a length of at least 1");
            }
        }
    }

    /**
     * A named member.
     *
     * @param name the member name
     * @param type the member type
     */
    public record Member(String name, Type type) {
        /**
         * Validates the components.
         *
         * @throws IllegalArgumentException if {@code name} is null or empty, or {@code type} is null
         */
        public Member {
            if (name == null || name.isEmpty() || type == null) {
                throw new IllegalArgumentException("members need a name and a type");
            }
        }
    }

    /**
     * A struct (or block) type with at least one member, all names distinct.
     *
     * @param name the GLSL type name
     * @param members the members, in declaration order
     */
    public record Struct(String name, List<Member> members) implements Type {
        /**
         * Copies {@code members} into an unmodifiable list and validates it.
         *
         * @throws IllegalArgumentException if {@code name} is null or empty, {@code members} or one of its elements is
         *         null, there are no members or two members share a name
         */
        public Struct {
            if (name == null || name.isEmpty()) {
                throw new IllegalArgumentException("structs need a name");
            }
            if (members == null) {
                throw new IllegalArgumentException("struct " + name + ": members must not be null");
            }
            for (Member m : members) {
                if (m == null) {
                    throw new IllegalArgumentException("struct " + name + ": members must not be null");
                }
            }
            members = List.copyOf(members);
            if (members.isEmpty()) {
                throw new IllegalArgumentException("structs need at least one member");
            }
            Set<String> names = new HashSet<>();
            for (Member m : members) {
                if (!names.add(m.name())) {
                    throw new IllegalArgumentException("duplicate member " + m.name());
                }
            }
        }
    }

    /**
     * One member of a layout.
     *
     * @param path the member path, for example {@code lights[0].color}
     * @param offset the byte offset from the start of the block
     * @param size the member size in bytes
     * @param arrayStride the array stride, or 0 if the member is not an array
     * @param matrixStride the matrix stride, or 0 if the member is not a matrix (or array of matrices)
     */
    public record Entry(String path, int offset, int size, int arrayStride, int matrixStride) {
    }

    /** The GLSL {@code float} type. */
    public static final Basic FLOAT = new Basic(Scalar.FLOAT, 1, 1, false);
    /** The GLSL {@code int} type. */
    public static final Basic INT = new Basic(Scalar.INT, 1, 1, false);
    /** The GLSL {@code uint} type. */
    public static final Basic UINT = new Basic(Scalar.UINT, 1, 1, false);
    /** The GLSL {@code bool} type (4 bytes). */
    public static final Basic BOOL = new Basic(Scalar.BOOL, 1, 1, false);
    /** The GLSL {@code double} type. */
    public static final Basic DOUBLE = new Basic(Scalar.DOUBLE, 1, 1, false);

    /**
     * The {@code vecN} type; {@code n == 1} gives the scalar type.
     *
     * @param n the component count, 1 to 4
     * @return the type
     * @throws IllegalArgumentException if {@code n} is outside {@code [1, 4]}
     */
    public static Basic vec(int n) {
        return new Basic(Scalar.FLOAT, 1, n, false);
    }

    /**
     * The {@code ivecN} type; {@code n == 1} gives the scalar type.
     *
     * @param n the component count, 1 to 4
     * @return the type
     * @throws IllegalArgumentException if {@code n} is outside {@code [1, 4]}
     */
    public static Basic ivec(int n) {
        return new Basic(Scalar.INT, 1, n, false);
    }

    /**
     * The {@code uvecN} type; {@code n == 1} gives the scalar type.
     *
     * @param n the component count, 1 to 4
     * @return the type
     * @throws IllegalArgumentException if {@code n} is outside {@code [1, 4]}
     */
    public static Basic uvec(int n) {
        return new Basic(Scalar.UINT, 1, n, false);
    }

    /**
     * The {@code bvecN} type; {@code n == 1} gives the scalar type.
     *
     * @param n the component count, 1 to 4
     * @return the type
     * @throws IllegalArgumentException if {@code n} is outside {@code [1, 4]}
     */
    public static Basic bvec(int n) {
        return new Basic(Scalar.BOOL, 1, n, false);
    }

    /**
     * The {@code dvecN} type; {@code n == 1} gives the scalar type.
     *
     * @param n the component count, 1 to 4
     * @return the type
     * @throws IllegalArgumentException if {@code n} is outside {@code [1, 4]}
     */
    public static Basic dvec(int n) {
        return new Basic(Scalar.DOUBLE, 1, n, false);
    }

    /**
     * The {@code matCxR} type, column-major. {@code columns == 1} gives a vector of {@code rows} components.
     *
     * @param columns the column count, 1 to 4
     * @param rows the row count, 2 to 4 for a matrix
     * @return the type
     * @throws IllegalArgumentException if the dimensions are out of range
     */
    public static Basic mat(int columns, int rows) {
        return new Basic(Scalar.FLOAT, columns, rows, false);
    }

    /**
     * The {@code dmatCxR} type, column-major. {@code columns == 1} gives a vector of {@code rows} components.
     *
     * @param columns the column count, 1 to 4
     * @param rows the row count, 2 to 4 for a matrix
     * @return the type
     * @throws IllegalArgumentException if the dimensions are out of range
     */
    public static Basic dmat(int columns, int rows) {
        return new Basic(Scalar.DOUBLE, columns, rows, false);
    }

    /**
     * The row-major form of a matrix type.
     *
     * @param matrix a matrix type
     * @return the same matrix type, stored row by row
     * @throws IllegalArgumentException if {@code matrix} is not a matrix
     */
    public static Basic rowMajor(Basic matrix) {
        return new Basic(matrix.scalar(), matrix.columns(), matrix.rows(), true);
    }

    /**
     * An array type.
     *
     * @param element the element type; nest arrays for multidimensional ones
     * @param length the element count, at least 1
     * @return the type
     * @throws IllegalArgumentException if {@code element} is null or {@code length} is less than 1
     */
    public static ArrayOf array(Type element, int length) {
        return new ArrayOf(element, length);
    }

    /**
     * A named member.
     *
     * @param name the member name
     * @param type the member type
     * @return the member
     * @throws IllegalArgumentException if {@code name} is null or empty, or {@code type} is null
     */
    public static Member member(String name, Type type) {
        return new Member(name, type);
    }

    /**
     * A struct type.
     *
     * @param name the GLSL type name
     * @param members the members, in declaration order
     * @return the type
     * @throws IllegalArgumentException if {@code name} is null or empty, a member is null, there are no members or two
     *         members share a name
     */
    public static Struct struct(String name, Member... members) {
        if (members == null) {
            throw new IllegalArgumentException("struct " + name + ": members must not be null");
        }
        return new Struct(name, java.util.Arrays.asList(members));
    }

    private static int roundUp(int x, int a) {
        return (x + a - 1) / a * a;
    }

    /**
     * The base alignment of {@code t} under {@code rules}.
     *
     * @param rules the layout rules
     * @param t the type
     * @return the alignment in bytes
     */
    public static int alignment(Rules rules, Type t) {
        switch (t) {
            case Basic b -> {
                if (b.isMatrix()) {
                    return arrayAlignment(rules, b.rowMajor() ? new Basic(b.scalar(), 1, b.columns(), false)
                            : new Basic(b.scalar(), 1, b.rows(), false));
                }
                int n = b.scalar().bytes;
                if (rules == Rules.SCALAR || b.rows() == 1) {
                    return n;
                }
                return b.rows() == 2 ? 2 * n : 4 * n;
            }
            case ArrayOf a -> {
                return arrayAlignment(rules, a.element());
            }
            case Struct s -> {
                int align = 1;
                for (Member m : s.members()) {
                    align = Math.max(align, alignment(rules, m.type()));
                }
                return rules == Rules.STD140 ? roundUp(align, 16) : align;
            }
        }
    }

    private static int arrayAlignment(Rules rules, Type element) {
        int a = alignment(rules, element);
        return rules == Rules.STD140 ? roundUp(a, 16) : a;
    }

    /**
     * The stride between consecutive elements of an array of {@code element}.
     *
     * @param rules the layout rules
     * @param element the element type
     * @return the stride in bytes
     */
    public static int arrayStride(Rules rules, Type element) {
        return roundUp(size(rules, element), arrayAlignment(rules, element));
    }

    /**
     * The stride between consecutive columns (rows, if row-major) of a matrix.
     *
     * @param rules the layout rules
     * @param matrix the matrix type
     * @return the stride in bytes
     * @throws IllegalArgumentException if {@code matrix} is not a matrix
     */
    public static int matrixStride(Rules rules, Basic matrix) {
        if (!matrix.isMatrix()) {
            throw new IllegalArgumentException("not a matrix");
        }
        int components = matrix.rowMajor() ? matrix.columns() : matrix.rows();
        return arrayStride(rules, new Basic(matrix.scalar(), 1, components, false));
    }

    /**
     * The size of {@code t} in bytes, including a struct's trailing padding.
     *
     * @param rules the layout rules
     * @param t the type
     * @return the size in bytes
     */
    public static int size(Rules rules, Type t) {
        return switch (t) {
            case Basic b -> b.isMatrix() ? (b.rowMajor() ? b.rows() : b.columns()) * matrixStride(rules, b)
                    : b.rows() * b.scalar().bytes;
            case ArrayOf a -> a.length() * arrayStride(rules, a.element());
            case Struct s -> roundUp(end(rules, s), alignment(rules, s));
        };
    }

    /** The end of the last member of {@code s}, before trailing padding. */
    private static int end(Rules rules, Struct s) {
        int cursor = 0;
        for (Member m : s.members()) {
            cursor = roundUp(cursor, alignment(rules, m.type())) + footprint(rules, m.type());
        }
        return cursor;
    }

    /**
     * How far a member pushes the next member's offset. Under std140 and std430 that is its size, since the next
     * offset is rounded up to the member's alignment anyway. Under {@code SCALAR} the next member may start in the
     * trailing padding of a struct, or of an array's last element, so only the bytes actually used count, as glslang
     * assigns them.
     */
    private static int footprint(Rules rules, Type t) {
        if (rules != Rules.SCALAR) {
            return size(rules, t);
        }
        if (t instanceof ArrayOf a) {
            return (a.length() - 1) * arrayStride(rules, a.element()) + footprint(rules, a.element());
        }
        if (t instanceof Struct inner) {
            return end(rules, inner);
        }
        return size(rules, t);
    }

    /** The end of the last member of {@code s} using declared sizes (no trailing struct padding, recursively). */
    private static int declaredEnd(Rules rules, Struct s) {
        List<Member> ms = s.members();
        Member last = ms.get(ms.size() - 1);
        int off = memberOffset(rules, s, last.name());
        return off + (last.type() instanceof Struct inner ? declaredEnd(rules, inner) : size(rules, last.type()));
    }

    private static int memberOffset(Rules rules, Struct s, String name) {
        int cursor = 0;
        for (Member m : s.members()) {
            cursor = roundUp(cursor, alignment(rules, m.type()));
            if (m.name().equals(name)) {
                return cursor;
            }
            cursor += footprint(rules, m.type());
        }
        throw new IllegalArgumentException("no member " + name + " in " + s.name());
    }

    private static Type memberType(Struct s, String name) {
        for (Member m : s.members()) {
            if (m.name().equals(name)) {
                return m.type();
            }
        }
        throw new IllegalArgumentException("no member " + name + " in " + s.name());
    }

    /**
     * Lays out {@code block} as an interface block under {@code rules}.
     *
     * @param rules the layout rules
     * @param block the block's members, as a struct
     * @return the layout
     * @throws IllegalArgumentException if {@code rules} or {@code block} is null
     */
    public static Block layout(Rules rules, Struct block) {
        if (rules == null || block == null) {
            throw new IllegalArgumentException("rules and block must not be null");
        }
        return new Block(rules, block);
    }

    /** An interface block laid out under one set of rules. */
    public static final class Block {
        private final Rules rules;
        private final Struct type;
        private final List<Entry> entries;

        Block(Rules rules, Struct type) {
            this.rules = rules;
            this.type = type;
            List<Entry> list = new ArrayList<>();
            collect(type, "", 0, list);
            this.entries = List.copyOf(list);
        }

        private void collect(Struct s, String prefix, int base, List<Entry> out) {
            for (Member m : s.members()) {
                int off = base + memberOffset(rules, s, m.name());
                String path = prefix + m.name();
                Type t = m.type();
                int arrayStride = 0;
                String elementPath = path;
                while (t instanceof ArrayOf a) {
                    if (arrayStride == 0) {
                        arrayStride = ShaderLayout.arrayStride(rules, a.element());
                    }
                    t = a.element();
                    elementPath += "[0]";
                }
                int matrixStride = t instanceof Basic b && b.isMatrix() ? ShaderLayout.matrixStride(rules, b) : 0;
                out.add(new Entry(path, off, ShaderLayout.size(rules, m.type()), arrayStride, matrixStride));
                if (t instanceof Struct inner) {
                    collect(inner, elementPath + ".", off, out);
                }
            }
        }

        /**
         * The rules this block was laid out with.
         *
         * @return the rules
         */
        public Rules rules() {
            return rules;
        }

        /**
         * The block's type.
         *
         * @return the struct this block was laid out from
         */
        public Struct type() {
            return type;
        }

        /**
         * The block size as SPIR-V reflection reports it: the offset of the last member plus that member's declared
         * size. A trailing array counts its full stride; a trailing struct counts up to its own last member, without
         * its trailing padding.
         *
         * @return the size in bytes
         */
        public int size() {
            return declaredEnd(rules, type);
        }

        /**
         * The block's base alignment.
         *
         * @return the alignment in bytes
         */
        public int alignment() {
            return ShaderLayout.alignment(rules, type);
        }

        /**
         * The size rounded up to the alignment: the stride of an array of these blocks, or of the struct.
         *
         * @return the padded size in bytes
         */
        public int paddedSize() {
            return ShaderLayout.size(rules, type);
        }

        /**
         * Every member, depth first; elements of arrays of structs appear as {@code name[0].member}.
         *
         * @return an unmodifiable list of entries
         */
        public List<Entry> entries() {
            return entries;
        }

        /**
         * The byte offset of a member path such as {@code "lights[3].color"} or {@code "weights[200]"}.
         *
         * @param path the member path: names separated by {@code .}, each optionally followed by {@code [index]}
         *        subscripts
         * @return the offset in bytes from the start of the block
         * @throws IllegalArgumentException
         *             for an unknown member, an index out of range, or an index on a non-array
         */
        public int offset(String path) {
            return resolve(path, new Type[1]);
        }

        /**
         * The array stride of the array at {@code path}.
         *
         * @param path a member path (see {@link #offset}) that names an array
         * @return the stride in bytes
         * @throws IllegalArgumentException if the path is invalid or does not name an array
         */
        public int arrayStride(String path) {
            Type t = typeAt(path);
            if (!(t instanceof ArrayOf a)) {
                throw new IllegalArgumentException(path + " is not an array");
            }
            return ShaderLayout.arrayStride(rules, a.element());
        }

        /**
         * The matrix stride of the matrix at {@code path} (index into arrays of matrices first).
         *
         * @param path a member path (see {@link #offset}) that names a matrix
         * @return the stride in bytes
         * @throws IllegalArgumentException if the path is invalid or does not name a matrix
         */
        public int matrixStride(String path) {
            Type t = typeAt(path);
            if (!(t instanceof Basic b) || !b.isMatrix()) {
                throw new IllegalArgumentException(path + " is not a matrix");
            }
            return ShaderLayout.matrixStride(rules, b);
        }

        private Type typeAt(String path) {
            Type[] holder = new Type[1];
            resolve(path, holder);
            return holder[0];
        }

        private int resolve(String path, Type[] typeOut) {
            Type t = type;
            int off = 0;
            for (String seg : path.split("\\.", -1)) {
                int bracket = seg.indexOf('[');
                String name = bracket < 0 ? seg : seg.substring(0, bracket);
                if (!(t instanceof Struct s)) {
                    throw new IllegalArgumentException("cannot select " + name + " in " + path);
                }
                off += memberOffset(rules, s, name);
                t = memberType(s, name);
                while (bracket >= 0) {
                    int close = seg.indexOf(']', bracket);
                    if (close < 0) {
                        throw new IllegalArgumentException("unbalanced [ in " + path);
                    }
                    int index = Integer.parseInt(seg.substring(bracket + 1, close));
                    if (!(t instanceof ArrayOf a) || index < 0 || index >= a.length()) {
                        throw new IllegalArgumentException("bad index " + index + " in " + path);
                    }
                    off += index * ShaderLayout.arrayStride(rules, a.element());
                    t = a.element();
                    bracket = seg.indexOf('[', close);
                }
            }
            typeOut[0] = t;
            return off;
        }

        /**
         * The GLSL declaration of this block: the struct definitions it uses, then
         * {@code layout(<rules>, binding = <binding>) <storage> <blockName> { ... } <instanceName>;}. With
         * {@code SCALAR}, or {@code STD430} for a uniform block, the declaration is preceded by
         * {@code #extension GL_EXT_scalar_block_layout : require}. GLSL allows {@code layout(row_major)} on block
         * members only, so a struct holding a row-major matrix is rejected. Each struct type is declared once, before
         * its first use; GLSL has one name per struct type, so two different structs with the same name are rejected
         * (the same struct used twice is fine).
         *
         * @param storage {@code "uniform"} or {@code "buffer"}
         * @param blockName the block name, not empty
         * @param instanceName the instance name; empty for an anonymous block, whose members are referred to without a
         *        prefix
         * @param binding the binding number, at least 0
         * @return the GLSL source, one declaration per line
         * @throws IllegalArgumentException if {@code storage} is neither {@code "uniform"} nor {@code "buffer"},
         *         {@code blockName} is null or empty, {@code instanceName} is null, {@code binding} is negative, a
         *         struct contains a row-major matrix, or two different structs share a name
         */
        public String toGlsl(String storage, String blockName, String instanceName, int binding) {
            if (!"uniform".equals(storage) && !"buffer".equals(storage)) {
                throw new IllegalArgumentException("storage must be \"uniform\" or \"buffer\": " + storage);
            }
            if (blockName == null || blockName.isEmpty() || instanceName == null) {
                throw new IllegalArgumentException("blockName must not be empty, instanceName must not be null");
            }
            if (binding < 0) {
                throw new IllegalArgumentException("binding must not be negative: " + binding);
            }
            StringBuilder sb = new StringBuilder();
            if (rules == Rules.SCALAR || (rules == Rules.STD430 && "uniform".equals(storage))) {
                sb.append("#extension GL_EXT_scalar_block_layout : require\n");
            }
            Map<String, Struct> declared = new HashMap<>();
            for (Member m : type.members()) {
                declareStructs(m.type(), declared, sb);
            }
            sb.append("layout(").append(rules.name().toLowerCase(java.util.Locale.ROOT)).append(", binding = ")
                    .append(binding).append(") ").append(storage).append(' ').append(blockName).append(" {\n");
            for (Member m : type.members()) {
                sb.append("    ").append(declaration(m)).append(";\n");
            }
            sb.append(instanceName.isEmpty() ? "};\n" : "} " + instanceName + ";\n");
            return sb.toString();
        }

        private static void declareStructs(Type t, Map<String, Struct> declared, StringBuilder sb) {
            while (t instanceof ArrayOf a) {
                t = a.element();
            }
            if (t instanceof Struct s) {
                Struct prior = declared.get(s.name());
                if (prior != null) {
                    if (!prior.equals(s)) {
                        throw new IllegalArgumentException("two different structs are named " + s.name());
                    }
                    return;
                }
                for (Member m : s.members()) {
                    declareStructs(m.type(), declared, sb);
                }
                if (declared.containsKey(s.name())) { // a nested struct of the same name, necessarily different
                    throw new IllegalArgumentException("two different structs are named " + s.name());
                }
                declared.put(s.name(), s);
                sb.append("struct ").append(s.name()).append(" {\n");
                for (Member m : s.members()) {
                    Type mt = m.type();
                    while (mt instanceof ArrayOf a) {
                        mt = a.element();
                    }
                    if (mt instanceof Basic b && b.rowMajor()) {
                        throw new IllegalArgumentException("GLSL cannot declare row-major matrices inside a struct ("
                                + s.name() + "." + m.name() + "); qualify a block member instead");
                    }
                    sb.append("    ").append(declaration(m)).append(";\n");
                }
                sb.append("};\n");
            }
        }

        private static String declaration(Member m) {
            Type t = m.type();
            StringBuilder dims = new StringBuilder();
            while (t instanceof ArrayOf a) {
                dims.append('[').append(a.length()).append(']');
                t = a.element();
            }
            String typeName;
            String qualifier = "";
            if (t instanceof Struct s) {
                typeName = s.name();
            } else {
                Basic b = (Basic) t;
                if (b.isMatrix()) {
                    typeName = (b.scalar() == Scalar.DOUBLE ? "dmat" : "mat") + b.columns() + "x" + b.rows();
                    if (b.rowMajor()) {
                        qualifier = "layout(row_major) ";
                    }
                } else if (b.rows() == 1) {
                    typeName = b.scalar().glsl;
                } else {
                    typeName = b.scalar().vectorPrefix + "vec" + b.rows();
                }
            }
            return qualifier + typeName + " " + m.name() + dims;
        }
    }
}
