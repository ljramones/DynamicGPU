/*
 * The MIT License
 *
 * Copyright (c) 2026 Vectrix
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.dynamisengine.gpu.api.layout.ShaderLayout.Block;
import org.dynamisengine.gpu.api.layout.ShaderLayout.Entry;
import org.dynamisengine.gpu.api.layout.ShaderLayout.Rules;
import org.dynamisengine.gpu.api.layout.ShaderLayout.Struct;
import org.dynamisengine.gpu.api.layout.ShaderLayout.Type;
import org.junit.jupiter.api.Test;

/**
 * ShaderLayout against glslang. The golden ({@code gpu/shader-layout-golden.txt}) holds the layouts glslang assigned
 * to random blocks, read back by SPIR-V reflection; vectrix's {@code scripts/gen-shader-layout-golden.py} recorded it.
 */
class ShaderLayoutTest {

    /** Parses the generator's compact type syntax. */
    private static final class Parser {
        private final String s;
        private int i;

        Parser(String s) {
            this.s = s;
        }

        Type type() {
            Type t;
            if (s.charAt(i) == '{') {
                t = struct();
            } else {
                boolean rowMajor = s.startsWith("R:", i);
                if (rowMajor) {
                    i += 2;
                }
                int start = i;
                while (i < s.length() && Character.isLetterOrDigit(s.charAt(i))) {
                    i++;
                }
                ShaderLayout.Basic b = basic(s.substring(start, i));
                t = rowMajor ? ShaderLayout.rowMajor(b) : b;
            }
            List<Integer> dims = new ArrayList<>();
            while (i < s.length() && s.charAt(i) == '[') {
                int close = s.indexOf(']', i);
                dims.add(Integer.parseInt(s.substring(i + 1, close)));
                i = close + 1;
            }
            for (int d = dims.size() - 1; d >= 0; d--) { // outermost first in the text
                t = ShaderLayout.array(t, dims.get(d));
            }
            return t;
        }

        Struct struct() {
            i++; // {
            int semi = s.indexOf(';', i);
            String name = s.substring(i, semi);
            i = semi + 1;
            List<ShaderLayout.Member> members = new ArrayList<>();
            while (true) {
                int colon = s.indexOf(':', i);
                String member = s.substring(i, colon);
                i = colon + 1;
                members.add(ShaderLayout.member(member, type()));
                char c = s.charAt(i++);
                if (c == '}') {
                    return new Struct(name, members);
                }
            }
        }

        static ShaderLayout.Basic basic(String t) {
            switch (t) {
                case "float":
                    return ShaderLayout.FLOAT;
                case "int":
                    return ShaderLayout.INT;
                case "uint":
                    return ShaderLayout.UINT;
                case "bool":
                    return ShaderLayout.BOOL;
                case "double":
                    return ShaderLayout.DOUBLE;
                default:
                    break;
            }
            int v = t.indexOf("vec");
            if (v >= 0) {
                int n = t.charAt(t.length() - 1) - '0';
                switch (t.substring(0, v)) {
                    case "":
                        return ShaderLayout.vec(n);
                    case "i":
                        return ShaderLayout.ivec(n);
                    case "u":
                        return ShaderLayout.uvec(n);
                    case "b":
                        return ShaderLayout.bvec(n);
                    default:
                        return ShaderLayout.dvec(n);
                }
            }
            boolean dbl = t.startsWith("d");
            String dims = t.substring(dbl ? 4 : 3);
            int c = dims.charAt(0) - '0', r = dims.charAt(2) - '0';
            return dbl ? ShaderLayout.dmat(c, r) : ShaderLayout.mat(c, r);
        }
    }

    private static List<String> golden() throws Exception {
        List<String> lines = new ArrayList<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(
                ShaderLayoutTest.class.getResourceAsStream("/gpu/shader-layout-golden.txt"), StandardCharsets.UTF_8))) {
            for (String line; (line = in.readLine()) != null;) {
                if (!line.startsWith("#") && !line.isBlank()) {
                    lines.add(line);
                }
            }
        }
        return lines;
    }

    @Test
    void matchesGlslangOnRandomBlocks() throws Exception {
        List<String> lines = golden();
        assertEquals(450, lines.size());
        int members = 0;
        for (String line : lines) {
            String[] f = line.split("\\|");
            Rules rules = Rules.valueOf(f[0]);
            Block block = ShaderLayout.layout(rules, new Parser(f[1]).struct());
            String[] expected = f[3].split(";");
            List<Entry> got = block.entries();
            assertEquals(expected.length, got.size(), "member count " + f[1]);
            for (int k = 0; k < expected.length; k++) {
                Entry e = got.get(k);
                String actual = e.path() + "@" + e.offset() + "/" + e.arrayStride() + "/" + e.matrixStride();
                assertEquals(expected[k], actual, rules + " " + f[1]);
                members++;
            }
            assertEquals(Integer.parseInt(f[2]), block.size(), "block size " + rules + " " + f[1]);
        }
        assertTrue(members > 3000, "members " + members);
    }

    @Test
    void pathsStridesAndPadding() {
        Struct light = ShaderLayout.struct("Light", ShaderLayout.member("position", ShaderLayout.vec(3)),
                ShaderLayout.member("range", ShaderLayout.FLOAT), ShaderLayout.member("color", ShaderLayout.vec(3)));
        Struct globals = ShaderLayout.struct("Globals", ShaderLayout.member("viewProj", ShaderLayout.mat(4, 4)),
                ShaderLayout.member("lightCount", ShaderLayout.UINT),
                ShaderLayout.member("lights", ShaderLayout.array(light, 8)),
                ShaderLayout.member("weights", ShaderLayout.array(ShaderLayout.FLOAT, 256)));
        Block ubo = ShaderLayout.layout(Rules.STD140, globals);
        assertEquals(64, ubo.offset("lightCount"));
        assertEquals(80, ubo.offset("lights"));
        assertEquals(32, ubo.arrayStride("lights"));
        assertEquals(80 + 3 * 32 + 16, ubo.offset("lights[3].color"));
        // the morph-weights trap: a std140 float array has a 16-byte stride
        assertEquals(16, ubo.arrayStride("weights"));
        assertEquals(336 + 256 * 16, ubo.size(), "a trailing array counts its full stride, as SPIR-V reflection does");
        Block ssbo = ShaderLayout.layout(Rules.STD430, globals);
        assertEquals(4, ssbo.arrayStride("weights"));
        assertEquals(16, ssbo.matrixStride("viewProj"));
        assertThrows(IllegalArgumentException.class, () -> ubo.offset("lights[8]"));
        assertThrows(IllegalArgumentException.class, () -> ubo.offset("nothing"));
        assertThrows(IllegalArgumentException.class, () -> ubo.arrayStride("lightCount"));
        String glsl = ubo.toGlsl("uniform", "Globals", "g", 0);
        assertTrue(glsl.startsWith("struct Light {\n    vec3 position;\n    float range;\n    vec3 color;\n};\n"
                + "layout(std140, binding = 0) uniform Globals {\n    mat4x4 viewProj;\n"), glsl);
        assertTrue(glsl.contains("    Light lights[8];\n    float weights[256];\n} g;\n"), glsl);
        Struct bad = ShaderLayout.struct("Bad", ShaderLayout.member("m", ShaderLayout.rowMajor(ShaderLayout.mat(3, 3))));
        Block withBad = ShaderLayout.layout(Rules.STD430, ShaderLayout.struct("B", ShaderLayout.member("b", bad)));
        assertThrows(IllegalArgumentException.class, () -> withBad.toGlsl("buffer", "B", "b", 0));
        // float at 0, vec3 at 16, mat4 at 32: 96 bytes (formerly checked against StdLayout, removed in 3.0)
        assertEquals(96, ShaderLayout.layout(Rules.STD430, ShaderLayout.struct("S",
                ShaderLayout.member("a", ShaderLayout.FLOAT), ShaderLayout.member("b", ShaderLayout.vec(3)),
                ShaderLayout.member("c", ShaderLayout.mat(4, 4)))).paddedSize());
    }

    @Test
    void glslRejectsConflictingStructNamesAndBadArguments() {
        Struct a = ShaderLayout.struct("Light", ShaderLayout.member("color", ShaderLayout.vec(3)));
        Struct b = ShaderLayout.struct("Light", ShaderLayout.member("range", ShaderLayout.FLOAT));
        Block conflict = ShaderLayout.layout(Rules.STD430, ShaderLayout.struct("B",
                ShaderLayout.member("a", a), ShaderLayout.member("b", b)));
        // two different Light structs used to be declared once, silently giving member b the wrong type
        assertThrows(IllegalArgumentException.class, () -> conflict.toGlsl("buffer", "B", "b", 0));
        Struct outer = ShaderLayout.struct("Light", ShaderLayout.member("inner", b));
        Block nested = ShaderLayout.layout(Rules.STD430, ShaderLayout.struct("B", ShaderLayout.member("o", outer)));
        assertThrows(IllegalArgumentException.class, () -> nested.toGlsl("buffer", "B", "b", 0));
        // the same struct used twice is declared once
        Block twice = ShaderLayout.layout(Rules.STD430, ShaderLayout.struct("B",
                ShaderLayout.member("a", a), ShaderLayout.member("c", ShaderLayout.array(a, 2))));
        String glsl = twice.toGlsl("buffer", "B", "b", 1);
        assertEquals(glsl.indexOf("struct Light"), glsl.lastIndexOf("struct Light"), glsl);
        assertTrue(twice.toGlsl("buffer", "B", "", 1).endsWith("};\n"), "anonymous instance");
        assertThrows(IllegalArgumentException.class, () -> twice.toGlsl("ssbo", "B", "b", 0));
        assertThrows(IllegalArgumentException.class, () -> twice.toGlsl(null, "B", "b", 0));
        assertThrows(IllegalArgumentException.class, () -> twice.toGlsl("buffer", "", "b", 0));
        assertThrows(IllegalArgumentException.class, () -> twice.toGlsl("buffer", "B", null, 0));
        assertThrows(IllegalArgumentException.class, () -> twice.toGlsl("buffer", "B", "b", -1));
        assertThrows(IllegalArgumentException.class, () -> ShaderLayout.struct("", ShaderLayout.member("x", a)));
        assertThrows(IllegalArgumentException.class,
                () -> ShaderLayout.struct("S", ShaderLayout.member("x", a), null));
        assertThrows(IllegalArgumentException.class, () -> ShaderLayout.layout(null, a));
        assertThrows(IllegalArgumentException.class, () -> new ShaderLayout.Basic(null, 1, 1, false));
    }

    /** Local check, with -Ddynamisgpu.glslang=glslangValidator: every emitted declaration compiles. */
    @Test
    void emittedGlslCompiles() throws Exception {
        String tool = System.getProperty("dynamisgpu.glslang");
        if (tool == null) {
            return;
        }
        Path dir = Files.createTempDirectory("dynamisgpu-glsl");
        int n = 0;
        for (String line : golden()) {
            String[] f = line.split("\\|");
            Rules rules = Rules.valueOf(f[0]);
            Struct s = new Parser(f[1]).struct();
            String src = "#version 450\n" + ShaderLayout.layout(rules, s).toGlsl("buffer", "Block", "blk", 0)
                    + "layout(local_size_x = 1) in;\nvoid main() { }\n";
            if (!src.startsWith("#version 450\n#extension")) {
                src = src.replace("#version 450\n", "#version 450\n#extension GL_EXT_scalar_block_layout : enable\n");
            }
            Path p = dir.resolve("b" + n++ + ".comp");
            Files.writeString(p, src);
            Process proc = new ProcessBuilder(tool, "-V", "--target-env", "vulkan1.2", p.toString(), "-o",
                    dir.resolve("b.spv").toString()).redirectErrorStream(true).start();
            String out = new String(proc.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertEquals(0, proc.waitFor(), src + out);
        }
    }
}
