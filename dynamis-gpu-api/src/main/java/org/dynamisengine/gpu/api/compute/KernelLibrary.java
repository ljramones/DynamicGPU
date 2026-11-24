package org.dynamisengine.gpu.api.compute;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeSet;

/**
 * Compiled kernels on the class path, as written by the DynamisGPU shader build ({@code dynamis-gpu-shaders}): for
 * each kernel {@code name.spv} (Vulkan) and {@code name.metal} (Metal source), with their ABI in a manifest. The
 * library turns them into {@link KernelBundle}s and reports each kernel's entry point, group size, buffer slots and
 * parameter block, so the Java side of a kernel never restates them.
 * <p>
 * Bundles carry a {@link Backend#MOCK} executable holding the entry point's name, so test drivers that model kernels
 * in Java can create pipelines. A bundle's version is the start of the manifest's source hash.
 *
 * @since 1.2.0
 */
public final class KernelLibrary {
    private final Class<?> anchor;
    private final String directory;
    private final String namePrefix;
    private final Properties manifest = new Properties();

    private KernelLibrary(Class<?> anchor, String directory, String manifestName, String namePrefix) {
        this.anchor = anchor;
        this.directory = directory.endsWith("/") ? directory : directory + "/";
        this.namePrefix = namePrefix;
        try (InputStream in = open(manifestName)) {
            manifest.load(in);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * The library whose manifest is {@code directory/kernels.properties}, resolved like
     * {@link Class#getResourceAsStream} against {@code anchor} (relative to its package unless it starts with
     * {@code /}). Bundles are named after their kernels.
     */
    public static KernelLibrary load(Class<?> anchor, String directory) {
        return new KernelLibrary(anchor, directory, "kernels.properties", "");
    }

    /** As {@link #load(Class, String)}, with another manifest file name and a prefix for bundle names. */
    public static KernelLibrary load(Class<?> anchor, String directory, String manifestName, String namePrefix) {
        return new KernelLibrary(anchor, directory, manifestName, namePrefix);
    }

    private InputStream open(String file) {
        InputStream in = anchor.getResourceAsStream(directory + file);
        if (in == null) throw new IllegalStateException("kernel library resource missing: " + directory + file);
        return in;
    }

    private byte[] bytes(String file) {
        try (InputStream in = open(file)) {
            return in.readAllBytes();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private String get(String key) {
        String v = manifest.getProperty(key);
        if (v == null) throw new IllegalArgumentException("kernel library " + directory + " has no " + key);
        return v;
    }

    /** The names of the kernels in the library. */
    public Set<String> kernels() {
        Set<String> names = new TreeSet<>();
        for (String k : manifest.stringPropertyNames()) {
            if (k.endsWith(".entry")) names.add(k.substring(0, k.length() - ".entry".length()));
        }
        return Collections.unmodifiableSet(names);
    }

    /** The SHA-256 of the sources the library was built from ({@code source.sha256}). */
    public String sourceHash() { return get("source.sha256"); }

    /** The slangc version that built the library. */
    public String compilerVersion() { return get("slangc.version"); }

    /**
     * The kernel's executables: SPIR-V for Vulkan, Metal source for Metal, and the entry point for mocks. A kernel
     * built for one target only (its manifest has no {@code spirv.sha256} or {@code metal.sha256}) has no executable
     * for the other backend.
     */
    public KernelBundle bundle(String kernel) {
        String entry = entryPoint(kernel);
        Map<Backend, KernelBundle.Executable> ex = new EnumMap<>(Backend.class);
        if (manifest.containsKey(kernel + ".spirv.sha256"))
            ex.put(Backend.VULKAN, new KernelBundle.Executable(KernelBundle.Format.SPIRV, bytes(kernel + ".spv")));
        if (manifest.containsKey(kernel + ".metal.sha256"))
            ex.put(Backend.METAL, new KernelBundle.Executable(KernelBundle.Format.MSL_SOURCE, bytes(kernel + ".metal")));
        ex.put(Backend.MOCK, new KernelBundle.Executable(KernelBundle.Format.MOCK, entry.getBytes(StandardCharsets.UTF_8)));
        return new KernelBundle(namePrefix + kernel, sourceHash().substring(0, 16), ex);
    }

    /** The kernel's entry point. */
    public String entryPoint(String kernel) { return get(kernel + ".entry"); }

    /** The kernel's fixed group size. */
    public Dimensions localSize(String kernel) {
        String[] v = get(kernel + ".local").split(",");
        return new Dimensions(Integer.parseInt(v[0]), Integer.parseInt(v[1]), Integer.parseInt(v[2]));
    }

    /** The kernel's storage buffers by name, in slot order (slot = index). */
    public List<String> buffers(String kernel) {
        List<String> names = new ArrayList<>();
        String v = get(kernel + ".buffers");
        if (v.isEmpty()) return names;
        for (String b : v.split(",")) {
            String[] f = b.split(":");
            if (Integer.parseInt(f[1]) != names.size()) {
                throw new IllegalStateException(kernel + ": buffer slots are not 0, 1, 2, ...: " + v);
            }
            names.add(f[0]);
        }
        return Collections.unmodifiableList(names);
    }

    /** The parameter block's fields with their byte offsets, in declaration order. */
    public Map<String, Integer> parameterOffsets(String kernel) {
        Map<String, Integer> m = new LinkedHashMap<>();
        for (String[] f : fields(kernel)) m.put(f[0], Integer.parseInt(f[1]));
        return Collections.unmodifiableMap(m);
    }

    /** The parameter block's fields with their Slang scalar types ({@code uint32}, {@code float32}, ...). */
    public Map<String, String> parameterTypes(String kernel) {
        Map<String, String> m = new LinkedHashMap<>();
        for (String[] f : fields(kernel)) m.put(f[0], f[2]);
        return Collections.unmodifiableMap(m);
    }

    /** The size of the parameter block in bytes, for a block of 32-bit fields. */
    public int parameterBytes(String kernel) {
        int end = 0;
        for (String[] f : fields(kernel)) end = Math.max(end, Integer.parseInt(f[1]) + 4);
        return end;
    }

    /**
     * The kernel's specialization constants by name: their ids ({@code [vk::constant_id(n)]} in Slang, which is also
     * the Metal function-constant index) and scalar types ({@code uint32}, {@code int32}, {@code float32},
     * {@code bool}). Empty for a kernel without any. @since 1.7.0
     */
    public Map<String, Constant> constants(String kernel) {
        entryPoint(kernel);
        Map<String, Constant> m = new LinkedHashMap<>();
        String v = manifest.getProperty(kernel + ".constants", "");
        if (v.isEmpty()) return Map.of();
        for (String f : v.split(",")) {
            String[] p = f.split(":");
            m.put(p[0], new Constant(p[0], Integer.parseInt(p[1]), p[2]));
        }
        return Collections.unmodifiableMap(m);
    }

    /** A specialization constant. @since 1.7.0 */
    public record Constant(String name, int id, String type) {
        /** {@code value} as the constant's 32 bits: a {@link Number} for numeric types, a {@link Boolean} for bool. */
        public int bits(Object value) {
            return switch (type) {
                case "uint32" -> {
                    long v = ((Number) value).longValue();
                    if (v < 0 || v > 0xFFFF_FFFFL) throw new IllegalArgumentException(name + " does not fit uint32: " + v);
                    yield (int) v;
                }
                case "int32" -> Math.toIntExact(((Number) value).longValue());
                case "float32" -> Float.floatToRawIntBits(((Number) value).floatValue());
                case "bool" -> ((Boolean) value) ? 1 : 0;
                default -> throw new IllegalArgumentException(name + " has unsupported type " + type);
            };
        }
    }

    /**
     * The group-shared memory the kernel declares, from its SPIR-V (see {@code SpirvModule.workgroupBytes}); -1 for a
     * kernel built without SPIR-V. @since 1.7.0
     */
    public long groupSharedBytes(String kernel) {
        var spirv = bundle(kernel).executables().get(Backend.VULKAN);
        return spirv == null ? -1 : org.dynamisengine.gpu.api.compute.spi.SpirvModule.workgroupBytes(spirv.bytes());
    }

    /**
     * As {@link #pipeline(String, List, Set, Set)}, specialized: {@code constants} maps constant names to values
     * (see {@link Constant#bits}); constants left out keep their defaults. The pipeline compiles that variant of the
     * kernel (Vulkan specialization constants, Metal function constants). @since 1.7.0
     */
    public PipelineDescriptor pipeline(String kernel, List<Access> accesses, Set<String> dataTypes, Set<String> features,
            Map<String, ?> constants) {
        var declared = constants(kernel);
        Map<Integer, Integer> values = new java.util.TreeMap<>();
        for (var e : constants.entrySet()) {
            Constant c = declared.get(e.getKey());
            if (c == null) throw new IllegalArgumentException(kernel + " has no specialization constant " + e.getKey() + "; it has " + declared.keySet());
            values.put(c.id(), c.bits(e.getValue()));
        }
        var d = pipeline(kernel, accesses, dataTypes, features);
        return new PipelineDescriptor(d.kernel(), d.entryPoint(), d.layout(), d.localSize(), values, d.requiredDataTypes(), d.requiredFeatures());
    }

    private List<String[]> fields(String kernel) {
        List<String[]> list = new ArrayList<>();
        String v = get(kernel + ".params");
        if (v.isEmpty()) return list;
        for (String f : v.split(",")) list.add(f.split(":"));
        return list;
    }

    /**
     * A pipeline descriptor for the kernel: its bundle, entry point and group size, with one binding per buffer slot
     * at the given accesses (minimum range one 32-bit element) and its parameter block.
     */
    public PipelineDescriptor pipeline(String kernel, List<Access> accesses, Set<String> dataTypes, Set<String> features) {
        int slots = buffers(kernel).size();
        if (accesses.size() != slots) {
            throw new IllegalArgumentException(kernel + " has " + slots + " buffers, got " + accesses.size() + " accesses");
        }
        List<BindingLayout.BufferBinding> b = new ArrayList<>();
        for (int i = 0; i < slots; i++) b.add(new BindingLayout.BufferBinding(i, accesses.get(i), 4));
        return new PipelineDescriptor(bundle(kernel), entryPoint(kernel), new BindingLayout(b, parameterBytes(kernel)),
                localSize(kernel), Map.of(), dataTypes, features);
    }
}
